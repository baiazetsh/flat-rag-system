#rag_local/backend/app/routes/base.py
import uuid
import logging
import httpx
from fastapi import APIRouter
from qdrant_client.http import models as qmodels
from fastapi import UploadFile, File, Form, HTTPException, Request
from starlette.requests import Request as StarletteRequest
from fastapi.templating import Jinja2Templates

from app.clients import cfg, qdrant, http_client
from app.utils import chunk_text_by_sentences
from app.services.rag_services import generate_rag_answer, _get_embedding
from app.models import (
    AskRequest,
    EmbedRequest,
    SearchRequest,
    RAGRequest
)

# ========= Logger setup =========
log = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "RAG backend API",
        "version": "1.0.0",
        "status": "OK"
        }

@router.get("/health")
async def health_check():
    """Check if all services are available"""
    try:
        # Check Qdrant
        collections = qdrant.get_collections()
        qdrant_status = "healthy"
    except Exception as e:
        qdrant_status = f"unhealthy: {str(e)}"

    try:
        # Check Ollama
        resp = await http_client.get(f"{cfg.ollama.base_url}/api/tags", timeout=5.0)
        ollama_status = "healthy" if resp.status_code == 200 else "unhealthy"
    except Exception as e:
        ollama_status = f"unhealthy: {str(e)}"
    
    return {
        "qdrant": qdrant_status,
        "ollama": ollama_status,
        "collection": cfg.qdrant.collection
    } 
  


@router.post("/ask")
async def ask_ollama(request: AskRequest):
    """Send raw prompt to LLM without RAG. """      
    try:  
        data = {
            "model": cfg.ollama.ollama_model,
            "prompt": request.prompt,
            "stream": False
        }        
        response = await http_client.post(
            f"{cfg.ollama.base_url}/api/generate",
            json=data,
            timeout=120.0
            )            
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Ollama  error: {response.text}"
                )        
        result = response.json()
        return {
            "answer": result.get("response", "No answer"),
            "model": cfg.ollama.llm_model
            }
    except Exception as e:
        log.error(f"Ask endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    
@router.post("/embed")    
async def embed_text(request: EmbedRequest):
    """Embed text and store in Qdrant"""  
    try:
        embedding, elapsed_embedding = await _get_embedding(request.text)

        # store in Qdrant =========
        point = qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector = embedding,
            payload = {
                "text": request.text,
                "source": "api_embed"
            }
        )
        qdrant.upsert(
            collection_name=cfg.qdrant.collection,
            points=[point]
        )        
        return {
            "message": "Vector saved to Qdrant",
            "vector_dim": len(embedding),
            "text_preview": request.text[:100],
            "collection": cfg.qdrant.collection,
            "embedding_time_ms": round(elapsed_embedding, 1)
        }
    except Exception as e:
        log.error(f"Embed endpoint error: {str(e)}")            
        raise HTTPException(status_code=500, detail=str(e))
            
        
@router.post("/search")        
async def search_text(request: SearchRequest):
    """Semantic search in Qdrant without LLM generation"""
    try:
        query_vector, elapsed_search = await _get_embedding(request.query)    

        # Search in Qdrant =========
        search_result = qdrant.search(
        collection_name=cfg.qdrant.collection,
        query_vector=query_vector,
        limit=request.top_k,
        with_payload=True,
        score_threshold=cfg.qdrant.score_threshold,
        )
        results = [
            {"text": hit.payload.get("text", ""),
            "score": round(hit.score, 4),
            "source": hit.payload.get("source", "unknown")
            }
            for hit in search_result
        ]
        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "embedding_time_ms": round(elapsed_search)
            }
    except Exception as e:
        log.error(f"Search endpoint error: {str(e)}")    
        raise HTTPException(status_code=500, detail=str(e))
        

@router.post("/upload_docs")
async def upload_docs(
    file: UploadFile = File(...),
    collection: str = Form('docs'),
    chunk_size: int = Form(500),
    overlap: int = Form(1)
    ):
    """
    Upload .txt / .md/ .json and index content into the Qdrant.
    Support chunking with configurate size and overlap.
    """   
    collection_name = collection

    # Check for collection exists
    collections = [c.name for c in qdrant.get_collections().collections]
    if collection_name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    
    try:
        # Validate file type ===========
        allowed_extensions = {".txt", ".md", ".json"}
        file_ext = "." + file.filename.split(".")[-1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Use: {allowed_extensions}"
            )

        # read file content ======
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")

        if not text.strip():
            raise HTTPException(status_code=400, detail="File is empty")

        # chunk text ========
        chunks =chunk_text_by_sentences(
            text,
            max_chunk_size=chunk_size,
            overlap_sentences=overlap
        )        
        log.info(f" Uploading '{file.filename}' -> {len(chunks)} chunks total")

        # Process chunks
        stored = 0   
        failed = 0

        for idx,  chunk in enumerate(chunks, 1):
            try:
                embedding, _ = await _get_embedding(chunk)

                # Store in Qdrant
                point = qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": file.filename,
                    "chunk_index": idx,
                    "collection": collection_name
                    }
                )
                qdrant.upsert(
                    collection_name=collection_name,
                    points=[point]
                )                
                stored += 1

                if idx % 10 == 0 or idx == len(chunks):
                    log.info(f" Indexed {idx}/{len(chunks)} chunks")

            except Exception as e:
                log.warning(f"!!! Embedding failed for chunk {idx}: {str(e)}")
                failed += 1
                continue

        return { 
            "message": f"Successfully uploaded '{file.filename}'",
            "filename": file.filename,
            "total_chunks": len(chunks),
            "chunks_indexed": stored,
            "chunks_failed": failed,
            "collection": collection_name,
            "chunk_size": chunk_size
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Uload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        

@router.post("/search_with_llm")        
async def search_with_llm(request: RAGRequest):
    """
    Full RAG pipeline: search +LLM generation.
    This is the main endpoint for answering questions.
    """
    try:
        # Call the main RAG service
        result = await generate_rag_answer(
            query=request.query,
            top_k=request.top_k,
            collection=request.collection
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"RAG endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    """    
        if "application/json" in request.__dict__.get("_headers", {}).get("content-type", ""):
            return result

        # 🟩 If request comes from HTML form (browser), render template
        
        templates = Jinja2Templates(directory="app/templates")

        # 🟩 Create fake Request object if needed (for template rendering)
        if not isinstance(request, Request):
            
            request = StarletteRequest(scope={"type": "http"})

        # 🟩 Render the result in HTML template
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "result": result
            }
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"RAG endpoint error: {str(e)}")
    """

@router.post("/create_collection")
async def create_collection(
    name: str = Form(...),
    vector_size: int = Form(768)
):
    """
    Create a ne Qdrant collection from web or API call.
    """
    collections  = [c.name for c in qdrant.get_collections().collections]
    if name in collections:
        return{
            "status": "exists",
            "message": f"Colection: '{name}' already exists."}
    
    qdrant.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE
        )
    )
    return {
        "status": "created", 
        "message": f"Collection: '{name}' created successfully!"
    }
