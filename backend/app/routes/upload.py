# app/routes/upload.py v1.0.1
"""Upload a text data, chunking"""
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
    Depends,
)
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import uuid
from app.dependencies.splitter_factory import get_splitter
from app.services.smart_text_splitter_service import SmartTextSplitter
from app.dependencies.embedding_client import embedding_client_dependency
from app.dependencies.vector_client import vector_client_dependency
from app.services.vector_services import upsert_vectors
from app.core.config import cfg
from app.core.logger import log

router = APIRouter(prefix="/upload", tags=["Upload"])
templates = Jinja2Templates(directory="app/templates")

# render upload form
@router.get("/upload_docs/form")
async def upload_for(request: Request):
    """Render upload form with default from cfg."""
    return templates.TemplateResponse(
        "upload_form.html",
        {
            "request": request,
            "collection": cfg.default_collection_name,
            "overlap": cfg.overlap_sentences,
            "max_chunks": cfg.max_chunks,            
            "embedding_model": cfg.embedding_model,
            "llm_model": cfg.llm_model,
        },
    )

# main upload handler
@router.post("/upload_docs")
async def upload_docs(
    request: Request,
    file: UploadFile = File(...),
    collection: str = Form(cfg.default_collection_name),
    chunk_size: int = Form(cfg.max_chunk_size),
    overlap: int = Form(cfg.overlap_sentences),
    max_chunks: int =Form(cfg.max_chunks),

    splitter: SmartTextSplitter = Depends(get_splitter),
    embedding_client = Depends(embedding_client_dependency),
    vector_client = Depends(vector_client_dependency),
):
    """Upload and process document - chunk -> embed -. store."""
    try:
        #read & decode file
        content = await file.read()
        text = content.decode("utf-8", errors="ignore").strip()
        if not text:
            raise HTTPException(status_code=400, detail="File is empty")        
        
        # run splitter
        chunks_raw = await splitter.split(text)
        if not chunks_raw:
            raise HTTPException(status_code=400, detail="No chunks produced")        
        log.info(f"Splitter created {len(chunks_raw)} chunks from {file.filename}")

        stats = splitter.get_stats(chunks_raw)
        log.info(f"Chunks stats: {stats}")

        if chunks_raw and isinstance(chunks_raw[0], tuple):
            chunks = [chunk_text for chunk_text, _ in chunks_raw]
        else:
            chunks = chunks_raw
       
        # embedding & upsert
        payloads = []
        vectors = []
        for idx, chunk in enumerate(chunks):
            emb = await embedding_client.embed(chunk)

            payloads.append({
                "id": str(uuid.uuid4()),
                "vector": emb,                
                "text": chunk,
                "source":file.filename,
                "chunk_index": idx,
                "collection": collection,                
            })        
            vectors.append(emb)    

        await upsert_vectors(
            vector_client,
            collection,
            payloads,
            vectors,
        )
        log.info(f"Uploaded {len(chunks)} chunks into collection: '{collection}'")

        result = {
            "filename": file.filename,
            "total_chunks": len(chunks),
            "collection": collection,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "max_chunks": max_chunks,
            "stats": stats,
        }
        # if browser form - render Html
        if "text/html" in request.headers.get("accept", ""):
            return templates.TemplateResponse(
                "upload_result.html",
                {"request": request, **result},
            )
        # Otherwise -> return JSON
        return JSONResponse(content=result)
    
    except Exception as e:
        log.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))