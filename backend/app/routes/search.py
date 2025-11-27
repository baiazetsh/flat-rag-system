# app/routes/search.py v 1.0.1
"""
Semantic search endpoint — retrieves context chunks without LLM generation.
"""
from fastapi import APIRouter, HTTPException, Request, Depends
from app.models import SearchRequest
from app.core.logger import log
from app.core.config import cfg
from app.services.embeddings_service import get_embedding
from app.services.vector_services  import search_in_vector_db
from app.dependencies.vector_client import vector_client_dependency
from app.dependencies.embedding_client import embedding_client_dependency
from app.clients.base_client import IVectorClient, IEmbeddingClient


router = APIRouter(prefix="/api", tags=["Semantic Search"])

@router.post("/search")
async def search_text(   
    body: SearchRequest,
    vector_client: IVectorClient = Depends(vector_client_dependency),
    embedding_client: IEmbeddingClient = Depends(embedding_client_dependency),
):
    """
    Semantic search in vector DB (Qdrant, Milvus, etc.) without LLM generation.
    """    
    try:
        # === 1. Getting embedding ===
        embedding_vector = await get_embedding(body.query, embedding_client)
        log.info(f"Embedding computed ")

        # === 2. Search ===
        collection = body.collection or cfg.default_collection_name
        top_k = body.top_k or cfg.top_k

        results = await search_in_vector_db(    
            vector_client=vector_client,        
            vector=embedding_vector,
            top_k=top_k,
            collection=collection,
        )

        # === 3. Make answer ===
        response = {
            "query": body.query,
            "collection": collection,
            "results": results,
            "count": len(results),
            #"timing_ms": {
                #"embedding": round(elapsed_embedding, 1)
            #},
        }
        return response

    except Exception as e:
        log.error(f"Search endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
