# app/routes/health.py v1.0.1
"""
Health check endpoints for RAG backend components:
- Vector DB (Qdrant, Milvus, etc.)
- Embedding provider
- LLM provider
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.core.logger import log
from app.core.config import cfg
from app.dependencies.vector_client import vector_client_dependency
from app.dependencies.embedding_client import embedding_client_dependency
from app.dependencies.llm_client import llm_client_dependency
from app.clients.base_client import IEmbeddingClient, ILLMClient, IVectorClient
import datetime

router = APIRouter(prefix="/api/health", tags=["Health"])
templates = Jinja2Templates(directory="app/templates")

# MAIN HEALTH JSON: /api/health
@router.get("/")
async def health_check(
    request: Request,
    vector_client = Depends(vector_client_dependency),
    embedding_client = Depends(embedding_client_dependency),
    llm_client = Depends(llm_client_dependency),
):
    """
     Simple JSON health summary.
    """    
    status: dict = {
        "app_name": cfg.app_name,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }    
    degraded: list[str] = []

    # Vector backend
    try:         
        collections = await vector_client.list_collections()
        status["vector_backend"] = {
            "backend": cfg.vector_backend,
            "url": cfg.qdrant_url,
            "collections": collections,
            "status": "ok"
        }       
    except Exception as e:
        log.warning(f"⚠ Vector backend unavailable: {e}")
        degraded.append("vector")
        status["vector_backend"] = {
            "backend": cfg.vector_backend,
            "url": cfg.qdrant_url,
            "status": "error",
            "detail": str(e),
        }

    # Embedding provider
    try:    
        test_vec = await embedding_client.embed("health probe")

        if not isinstance(test_vec, list) or len(test_vec) == 0:
            raise ValueError("Invalid embedding vector: empty or non-list")

        if (not isinstance(test_vec, list)) or len(test_vec) == 0:
            raise ValueError("!!! Empty embedding vector")        
        
        if not all(isinstance(x, (int, float)) for x in test_vec[:5]):
            raise ValueError("! Invalid embedding vector: non-numeric values detected")

        status["embedding_provider"] = {
            "provider": cfg.embedding_provider,
            "model": cfg.embedding_model,
            "url": cfg.embedding_url,
            "dimension": len(test_vec),
            "status": "ok",
        }        
    except Exception as e:
        log.warning(f"⚠ Embedding provider unavailable: {e}")
        degraded.append("embedding")
        status["embedding_provider"] = {
            "provider": cfg.embedding_provider,
            "model": cfg.embedding_model,
            "status": "error",
            "detail": str(e),
        }

    # LLM provider
    try:       
        response = await llm_client.generate("hi", stream = False)

        #if not isinstance(response, dict):
         #   raise ValueError(f"! Invalid LLM response type: {type(response)}")
        
        #text = response.get("answer") or response.get("response") or ""        

        if not isinstance(response, str):
            raise ValueError(f"! Invalid LLM response type: {type(response)}")

        status["llm_provider"] = {
            "provider": cfg.llm_provider,
            "model": cfg.llm_model,
            "url": cfg.llm_url,
            "status": "ok",
        }            
    except Exception as e:
        log.warning(f"⚠ LLM provider unavailable: {e}")
        degraded.append("llm")
        status["llm_provider"] = {
            "provider": cfg.llm_provider,
            "model": cfg.llm_model,
            "status": "error",
            "detail": str(e)
        }

    # final status
    status["overall_status"] = ("ok" if not degraded else f"degraded")
    status["degraded_services"] = degraded if degraded else None

    return status

# HTML UI: /api/health/ui
@router.get("/ui", response_class=HTMLResponse)
async def health_check_ui(request: Request):
    """Render health check UI page."""
    return templates.TemplateResponse(
        "health.html",
        {"request": request}
    )


