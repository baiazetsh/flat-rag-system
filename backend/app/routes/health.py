# app/routes/health.py - OPTIMIZED with caching

from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.core.logger import log
from app.core.config import cfg
from app.dependencies.vector_client import vector_client_dependency
from app.dependencies.embedding_client import embedding_client_dependency
from app.dependencies.llm_client import llm_client_dependency
from app.dependencies.reranker_client import reranker_client_dependency
from app.clients.base_client import (
    IEmbeddingClient,
    ILLMClient,
    IVectorClient,
    IRerankerClient,
)
import datetime
from functools import lru_cache
import time

router = APIRouter(prefix="/api/health", tags=["Health"])
templates = Jinja2Templates(directory="app/templates")

# === CACHE CONFIG ===
CACHE_TTL_SECONDS = 30  # кеш на 30 секунд

_health_cache = {
    "timestamp": 0,
    "data": None
}

@router.get("/")
async def health_check(
    request: Request,
    vector_client = Depends(vector_client_dependency),
    embedding_client = Depends(embedding_client_dependency),
    llm_client = Depends(llm_client_dependency),
    reranker_client = Depends(reranker_client_dependency),
):
    """
    Health check with 30-second cache to reduce reranker load.
    """
    global _health_cache
    
    # Проверка кеша
    now = time.time()
    if _health_cache["data"] and (now - _health_cache["timestamp"]) < CACHE_TTL_SECONDS:
        log.info("🔄 Health check: returning cached result")
        return _health_cache["data"]
    
    # Полная проверка
    log.info("🔍 Health check: running full checks")
    
    status = {
        "app_name": cfg.app_name,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    degraded = []

    # Vector backend (fast, no cache needed)
    try:
        collections = await vector_client.list_collections()
        status["vector_backend"] = {
            "backend": cfg.vector_backend,
            "url": cfg.qdrant_url,
            "collections": collections,
            "status": "ok"
        }
    except Exception as e:
        log.warning(f"⚠️ Vector backend unavailable: {e}")
        degraded.append("vector")
        status["vector_backend"] = {
            "backend": cfg.vector_backend,
            "status": "error",
            "detail": str(e),
        }

    # Embedding provider
    try:
        test_vec = await embedding_client.embed("health probe")
        if not isinstance(test_vec, list) or len(test_vec) == 0:
            raise ValueError("Invalid embedding vector")
        
        status["embedding_provider"] = {
            "provider": cfg.embedding_provider,
            "model": cfg.embedding_model,
            "dimension": len(test_vec),
            "status": "ok",
        }
    except Exception as e:
        log.warning(f"⚠️ Embedding provider unavailable: {e}")
        degraded.append("embedding")
        status["embedding_provider"] = {
            "provider": cfg.embedding_provider,
            "status": "error",
            "detail": str(e),
        }

    # LLM provider
    try:
        response = await llm_client.generate("hi", stream=False)
        if not isinstance(response, str):
            raise ValueError(f"Invalid LLM response type: {type(response)}")

        status["llm_provider"] = {
            "provider": cfg.llm_provider,
            "model": cfg.llm_model,
            "status": "ok",
        }
    except Exception as e:
        log.warning(f"⚠️ LLM provider unavailable: {e}")
        degraded.append("llm")
        status["llm_provider"] = {
            "provider": cfg.llm_provider,
            "status": "error",
            "detail": str(e)
        }

    # Reranker provider (SLOW - cache important!)
    try:
        scores = await reranker_client.rerank(
            query="test",
            docs=["Hello user"]
        )
        if not isinstance(scores, list) or len(scores) != 1:
            raise ValueError("Invalid score output")

        status["reranker_provider"] = {
            "provider": cfg.reranker_provider,
            "model": cfg.reranker_model,
            "status": "ok",
        }
    except Exception as e:
        log.warning(f"⚠️ Reranker provider unavailable: {e}")
        degraded.append("reranker")
        status["reranker_provider"] = {
            "provider": cfg.reranker_provider,
            "status": "error",
            "detail": str(e)
        }

    # Final status
    status["overall_status"] = "ok" if not degraded else "degraded"
    status["degraded_services"] = degraded if degraded else None
    
    # Обновляем кеш
    _health_cache["timestamp"] = now
    _health_cache["data"] = status
    
    return status


@router.get("/ui", response_class=HTMLResponse)
async def health_check_ui(request: Request):
    """Render health check UI page."""
    return templates.TemplateResponse(
        "health.html",
        {"request": request}
    )