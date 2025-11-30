# app.dependencies/rerank_client.py
from fastapi import Request
from app.clients.base_client import IRerankerClient
from app.core.logger import log

def reranker_client_dependency(request: Request) -> IRerankerClient:
    """FastAPI dependency to access the reranker client from app state."""
    reranker_client = getattr(request.app.state, "reranker", None)
    if reranker_client is None:
        log.error(f"!!! Reranker client not found in app state")
        raise RuntimeError("Reranker client missing")
    return reranker_client




    