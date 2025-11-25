#app.dependencies.embedding_client.py
from fastapi import Request
from app.clients.base_client import IEmbeddingClient
from app.core.logger import log

def embedding_client_dependency(request: Request) -> IEmbeddingClient:
    """
    FastAPI dependency to access the embedding client from app state.
    """
    embedding_client = getattr(request.app.state, "embedding", None)    
    if  embedding_client is None:
        log.error("!!! Embedding client not found in app state")
        raise RuntimeError("Embedding client not initialized")
    
    if not isinstance(embedding_client, IEmbeddingClient):
        log.error("!!! Embedding client in app.state is not an ILLMClient instance")
        raise RuntimeError("Embedding client has invalid type")

    
    return embedding_client


    