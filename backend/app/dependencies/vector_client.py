#app.dependencies.vector_client.py
from fastapi import Request
from app.clients.base_client import IVectorClient
from app.core.logger import log

def vector_client_dependency(request: Request) -> IVectorClient:
    """
    FastAPI dependency to access the vector client from app state.
    """
    vector_client = getattr(request.app.state, "vector", None)
    if vector_client is None:
        log.error("!!! Vector client not found in app state")
        raise RuntimeError("Vector client not initialized")
    
    if not isinstance(vector_client, IVectorClient):
        log.error("!!! Vector client in app.state is not an ILLMClient instance")
        raise RuntimeError("Vector client has invalid type")

    return vector_client