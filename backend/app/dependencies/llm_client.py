#app.dependencies.llm_client.py
from fastapi import Request
from app.clients.base_client import ILLMClient
from app.core.logger import log

def llm_client_dependency(request: Request) -> ILLMClient:
    """
    FastAPI dependency to access the llm client from app state.
    """
    llm_client = getattr(request.app.state, "llm", None)    
    if  llm_client is None:
        log.error("!!! LLM client not found in app state")
        raise RuntimeError("LLM client not initialized")
    
    return llm_client




