#app/services/llm_service.py v1.0.1
"""Handles LLM text generation through provider defined in ClientFactory."""
from fastapi import Request
from app.core.logger import log
from app.clients.base_client import ILLMClient

async def generate_answer(
        llm_client: ILLMClient,
        prompt: str,        
) -> str:
    """Generate response using LLM provider."""
    #if not llm_client:
        #raise RuntimeError("LLM client not initialized.")
    response = await llm_client.generate(prompt)
    log.info(f"LLM response ({len(response)} chars).")
    return response
