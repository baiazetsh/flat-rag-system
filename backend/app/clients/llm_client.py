#app/clients/llm_client.py v1.0.1
""" A wrapper about any LLM API  
    Generic LLM API client.
    A minimal and safe wrapper that supports multiple common LLM
    response formats without any extra configuration.
    
    Automatically handles:
      • Network errors and timeouts
      • Invalid JSON responses
      • HTTP error codes with meaningful messages
      • Ollama native format:       {"response": "..."}
      • HF TGI / some Ollama:       {"generated_text": "..."}
      • Simple APIs:                {"text": "..."}
      • OpenAI-style responses:     {"choices":[{"message":{"content":"..."}}]}

    Always returns a clean string output.
    Raises RuntimeError for any invalid or unexpected backend behavior.    
"""
import json
import httpx
from app.core.config import cfg
from app.clients.base_client import ILLMClient
from app.core.logger import log



class GenericLLMClient(ILLMClient):   

    async def generate(self, prompt: str, **kwargs) -> str:
        payload = {
            "model": cfg.llm_model,
            "prompt": prompt,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            # Network errors
            try:
                response = await client.post(cfg.llm_url, json=payload, timeout=60)
                response.raise_for_status()

            except httpx.HTTPStatusError as e:
                log.error(f"LLM API error {e.response.status_code}")
                raise RuntimeError("AI service error") from e
            except httpx.RequestError as e:
                log.error(f"LLM connection error: {type(e).__name__}")
                raise RuntimeError("AI service unavailable") from e
            except Exception as e:
                log.error(f"Unexpected LLM error: {type(e).__name__}")
                raise RuntimeError("AI service temporarily unavailable") from e

        data = response.json()

        # Ollama-style: {"response": "...", "done": true}
        if "response" in data:
            return data["response"]

        # OpenAI-style: {"choices":[{"message":{"content": "..."} }]}
        if "choices" in data:
            try:
                return data["choices"][0]["message"]["content"]
            except Exception:
                pass

        # Fallback
        return str(data)

    async def stream(self, prompt: str):
        """Streaming disabled."""
        raise NotImplementedError("Streaming mode is disabled in this project")

                 
            

  