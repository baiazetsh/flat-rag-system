#app/clients/embedding_client.py v1.0.1
""" A wrapper about any embedding API """ 

import httpx
from app.core.config import cfg
from app.clients.base_client import IEmbeddingClient
from app.core.logger import log

class GenericEmbeddingClient(IEmbeddingClient):
    async def embed(self, text: str) -> list[float]:
        payload ={
            "model": cfg.embedding_model,
            "prompt": text,
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    cfg.embedding_url,
                    json=payload,
                    timeout=30
                ) 
                response.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Embedding request failed: {e}")
            
            # If LLM returned error JSON
            try:
                data = response.json()
            except Exception:
                raise RuntimeError(f"Invalid JSON from embedding endpoint: {response.text}")
            
            # HTTP errors
            if response.status_code != 200:
                msg = data.get("error" or f"HTTP {response.status_code}")
                raise RuntimeError(f"Embedding service error: {msg}")
            

            # NEW OLLAMA FORMAT
            if "embeddings" in data:
                emb = data["embeddings"]
                if isinstance(emb, list) and emb and isinstance(emb[0], list):
                    return emb[0]

            # OLD FORMAT
            if "embedding" in data:
                emb = data["embedding"]
                if isinstance(emb, list):
                    return emb

            raise RuntimeError(f"Invalid embedding response: {data}")
