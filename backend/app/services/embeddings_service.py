#app/services/embeddings_service.py v1.0.1
"""Handles embedding generation via configured embedding provider."""

from app.core.logger import log
from app.clients.base_client import IEmbeddingClient


async def get_embedding(
    text: str,
    embedding_client: IEmbeddingClient,
) -> list[float]:
    """Request embedding from provider defined in ClientFactory."""
        
    vector = await embedding_client.embed(text)
    log.info(f"Embedding generated ({len(vector)} dims, {len(text)} chars).")

    return vector