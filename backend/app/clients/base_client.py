#app/clients/base_clients.py v1.0.1
"""
Base interfaces for RAG backend components:

- IEmbeddingClient  (returns embeddings)
- ILLMClient        (generates text)
- IVectorClient     (vector database operations)
"""

from abc import ABC, abstractmethod
from typing import Any

# Embedding interface
class IEmbeddingClient(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return embedding vector for given text"""
        raise NotImplementedError

# LLM interface
class ILLMClient(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text based on a prompt"""
        raise NotImplementedError

# Vector DB interface
class IVectorClient(ABC):
    """Async interface for all vector database clients."""
    client: Any  # underlying native client (e.g., QdrantClient, chromadb.Client)
    @abstractmethod
    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        score_threshold: float | None = None,
        ) -> list[dict]:
        """Search vector DB"""
        raise NotImplementedError

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """Return list of available collections in vector DB."""
        raise NotImplementedError
    
    @abstractmethod
    async def create_collection(
        self,
        name: str,
        vector_size: int,
        distance: str) -> None:
        """Create a new collection if not exists."""
        raise NotImplementedError       

    @abstractmethod
    async def recreate_collection(
        self,
        name: str,
        vector_size: int,
        distance: str) -> None:
        """Create a new collection if not exists."""
        raise NotImplementedError      
    
    @abstractmethod
    async def delete_collection(self, name: str) -> None:
        """Delete existing collection."""
        raise NotImplementedError

    @abstractmethod
    async def upsert(self, collection_name: str, points: list[dict]) -> None:
        """Insert or update vectors in a collection."""
        raise NotImplementedError
    
    
    @abstractmethod
    async def get_vectors(
        self,
        collection: str,
        limit: int,
    ) -> list[list[float]]:
        pass

    async def aclose(self) -> None:
        """Optional async cleanup method (default no-op)."""
        return None
    