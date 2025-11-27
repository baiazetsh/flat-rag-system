#app/clients/vector_clients.py v1.0.1
""" Work with vector base (Qdrant, Milvus, Pinecone, Chroma)"""
from qdrant_client import QdrantClient, models
from app.clients.base_client import IVectorClient
from app.core.config import cfg
from app.core.logger import log
import asyncio
#from qdrant_client.models import PointStruct


class QdrantVectorClient(IVectorClient):
    def __init__(self):
        self.client = QdrantClient(url=cfg.qdrant_url)

    
    # === SEARCH ===
    async def search(
            self,
            collection: str,
            vector: list[float],
            top_k: int,
            score_threshold: float | None = None,
        ):       
        # Execute Qdrant search        
        results = await asyncio.to_thread(
            self.client.search,
            collection_name=collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold,
        )
        #output
        output = []
        for hit in results:
            payload = hit.payload or {}

            output.append({            
                "text": payload.get("text"),
                "score": hit.score,
                "source": payload.get("source"),
                "chunk_index": payload.get("chunk_index"),
                "uploaded_at": payload.get("uploaded_at"),
                "file_size": payload.get("file_size"),
                "chunk_size": payload.get("chunk_size"),
            })
        return output

    # === LIST COLLECTIONS ===
    async def list_collections(self):
        collections = await asyncio.to_thread(self.client.get_collections)
        return [c.name for c in collections.collections]

    # === CREATE COLLECTION ===
    async def create_collection(self, name: str, vector_size: int, distance: str):
        await asyncio.to_thread(
            self.client.create_collection,
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance[distance.upper()],
            ),
        )

    # === RECREATE COLLECTION ===# Vector DB interface
    async def recreate_collection(self, name: str, vector_size: int, distance: str) -> None:
        """Delete and recreate collection."""
        try:
            await self.delete_collection(name)
            log.info(f"Deleted collection '{name}'")
        except Exception as e:
            log.warning(f"Collection '{name}' doesn't exist or couldn't be deleted: {e}")
        
        await self.create_collection(name, vector_size, distance)
        log.info(f"Recreated collection '{name}'")


    # === UPSERT ===
    async def upsert(self, collection_name: str, points: list[dict]):
       await asyncio.to_thread(
            self.client.upsert,
            collection_name=collection_name,
            points=points,
        )

    # === DELETE COLLECTION ===
    async def delete_collection(self, name: str):
        await asyncio.to_thread(self.client.delete_collection, collection_name=name)


    # === GET VECTORS ===
    async def get_vectors(
        self, collection: str,
        limit: int,
    ) -> list[list[float]]:    
        """Fetch up to `limit` vectors 
        from a collection for analysis/visualization.
        """   
        result, _ = await asyncio.to_thread(
            self.client.scroll,
            collection_name=collection,
            limit=limit,
            with_vectors=True,
        )

        return [point.vector for point in result if point.vector]
