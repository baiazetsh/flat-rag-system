#app/services/vector_services.py  v1.0.1
"""
Vector database service layer
==============================

This module contains business logic for interacting with any
vector database backend (Qdrant, Milvus, Chroma, Weaviate, etc.)
through a unified IVectorClient interface.

IMPORTANT:
- NO FastAPI dependencies.
- NO Depends().
- NO request or app objects.
- Pure service functions receiving `vector_client` explicitly.
"""

from app.core.config import cfg
from app.core.logger import log
from app.clients.base_client import IVectorClient as IVectorClient
#from app.clients.client_factory import ClientFactory



#==================================================
async def search_in_vector_db(
        vector_client: IVectorClient,
        vector: list[float],
        top_k: int, 
        collection: str,
        score_threshold: float | None = None,
    ):
    """Search for similar vectors."""       

    results = await vector_client.search(
        collection=collection,
        vector=vector,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    
    return results

#==================================================
async def create_collection(
    vector_client: IVectorClient,
    name: str,
    vector_size: int | None = None,
    distance:str | None = None
) ->dict:
    """
    Create a new collection if it doesn't exist.    
    """      

    size = vector_size or cfg.collection.size
    dist = distance or cfg.collection.distance

    collections = await vector_client.list_collections()
    if name in collections:
        log.warning(f"!!! Collection'{name}' already exists.")
        return {"status": "exists", "collection": name}

    await vector_client.create_collection(name, size, dist)
    log.info(f" ✅ Created collection: {name}")
    return {
        "status": "created",
        'collection': name,
        "size": size,
        "distance": dist,
    }


#==================================================
async def delete_collection(
        vector_client: IVectorClient,       
        name: str,
        ) -> dict:
    """Delete a collection if present."""

    collections = await vector_client.list_collections()
    if name not in collections:
        log.warning(f"⚠️ Collection '{name}' not found.")
        return {"status": "not_found", "collection": name}

    await vector_client.delete_collection(name)
    log.info(f"🗑️ Deleted collection '{name}'")

    return {"status": "deleted", "collection": name}


#==================================================
async def recreate_collection(
        vector_client: IVectorClient,
          name: str,
          vector_size=None,
          distance=None,
):
    """Drop and recreate a collection (dev utility)."""

    await delete_collection(vector_client, name)
    return await create_collection(
        vector_client=vector_client,
        name=name,
        vector_size=vector_size,
        distance=distance,
    )

#==================================================
async def list_collections(
        vector_client: IVectorClient,
) -> list[str]:    
    """Return a list of collection names."""
    return await vector_client.list_collections()



#==================================================
async def upsert_vectors(
        vector_client: IVectorClient,
        collection: str,
        payloads: list[dict],
        vectors: list[list[float]]
) -> dict:
    """
    Generic upsert — backend-agnostic (works with any vector DB client).
    points = [{"id": "...", "vector": [...], "payload": {...}}, ...]
    """    

    if len(payloads) != len(vectors):
        raise ValueError(
    f"Payloads/vectors length mismatch: {len(payloads)} vs {len(vectors)}"
)

    # Packing universal points
    points = [
        {
            "id": p.get("chunk_index") or i,
            "vector": vectors[i],
            "payload": p,
        }
        for i, p in enumerate(payloads)                        
    ]
    try:
        await vector_client.upsert(collection_name=collection, points=points)
        log.info(f"📥 Upserted {len(points)} vectors into '{collection}'")
        return {"status": "ok", "count": len(points)}
    except Exception as e:
        log.error(f"Vector upsert failed: {e}")
        raise     
    

#==================================================
async def ensure_collection_exists(
        vector_client: IVectorClient,
        name: str,
        ):
    collections = await vector_client.list_collections()
    if name not in collections:
        await create_collection(vector_client, name)
        log.info(f"✅ Collection '{name}' created automatically.")


#==================================================
def normalize_threshold(
        distance: str,
        raw_threshold: float | None
)-> float | None:
    """Convert user-defined score threshold to effective value based on distance metric."""
    if raw_threshold is None:
        return None

    distance = distance.lower()
    if distance == "cosine":
        if raw_threshold <= 0:
            return None
        if raw_threshold >= 0.95:
            return 0.95
        return raw_threshold
    
    if distance in ("dot", "ip"):
        return raw_threshold
    
    if distance in ("euclidean", "l2", "euclid"):
        return None  
    
    return None
