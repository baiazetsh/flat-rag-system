# app/routes/plot.py v1.0.2
"""
Vector Visualization Endpoint (PCA projection)
---------------------------------------------
Projects high-dimensional vectors from the selected collection
onto 2D space using PCA and returns a PNG plot.
Compatible with any vector backend (Qdrant, Chroma, etc.)
"""

import io
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from fastapi import APIRouter, Request, HTTPException, Depends
from app.clients.base_client import IVectorClient
from fastapi.responses import StreamingResponse
from sklearn.decomposition import PCA
from app.dependencies.vector_client import vector_client_dependency
from app.core.logger import log
from app.core.config import cfg

router = APIRouter(prefix="/api/plot", tags=["Visualization"])


@router.get("/")
async def plot_vectors(
    vector_client: IVectorClient = Depends(vector_client_dependency),
    limit: int = 100,
    collection: str | None = None,
):
    """
    PCA projection of stored vectors into 2D space.
    Useful for sanity-checking vector quality and distribution.
    """
    try:
        collection_name = collection or cfg.default_collection_name        
        vectors = []
        if not hasattr(vector_client, "get_vectors"):
            raise RuntimeError(
                "Vector client does not implement 'get_vectors' method. "
                "Add it to IVectorClient and its implementations."
            )            
        vectors = await vector_client.get_vectors(collection_name, limit)

        if not vectors:
            plt.figure(figsize=(5, 5))
            plt.text(0.5, 0.5, "No vectors found", ha="center", va="center", fontsize=12)
            plt.axis("off")

        else:
            arr = np.array(vectors)

            if arr.shape[0] < 2:
                raise ValueError("Not enough vectors for PCA visualization.")

            pca = PCA(n_components=2)
            reduced = pca.fit_transform(arr)

            plt.figure(figsize=(7, 7))
            plt.scatter(reduced[:, 0], reduced[:, 1], s=30, alpha=0.7)
            plt.title(f"PCA projection of {len(vectors)} vectors from '{collection_name}'")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close()
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        log.error(f"Vector plot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))