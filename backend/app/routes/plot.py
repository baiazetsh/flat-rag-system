import io, os
import numpy as np
import matplotlib.pyplot as plt
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sklearn.decomposition import PCA
from app.clients import qdrant

router = APIRouter()

@router.get("/plot")
def plot_vectors(limit: int = 100):
    collection_name = os.getenv("QDRANT_COLLECTION", "docs")

    points, _ = qdrant.scroll(
        collection_name=collection_name,
        limit=limit,
        with_vectors=True
    )

    if not points:
        plt.figure()
        plt.text(0.5, 0.5, "No vectors found", ha='center', va='center')
    else:
        # достаём все вектора
        vectors = [p.vector for p in points if p.vector]
        labels = [f"v{i}" for i in range(len(vectors))]

        # PCA → 2D
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)

        plt.figure(figsize=(7, 7))
        for label, (x, y) in zip(labels, reduced):
            plt.scatter(x, y, s=40)
            plt.text(x + 0.02, y + 0.02, label, fontsize=8)

        plt.title(f"PCA projection of {len(vectors)} vectors from '{collection_name}'")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
