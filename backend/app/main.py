#rag_local/backend/app/main.py
import collections
from math import sqrt
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from fastapi import FastAPI
from httpx import AsyncClient
import logging
import os

from app.clients import http_client, cfg, qdrant
from app.routes import base, plot, rag_ui

load_dotenv()

# ========= Logger setup =========
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


app = FastAPI(title="RAG Local API")
app.include_router(base.router, prefix="/api")
app.include_router(plot.router)
app.include_router(rag_ui.router)

#===============UTILS============================
def distance(p1, p2):
    return sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


@app.get("/ping_qdrant")
def ping_qdrant():
    """Checking alive Qdrant and return collections"""
    try:  
        info = qdrant.get_collections()
        collections = [c.name for c in info.collections]
        return {
            "qdrant_alive": True,
            "collections": collections
            }
    except Exception as e :
        return {"qdrant_alive": False, "error": str(e)}
    
#============LIFECYCLE======================

@app.on_event("startup")
async def startup_event():
    """Ensure Qdrant collection exists on startup"""
    app.state.client = AsyncClient(base_url="http://127.0.0.1:8000", timeout=60.0)

    collections = [c.name for c in qdrant.get_collections().collections]
    collection_name = cfg.qdrant.collection
    
    if collection_name not in collections:
        log.info(f"Creating Qdrant collection '{collection_name}'")
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=768,
                distance=qmodels.Distance.COSINE
            ),
        )       
    else:
        log.info(f"🚀🚀🚀🚀Collection '{collection_name}' already exists")
        
        
@app.on_event("shutdown")        
async def shutdown_event():
    """Gracefully close HTTP client"""
    if hasattr(app.state, "client"):
        await app.state.client.aclose()
        
    try: 
        #global client
        await http_client.aclose()
    except Exception:
        pass

