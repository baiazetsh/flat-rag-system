#app/clients/rerank_client.py
""" A wrapper about reranking API"""
from __future__ import annotations
from typing import List
import asyncio
from app.core.config import cfg
from sentence_transformers import CrossEncoder
from app.clients.base_client import IRerankerClient
from app.core.logger import log

class GenericRerankerClient(IRerankerClient):
    """
    
    GPU/CPU CrossEncoder reranker.
    Loads model once at startup.
    """
    def __init__(self):
        model_name = cfg.reranker_model

        #Determine device
        device = "cuda" if cfg.device_for_reranker == "cuda" else "cpu"

        log.info(f"🔄 Loading reranker model: {model_name} on {device}")
        
        # Heavy  model load - only once at startup
        self.model = CrossEncoder(model_name, device=device)
        log.info(f"✅ Reranker initialized")

    async def rerank(self,
                     query: str,
                     docs: list[str],
    ):
        """Return scores for reranking (higer = more relevant)."""

        if not docs:
            return []
        
        # prepare pairs query+doc
        pairs = [[query, d] for d in docs]

        #CrossEvcoder predict() is sync -> wrap in async thead
        scores = await asyncio.to_thread(self.model.predict, pairs)

        # Convert numpy to python floats
        return [float(s) for  s in scores]
        