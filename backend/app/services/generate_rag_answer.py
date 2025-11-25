#app/services/generate_rag_answer.py v1.0.1
"""
RAG Orchestrator:
Embedding → Vector search → Prompt building → LLM generation
Clean, DI-based architecture.
"""

import time
from app.core.logger import log
from app.core.config import cfg

from app.services.prompt_builder import build_rag_prompt
#from app.services.llm_service import generate_answer
from app.clients.base_client import IEmbeddingClient, ILLMClient, IVectorClient
from app.services.vector_services import normalize_threshold


#=============== Main RAG pipeline =============
async def generate_rag_answer(
        vector_client: IVectorClient,
        embedding_client: IEmbeddingClient,
        llm_client: ILLMClient,
        query: str,
        top_k: int | None = None,
        collection: str | None = None,        
        ) -> dict:
    """ Run the complete Rag pipeline."""

    if not query:
        raise ValueError("Query cannot be empty")
    
    # Using parametrs or fallback on cfg
    if top_k is None:
        top_k = cfg.top_k
    if collection is None:
        collection = cfg.default_collection_name

    start_time = time.perf_counter()
    log.info(f"🚀 RAG start: query='{query[:80]}…' (collection={collection})")
    
    
    # 1.request embedding=================
    query_vector = await embedding_client.embed(query)
    effective_threshold = normalize_threshold(
        cfg.collection.distance,
        cfg.collection.score_threshold,
    )

    # 2. Search  =================
    results = await vector_client.search(
        collection=collection,
        vector=query_vector,
        top_k=top_k,
        score_threshold=effective_threshold,
        )
    if not results:
        return {
            "query": query,
            "answer": "No relevant documents found.",
            "context_used": 0,
            "collection": collection,
        }
       
    context = "\n\n".join([r["text"] for r in results if r.get("text")])
    prompt = build_rag_prompt(context, query)
    answer = await llm_client.generate(prompt)

    total_time = time.perf_counter() - start_time
    log.info(f"✅ RAG completed in {total_time:.2f}s")

    return {
        "query": query,
        "answer": answer.strip(),
        "context_used": len(results),
        "collection": collection,
        "models": {
            "llm": cfg.llm_model,
            "embedding": cfg.embedding_model,
        },
        "results": results,
        "timing": {"total": round(total_time, 2)}
    }
         