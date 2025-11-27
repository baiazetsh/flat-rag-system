#app/services/generate_rag_answer.py v1.0.1
"""
RAG Orchestrator:
Embedding → Vector search → Prompt building → LLM generation
Clean, DI-based architecture.
"""

import time
from app.core.logger import log
from app.core.config import cfg

from app.services.prompt_builder import build_rag_prompt_advanced
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
    
    context_chunks = []
    metadata_list = []
    for r in results:
        text = r.get("text", "")
        if text:
            context_chunks.append(text)
            meta = {
                "source": r.get("source", "unknown"),
                "chunk_index": r.get("chunk_index", -1),
            }
            metadata_list.append(meta)
       
    context = "\n\n".join(context_chunks)
    prompt = build_rag_prompt_advanced(
        context=context,
        query=query,
        metadata=metadata_list,
    )
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
         