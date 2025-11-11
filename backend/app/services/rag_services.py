#app/services/rag_services.py
import os
import time
import uuid
from typing import List, Dict, Any
import logging
from fastapi import HTTPException

from app.clients import cfg, qdrant, http_client
from qdrant_client.http import models as qmodels

# ========= Logger setup =========
log = logging.getLogger(__name__)


#======== 1.  _get embedding ===========================
async def _get_embedding(text: str) -> List[float]:
    """ Get text embedding via Ollama  """
    t0 = time.perf_counter()
    data = {"model": cfg.ollama.embed_model, "prompt":text}    
    resp = await http_client.post(f"{cfg.ollama.base_url}/api/embeddings", json=data)
    elapsed_embedding = (time.perf_counter() - t0) * 1000

    if resp.status_code != 200:    
        raise HTTPException(
            status_code=500, detail=f"Ollama enbedding error: {resp.text}"
            )
    embedding = resp.json(). get("embedding", [])

    # check for empty embedding
    if not embedding:
        raise HTTPException(status_code=500, detail="LLM returned empty embedding")
    log.info(
        f" Embedding done in {elapsed_embedding:.1f} ms"
        f" ({len(embedding)} dims, {len(text)} chars)"
        )
    return embedding, elapsed_embedding
   
    
#========= 2. generate response  via llm   ========================
async def _generate_llm_response(prompt: str) -> str:
    """Generate answer via LLM"""
    t0 = time.perf_counter()
    data = {"model": cfg.ollama.llm_model, "prompt": prompt, "stream": False}
    resp = await http_client.post(f"{cfg.ollama.base_url}/api/generate", json=data)
    elapsed_generated = (time.perf_counter() - t0)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=500, detail=f"Ollama generation error: {resp.text}"
            ) 
    answer = resp.json().get("response", "").strip() or "No answer generated."  
    log.info(f" LLM response ready in {elapsed_generated:.2f} s ({len(answer)} chars)")
    return answer, elapsed_generated


#=============== 3. Main RAG pipeline =============
async def generate_rag_answer(
        query: str,
        top_k: int = None,
        collection: str = None
        ) -> Dict[str, Any]:
    """
    Full RAG-pipeline:
    1. Request embedding
    2. Find in Qdrant
    3. Promt formig
    4. Answer generation
    """
    if not query:
        raise ValueError("Query cannot be empty")
    
    # Using parametrs or fallback on cfg
    top_k = top_k if top_k is None else cfg.searchsettings.top_k
    collection = collection or cfg.qdrant.collections
    
    log.info(f"Starting RAG for query: '{query[:80]}..' (top_k={top_k})")
    total_start = time.perf_counter()
    
    # 1.request embedding=================
    query_vec, embedding_ms = await _get_embedding(query)

    # 2. Search in Qdrant =================
    t0 = time.perf_counter()
    hits = qdrant.search(
        collection_name=collection,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
        score_threshold=cfg.qdrant.score_threshold,
    )
    search_ms = (time.perf_counter() - t0) * 1000
    context_texts = [h.payload["text"] for h in hits if "text" in h.payload]
    scores = [h.score for h in hits]

    log.info(
        f" Qdrant search done in {search_ms:.1f} ms"
        f"({len(context_texts)} chunks, scores: {[f'{s:.3f}' for s in scores]})"
        )
    
    if not context_texts:
        log.warning("!!! No relevant documents found.")
        return {
            "query": query,
            "answer": "No relevant documents found.",
            "context_used": 0,
            "model": cfg.ollama.llm_model,
            "collection": collection,
            }
        
    
    # 3. Build prompt ==============
    # We limit by the numbers of chunks, not by characters
    max_context_chars = 6000
    context_parts = []
    current_length = 0

    for text in context_texts:
        if current_length + len(text) > max_context_chars:
            break
        context_parts.append(text)
        current_length += len(text)

    context = "\n\n".join(context_parts)

    prompt = f"""
        You are a scientific assistant. 
        Use ONLY the information from the context below to answer the question.
        Do NOT repeat the question, do NOT speculate, and do NOT answer philosophically.
        If the answer is not clearly in the context, say: "Answer not found in the documents."

        Context:
        {context}

        Question: {query}

        Answer concisely and factually (in the same language as the question):
        """
    
    # 4. Generate final answer ===============
    answer, llm_s = await _generate_llm_response(prompt)


     #  Log summary =======
    total_time = time.perf_counter() - total_start
    log.info(f"\n{'=*60'}")
    log.info(f"\n🚀 RAG pipeline complete")
    log.info(f"{'='*60}")
    log.info(f"🔍 LLM model: {cfg.ollama.llm_model}")
    log.info(f"🔹 Embedding model: {cfg.ollama.embed_model}")
    log.info(f"📦 Collection: {collection}")
    log.info(f"📚 Context chunks used: {len(context_parts)}/{len(context_texts)}")
    log.info(f"Relevance scores: {[f'{s:.3f}' for s in scores[:len(context_parts)]]}")
    log.info(f"🧩 Context preview: {context[:150].replace(chr(10), ' ')}...")
    log.info(f"💬 Final answer preview: {answer[:150].replace(chr(10), ' ')}...")
    log.info(f"Total duration: {total_time:.2f} s")
    log.info(f"✅ Status: OK\n")
    log.info(f"{'='*60}\n")


    # 5. Result return section ==================
    results = [
        {
            "rank": i + 1,
            "score": round(h.score, 3),
            "source": h.payload.get("source", "unknown"),
            "text_preview": h.payload.get("text", ""[:300]),
        }
        for i, h in enumerate(hits)
    ]
    return {
        "query": query,
        "answer": answer.strip(),
        "context_used": len(context_parts),
        "results": results,        
        "models": {
            "llm": cfg.ollama.llm_model,
            "embedding": cfg.ollama.embed_model
        },
        "collection": collection,        
        "timing": {
            "embedding": round(embedding_ms, 1),
            "search": round(search_ms, 1),
            "llm": round(llm_s, 2),
            "total": round(total_time, 2),
        },
    }


def create_collection(name: str, vector_size: int = 768):
    """
    Create a new Qdrant collection if it doesn't exist.
    """
    collections = [c.name for c in qdrant.get_collections().collections]
    if name in collections:
        log.warning(f"! Collection '{name}' already exists.")
        return
    
    qdrant.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE
        )
    )
    log.info(f"Created new collection: {name}")
        
   

    
    