#app/services/retriever.py  v2.0

from __future__ import annotations
from typing import List, Dict, Tuple

import numpy as np

from app.core.logger import log
from app.core.config import cfg
from app.clients.base_client import (
    IVectorClient,
    IRerankerClient,
    IEmbeddingClient,
    ILLMClient,
)
class Retriever:
    def __init__(
            self,
            embedder: IEmbeddingClient,
            reranker: IRerankerClient,
            vector: IVectorClient,
            llm: ILLMClient,
            top_k_base=cfg.top_k_base,
            top_after_rrf = cfg.top_after_rrf,
            top_final=cfg.top_final,
    ):
        self.embedder = embedder
        self.reranker = reranker
        self.vector = vector
        self.llm = llm
        self.top_k_base = top_k_base
        self.top_after_rrf = top_after_rrf
        self.top_final = top_final


    # Query expansion
    async def expand_query(self, query: str) -> str:
        prompt = (
            "Rewrite this query in a more explicit and semantically rich way,"
            "keeping intent unchanged:\n\n"
            f"QUERY: {query}\nREWRITE:"
        )
        return await self.llm.generate(prompt)
    

    # Multi-query generation
    async def multi_query(
            self,
            query:str,
            n: int = 4,

    ) -> List[str]:
        prompt = (
            f"Generate {n} alternative semantic variations of this questions:\n "
            f"{query}\n\n"
            "Keep the meaning but use different wording.\nReturn each on new line."        
        )
        raw = await self.llm.generate(prompt)
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return [query] + lines[:n]
    
    # RRF fusion
    def fuse_rrf(
            self,
            result_sets: List[List[Dict]],
            k: int  = 60,
    ) -> Dict[str, float]:
        scores = {}
        for results in result_sets:
            for rank, hit in enumerate(results):
                doc_id = hit.get("id") or hit.get("chunk_index")
                if doc_id is None:
                    continue

                scores.setdefault(doc_id, 0)    
                scores[doc_id] += 1 / (k +rank +1)

        return scores
    

    # Alpha fusion
    def fuse_alpha(
            self,
            base_scores: Dict[int, float],
            expanded_scores: Dict[int, float],
            alpha: float = 0.5,
    ) -> Dict:
        combined = {}

        all_ids = set(base_scores) | set(expanded_scores)

        for doc_id in all_ids:
            s1 = base_scores.get(doc_id, 0)
            s2 = expanded_scores.get(doc_id, 0)
            combined[doc_id] = alpha * s1 +(1 - alpha) * s2

        return combined


    # Rerank using cross-encoder
    async def rerank_docs(
            self,
            query: str,
            docs: List[Dict],
    ) -> List[Dict]:
        if not docs:
            return []
        
        texts = [d["text"] for d in docs]
        try:
            reranked_scores = await self.reranker.rerank(query, texts)
        except Exception as e:
            log.error(f"RERANKER FAILED: {e}")
            return docs[:self.top_final]

        if not reranked_scores or len(reranked_scores) !=len(docs):
            log.warning("⚠ Reranker size mismatch, fallback ")
            return docs[: self.top_final]
        
        ordered = []
        for score, doc in zip(reranked_scores, docs):
            d = dict(doc)
            d['rerank_score'] = float(score)
            ordered.append(d)

        ordered.sort(key=lambda x: x["rerank_score"], reverse=True)
        log.info(f"✅ Reranked {len(ordered)} docs, top score: {ordered[0]['rerank_score']:.3f}")
        return ordered[: self.top_final]

        
    # main entry
    async def retrieve(
            self,
            query: str,
            collection: str,             
    ) -> List[Dict]:  
        # 1) Expand query
        expanded = await self.expand_query(query)

        # 2) Multi-query set
        queries = await self.multi_query(expanded)
        log.info(f"Generated {len(queries)} query variations")

        # 3) Embeddings for all queries
        embeddings = [await self.embedder.embed(q) for q in queries]

        # 4) Search results
        results_all = [
            await self.vector.search(
                collection,
                emb,
                self.top_k_base,
                )
            for emb in embeddings
        ]

        total_hits = sum(len(r) for r in results_all)
        log.info(f"Vector search returned {total_hits} total hits")

        if total_hits ==0:
            log.warning("⚠️ No results from vector search")
            return

        # 5) RRF fusion
        base_fused = self.fuse_rrf(results_all)

        # 6) Alpha fusion (base vs expanded)
        expanded_emb = await self.embedder.embed(expanded)
        expanded_results = await self.vector.search(collection, expanded_emb, self.top_k_base)
        expanded_fused = self.fuse_rrf([expanded_results])
        alpha_fused = self.fuse_alpha(base_fused, expanded_fused)
        log.info(f"After RRF+Alpha fusion: {len(alpha_fused)} unique docs")

        # 7) Select top-N docs by fused rank
        sorted_ids = sorted(alpha_fused.items(), key=lambda x: x[1], reverse=True)
        top_ids  = [doc_id for doc_id, _ in sorted_ids[: self.top_after_rrf]]

        # 8) Extract corresponding chunk info
        flat = {}
        for results in results_all:
            for hit in results:
                doc_id = hit.get("id") or hit.get("chunk_index")
                if doc_id is not None:
                    flat[doc_id] = hit
        
        selected_docs = [flat[doc_id] for doc_id in top_ids if doc_id in flat]

        if not selected_docs:
            log.warning("⚠️ No docs after RRF fudion")
            return []
        
        log.info(f"Selected {len(selected_docs)} docs for reranking")

        # 9) Rerank
        final_docs = await self.rerank_docs(query, selected_docs)
        log.info(f"✅ Final result: {len(final_docs)} docs")

        return final_docs


