#app/services/retriever.py  v2.0

from __future__ import annotations
from typing import List, Dict

import numpy as np

from app.core.logger import log
from app.core.config import cfg
from app.clients.base_client import (
    IVectorClient,
    IRerankerClient,
    IEmbeddingClient,
    ILLMClient,
)
from app.services.rag_debugger import RagDebugger 



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


    # ---------- Complex search selector ----------
    def _should_use_complex_retrieval(self, query: str) -> bool:
        """Define that need the complex search strategy"""
        # simple criteries
        q = query.lower().strip()
        word_count = len(query.split())

        has_cyrillic = any('а'<= c <= 'я' or 'А' <= c <= 'Я' for c in q)

        has_comparison = any(
            w in query.lower() for w in [
            "compare", "difference", "distinction", " vs ", "vs ",
            "сравни", "разница", "отличие"
            ]
        )
        has_multiple_entities = (
            q.count(" and ") > 1 or
            q.count(" и ") > 1
        )

        return(
            has_cyrillic
            or word_count >10
            or has_comparison
            or has_multiple_entities
        )
    

    # ---------- Query expansion ----------,
    async def expand_query(self, query: str) -> str:
        prompt = (
            "Rewrite this query in a more explicit and semantically rich way,"
            "keeping intent unchanged:\n\n"
            f"QUERY: {query}\nREWRITE:"
        )
        return await self.llm.generate(prompt)
    

    # ---------- Multi-query ----------
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
    
    # ---------- RRF ----------
    def fuse_rrf(
            self,
            result_sets: List[List[Dict]],
            k: int  = 60,
    ) -> Dict[str, float]:
        scores:Dict[str, float] = {}
        for results in result_sets:
            for rank, hit in enumerate(results):
                raw_id = hit.get("id") or hit.get("chunk_index")
                if raw_id is None:
                    continue

                doc_id = str(raw_id)

                scores.setdefault(doc_id, 0)    
                scores[doc_id] += 1 / (k +rank +1)

        return scores
    

    # ---------- Alpha fusion ----------
    def fuse_alpha(
            self,
            base_scores: Dict[str, float],
            expanded_scores: Dict[str, float],
            alpha: float = 0.5,
    ) -> Dict[str, float]:
        combined: Dict[str, float] = {}

        all_ids = set(base_scores) | set(expanded_scores)

        for doc_id in all_ids:
            s1 = base_scores.get(doc_id, 0)
            s2 = expanded_scores.get(doc_id, 0)
            combined[doc_id] = alpha * s1 +(1 - alpha) * s2

        return combined


    # ---------- Reranking ----------
    async def rerank_docs(
            self,
            query: str,
            docs: List[Dict],
            min_rerank_score: float = cfg.min_rerank_score,
            debugger: RagDebugger = None, 
    ) -> List[Dict]:
        if not docs:
            return []
        
        texts = [d["text"] for d in docs]
        try:
            reranked_scores = await self.reranker.rerank(query, texts)
        except Exception as e:
            log.error(f"RERANKER FAILED: {e}")
            return docs[: self.top_final]

        if not reranked_scores or len(reranked_scores) !=len(docs):
            log.warning("⚠ Reranker size mismatch, fallback ")
            return docs[: self.top_final]
        
        ordered = []
        debug_scores = []
        for score, doc in zip(reranked_scores, docs):
            d = dict(doc)
            d['rerank_score'] = float(score)
            ordered.append(d)

            if debugger:
                debug_scores.append({
                    "score": float(score),
                    "id": d.get("id"),
                    "snippet": d["text"][:80].replace("\n", " ")
                })


        ordered.sort(key=lambda x: x["rerank_score"], reverse=True)
        filtered = [d for d in ordered if d["rerank_score"] >=min_rerank_score]

        if not filtered:
            log.warning(f"⚠️ There are not docs that have passed the reranking threshold: {min_rerank_score}")
            return []
        log.info(f"✅After filtriton has left {len(filtered)} docs")
        log.info(f"✅ Reranked {len(ordered)} docs, top score: {ordered[0]['rerank_score']:.3f}")

        if debugger:
            debugger.add("reranker", items=debug_scores[:5])

        return filtered[:self.top_final]
        #return ordered[: self.top_final]
    
    @staticmethod
    def _lexical_signal(query: str, text: str) -> bool:  # <<< ADDED
        q = query.lower().split()
        t = text.lower()
        return any(w in t for w in q)
    
        
    # ---------- MAIN ----------
    async def retrieve(
            self,
            query: str,
            collection: str,             
    ) -> List[Dict]:  
        
        debugger = RagDebugger()

        use_complex = self._should_use_complex_retrieval(query)
        log.info(
            f"Retrieval mode = {'complex' if use_complex else 'simple'} "
            f"for query: {query!r}"
        )
        if not use_complex:
            # simple way
            log.info("🎯 Using SIMPLE retrieval pipeline")
            query_emb = await self.embedder.embed(query)
            base_results = await self.vector.search(
                collection,
                query_emb,
                cfg.top_k_simple,
            ) 
            if not base_results:
                log.warning("⚠️ Simple retrieval returned 0 hits")
                debugger.dump()
                return []
            
            final_docs = await self.rerank_docs(
            query=query,
            docs=base_results,
            debugger=debugger,
            )
                
            # metrics
            if final_docs:
                avg_score = sum(
                    d.get("rerank_score", 0.0) for d in final_docs
                )/ len(final_docs)
                
                log.info(f"📊 SIMPLE: docs={len(final_docs)}, avg_rerank={avg_score:.3f}")
                if avg_score < cfg.min_rerank_score:
                    log.warning(f"⚠️ LOW QUALITY (SIMPLE) for query: {query[:50]!r}")
            
            return final_docs
        
        # complex way
        log.info("🔄 Using COMPLEX retrieval pipeline")


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
        for i, results in enumerate(results_all):
            debugger.add(
                "vector_search",
                query_variant=i,
                top_hits=[{
                    "id": h.get("id"),
                    "score": h.get("score"),
                    "snippet": h.get("text", "")[:80].replace("\n", " ")
                } for h in results[:5]
                ]
            )

        total_hits = sum(len(r) for r in results_all)
        log.info(f"Vector search returned {total_hits} total hits")

        if total_hits ==0:
            debugger.dump()
            log.warning("⚠️ No results from vector search")
            return []

        # 5) RRF fusion
        base_fused = self.fuse_rrf(results_all)

        dbg_rrf = sorted(
            base_fused.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        debugger.add(
            "rrf_fusion",
            top=[{"id": doc_id, "score": s} for doc_id, s in dbg_rrf]
        )

        # 6) Alpha fusion (base vs expanded)
        expanded_emb = await self.embedder.embed(expanded)
        expanded_results = await self.vector.search(collection, expanded_emb, self.top_k_base)
        expanded_fused = self.fuse_rrf([expanded_results])
        alpha_fused = self.fuse_alpha(base_fused, expanded_fused)
        log.info(f"After RRF+Alpha fusion: {len(alpha_fused)} unique docs")

        dbg_alpha = sorted(
            alpha_fused.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        debugger.add(
            "alpha_fusion",
            top = [{"id": doc_id, "score": s} for doc_id, s in dbg_alpha]
        )

        # 7) Select top-N docs by fused rank
        sorted_ids = sorted(alpha_fused.items(), key=lambda x: x[1], reverse=True)
        top_ids  = [doc_id for doc_id, _ in sorted_ids[: self.top_after_rrf]]

        # 8) Extract corresponding chunk info
        flat: Dict[str, Dict] = {}
        for results in results_all:
            for hit in results:
                raw_id = hit.get("id") or hit.get("chunk_index")
                if raw_id is  None:
                    continue
                doc_id = str(raw_id)
                flat[doc_id] = hit
        
        selected_docs = [flat[doc_id] for doc_id in top_ids if doc_id in flat]

        if not selected_docs:
            log.warning("⚠️ No docs after RRF fusion")
            return []
        
        log.info(f"Selected {len(selected_docs)} docs for reranking")

        debugger.add(
            "selected_docs_before_rerank",
            docs=[{
                "id": d.get("id"),
                "score": d.get("score"),
                "snippet": d["text"][:80].replace("\n", " ")
            } for d in selected_docs
            ]
        )

        # 9) Rerank
        final_docs = await self.rerank_docs(
            query=query,
            docs=selected_docs,
            debugger=debugger,
        )
        log.info(f"✅ Final result: {len(final_docs)} docs")

        # 10) lexical signal
        signals = [self._lexical_signal(query, d["text"]) for d in final_docs]
        debugger.add(
        "lexical_overlap",
        signals=signals,
        any_overlap=any(signals)
        )
        debugger.dump()
        return final_docs
   