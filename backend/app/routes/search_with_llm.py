#app/routes/search_with_llm.py v2.0

from fastapi import APIRouter, HTTPException, Depends
from app.models import RAGRequest

from app.core.logger import log
from app.core.config import cfg
from app.dependencies.vector_client import vector_client_dependency
from app.dependencies.embedding_client import embedding_client_dependency
from app.dependencies.llm_client import llm_client_dependency
from app.dependencies.reranker_client import reranker_client_dependency
from app.services.retriever import Retriever
from app.services.prompt_builder import build_rag_prompt_advanced


router = APIRouter(prefix="/api", tags=["RAG Core"])


@router.post("/search_with_llm")
async def search_with_llm(
    request: RAGRequest,
    vector_client = Depends(vector_client_dependency),
    embedding_client = Depends(embedding_client_dependency),
    llm_client = Depends(llm_client_dependency),
    reranker_client = Depends(reranker_client_dependency),
):
    """
    New RAG v2.0 endpoint:
    - expand query
    - multi-query generation
    - embeddings
    - vector search
    - RRF fusion
    - Alpha-fusion
    - reranking (cross-encoder)
    - final LLM answer
    """
    try:
        # 1 Construct retriever
        retriever = Retriever(            
            embedder=embedding_client,
            reranker=reranker_client,
            vector=vector_client,
            llm=llm_client,           
        )

        # 2 Run full retrieval pipeline
        final_docs = await retriever.retrieve(
            query=request.query,
            collection=request.collection,
        )

        if not final_docs:
            return{
                "query": request.query,
                "answer": "No relevant documents found.",
                "context_used": 0,
                "results": [],
                "collection": request.collection,
            }

        # 3 Build context block for LLM
        context = "\n\n".join(d["text"] for d in final_docs)

        metadata = [
            {
                "source": d.get("source", "uknown"),
                "chunk_index": d.get("chunk_index", -1)
            }
            for d in final_docs
        ]

        # 3.1 calculate avg rerank score
        scores = [d.get("rerank_score") for d in final_docs if "rerank_score" in d]
        avg_rerank = sum(scores) / len(scores) if scores else None

        # 3.2 determine confidence level
        confidence = None
        if avg_rerank is not None:
            if avg_rerank < cfg.min_rerank_score:
                confidence = "low"
            elif avg_rerank < 0.3:
                confidence = "medium"

         # 3.3 build improved prompt
        prompt = build_rag_prompt_advanced(
            context=context,
            query=request.query,
            metadata=metadata,
            confidence_level=confidence,
        )

        prompt = build_rag_prompt_advanced(
            context=context,
            query=request.query,
            metadata=metadata,
        )

            
        # 4 Generate answer
        answer = await llm_client.generate(prompt)
        answer = answer.strip() if isinstance(answer, str) else answer

        # 5 JSON reply for UI
        return{
            "query": request.query,
            "answer": answer,
            "context_used": len(final_docs),
            "results": final_docs,
            "collection": request.collection,
            "models": {
                "llm": cfg.llm_model,
                "embedding": cfg.embedding_model,
                "reranker": cfg.reranker_model,
            },
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Bad request: {e}")

    except Exception as e:
        log.error(f"RAG endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
