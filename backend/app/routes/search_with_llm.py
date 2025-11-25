#app/routes/search_with_llm.py
from fastapi import APIRouter, HTTPException, Depends
from app.models import RAGRequest
from app.services.generate_rag_answer import generate_rag_answer
from app.core.logger import log
from app.dependencies.vector_client import vector_client_dependency
from app.dependencies.embedding_client import embedding_client_dependency
from app.dependencies.llm_client import llm_client_dependency


router = APIRouter(prefix="/api", tags=["RAG Core"])

@router.post("/search_with_llm")
async def search_with_llm(
    request: RAGRequest,
    vector_client = Depends(vector_client_dependency),
    embedding_client = Depends(embedding_client_dependency),
    llm_client = Depends(llm_client_dependency),
):
    """
    Full RAG pipeline: embedding → vector search → LLM generation.
    Main endpoint for answering questions.
    """
    try:
        result = await generate_rag_answer(
            vector_client=vector_client,
            embedding_client=embedding_client,
            llm_client=llm_client,
            query=request.query,
            top_k=request.top_k,
            collection=request.collection,
            
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Bad request: {e}")

    except Exception as e:
        log.error(f"RAG endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
