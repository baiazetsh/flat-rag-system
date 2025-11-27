# # app/routes/llm.py
from fastapi import APIRouter, HTTPException, Depends
from app.core.logger import log
from app.core.config import cfg
from app.services.llm_service import generate_answer
from app.models import AskRequest
from app.dependencies.llm_client import llm_client_dependency
from app.clients.base_client import ILLMClient

router = APIRouter(prefix="/api/llm", tags=["LLM"])

@router.post("/ask")
async def ask_model(    
    body: AskRequest,
    llm_client: ILLMClient = Depends(llm_client_dependency),
    ):
    """Send prompt directly to configured LLM (no retrieval)."""
    try:
        result = await generate_answer(
            llm_client=llm_client,
            prompt=body.prompt,
        )

        return {
            "answer": result,
            "model": cfg.llm_model,
        }
    
    except Exception as e:
        log.error(f"LLM ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
