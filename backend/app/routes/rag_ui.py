#app/routes/rag_ui.py v1.0.1
"""
RAG Web UI Routes
-----------------
Serves the HTML interface for manual testing of the RAG pipeline.
Works with any configured vector backend (Qdrant, Chroma, etc.)
and LLM provider (Ollama, OpenAI, etc.).
"""

from fastapi import APIRouter, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx
import os
import json

from app.core.logger import log
from app.core.config import cfg
from app.dependencies.vector_client import vector_client_dependency

router = APIRouter(prefix="/ui", tags=["RAG UI"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
async def rag_index(
    request: Request,
    vector_client = Depends(vector_client_dependency),
):
    """Render main RAG form with available collections."""
    try:
        collections = await vector_client.list_collections()
        log.info(f"🧠 Available collections: {collections}")
    except Exception as e:
        log.warning(f"!!!Failed to load collections: {e}")
        collections = [cfg.default_collection_name]
       
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "collections": collections,
        },
        )


@router.post("/ask", response_class=HTMLResponse)
async def rag_ask(
    request: Request,
    query: str = Form(...),    
    collection: str = Form("docs"),
    mode: str = Form("local"),
    ): 
    """
    Handles user prompt submission via the HTML form.
    Sends the query to the main RAG endpoint (/api/search_with_llm)
    and renders the results into 'result.html'.
    """    
    backend_url = cfg.backend_url
    url = f"{backend_url}/api/search_with_llm"    
    async with httpx.AsyncClient(timeout=100) as client:
        try:
            response = await client.post(
                url,           
                json={
                    "query": query,
                    "top_k": cfg.top_k,
                    "collection": collection
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                    },
                timeout=100,
            )
            log.info(f"RAG response status: {response.status_code}")
            log.info(f"RAG raw: {response.text[:300]}")

            content_type = response.headers.get('content-type', '')
            if 'application/json' not in content_type:
                log.error(f"❌ Expected JSON but got: {content_type}")
                data = {"answer": "⛔ ERROR: Backend returned HTML instead of JSON"}
            else:            
                try:
                    data = response.json() if response and response.content else {}
                    if not isinstance(data, dict):
                        log.error(f"❌ Response is not a dict: {type(data)}")
                        data = {"answer": str(data)}
                except json.JSONDecodeError as e:
                    log.error(f"❌ JSON decode error: {e}")
                    data = {"answer": f"⛔ JSON parse error: {e}"}
                    #data = {"answer": await response.text()}
                
        except Exception as e:
            log.error(f"❌ Request error: {e}")
            data = {"answer":f"⛔ ERROR: {e}"}

    # Extract fields
    answer = data.get("answer", "No answer.") if isinstance(data, dict) else {}
    models = data.get("models", {}) if isinstance(data, dict) else {}
    model = models.get(
        "llm", cfg.llm_model
        ) if isinstance(models, dict) else cfg.llm_model
   
    vector_db = (
        models.get("embedding", cfg.embedding_model)
        if isinstance(models, dict)
        else cfg.embedding_model
    )
        
    results = data.get("results", []) if isinstance(data, dict) else []

    timing = data.get("timing_ms", {}) if isinstance(data, dict) else {}

    try:
        safe_data = json.dumps(
            data, ensure_ascii=False, indent=2, default=str
        )
    except Exception:
        safe_data = str(data)

    return templates.TemplateResponse(
        "result.html",   
        {
            "request": request,
            "query": query,
            "result": data,
            "model": model,
            "vector_db": vector_db,
            "collection": collection,         
            "raw_data": safe_data,
            "timing": timing,
            "results": results,
        }
    )
