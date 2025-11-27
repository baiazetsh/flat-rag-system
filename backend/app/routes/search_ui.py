# app/routes/search_ui.py v1.0.1
"""
UI for semantic search (without LLM).
Allows testing of vector retrieval results directly from the browser.
"""
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx
import os

from app.core.config import cfg
from app.core.logger import log

router = APIRouter(prefix="/ui/search", tags=["Semantic Search UI"])
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def search_form(request: Request):
    """Render semantic search form."""
    try:        
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{cfg.backend_url}/api/health/", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                collections = (
                    data.get("vector_backend", {})
                        .get("collections", [])
                    or [cfg.default_collection_name]
                )
            else:
                log.warning(f"!!! Health endpoint returned {resp.status_code}")
                collections = [cfg.default_collection_name]

    except Exception as e:
        log.warning(f"⚠ Could not load collections: {e}")
        collections = [cfg.default_collection_name]

    return templates.TemplateResponse(
        "search_form.html",
        {"request": request, "collections": collections},
    )


@router.post("/", response_class=HTMLResponse)
async def search_results(
    request: Request,
    query: str = Form(...),
    collection: str = Form(...),
    top_k: int = Form(5),
):
    """Perform semantic search via /api/search and display results."""    
    url = f"{cfg.backend_url}/api/search"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json={"query": query, "collection": collection, "top_k": top_k},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
        try:
            data = response.json()
        except Exception:
            data = {"error": f"Invalid JSON from backend: {response.text}"}
            
    except Exception as e:
        log.error(f"Search UI request failed: {e}")
        data = {"error": str(e)}

    return templates.TemplateResponse(
        "search_result.html",
        {
            "request": request,
            "query": query,
            "collection": collection,
            "results": data.get("results", []),
            "count": data.get("count", 0),
            "timing": data.get("timing_ms", {}),
        },
    )
