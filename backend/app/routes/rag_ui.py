#app/routes/rag_ui.py
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx, os
import json
import logging
from app.clients import cfg, qdrant

# ========= Logger setup =========
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


templates = Jinja2Templates(directory="app/templates")
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        qdrant_collections = qdrant.get_collections()
        collections = [c.name for c in qdrant_collections.collections]
        log.info(f"🧠 Found collections: {collections}")
    except Exception:
        collections = ["docs"]      #fallback

    return templates.TemplateResponse(
        "index.html",
        {"request": request,
        "collections": collections}
        )



@router.post("/ask", response_class=HTMLResponse)
async def ask(
    request: Request,
    query: str = Form(...),
    mode: str = Form("local"),
    collection: str = Form("docs")
    ): 
    
    #url = str(request.url_for("search_with_llm") )
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    url = f"{backend_url}/api/search_with_llm"
      
    client= request.app.state.client    
    
    try:
        response = await client.post(
            url,           
            json={
                "query": query,
                "top_k": cfg.searchsettings.top_k,
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

    answer = data.get("answer", "No answer.") if isinstance(data, dict) else {}
    models = data.get("models", {}) if isinstance(data, dict) else {}
    model = models.get(
        "llm", cfg.ollama.llm_model
        ) if isinstance(models, dict) else cfg.ollama.llm_model
    #model = data.get("models", {}).get("llm", cfg.ollama.llm_model)
    #vector_db = data.get("models", {}).get("embedding", cfg.ollama.embed_model)
    vector_db = models.get(
        "embedding", cfg.ollama.embed_model
        ) if isinstance(models, dict) else cfg.ollama.emdeb_model
        
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

