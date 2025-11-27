#app/routes/embed.py v1.0.1
from fastapi import Request, APIRouter, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from app.core.logger import log
from app.core.config import cfg
from app.dependencies.vector_client import vector_client_dependency
from app.clients.base_client import IVectorClient
import uuid

router = APIRouter(prefix="/api/embed", tags=["Embed"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/form")
async def embed_form(
    request: Request,
    vector_client: IVectorClient = Depends(vector_client_dependency),
):
    """Render HTML form for embedding text manually."""
    try:        
        collections = await vector_client.list_collections()
    except Exception as e:
        log.warning(f"!!! Failed to list collection: {e}")
        collections = [cfg.default_collection_name]

    return templates.TemplateResponse(
        "embed_form.html",
        {"request": request, "collections": collections},
    )



@router.post("/")
async def embed_text(
    request: Request,
    vector_client: IVectorClient = Depends(vector_client_dependency),
    ):
    """Embed text and store vector in the selected collection."""
    try:
        data = await request.json()
        text = data.get("text", "").strip()
        collection = data.get("collection", cfg.default_collection_name)
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty" )
        
        embedding_client = getattr(request.app.state, "embedding", None)
        if not embedding_client:
            raise RuntimeError("Embedding client not initialized")        

        vector = await embedding_client.embed(text)
        await vector_client.upsert(
            collection_name=collection,
            points=[
                {
                    "id": uuid.uuid4().hex,
                    "vector": vector,
                    "payload": {"text": text},
                }
            ],
        )
        log.info(f"✅ Embedded text saved to '{collection}' (dim={len(vector)})")
        return{
            "status": "ok",
            "vector_dim": len(vector),
            "collection": collection,
        }
    except Exception as e:
        log.error(f"❌  Embed failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
