#app/routes/vector_ui.py
"""Web UI for managing vector collections (list, create, delete, recreate)."""

from fastapi import APIRouter, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.dependencies.vector_client import vector_client_dependency
from app.clients.base_client import IVectorClient

from app.services.vector_services import (
    list_collections,
    create_collection,
    delete_collection,
    recreate_collection,
)
from app.core.logger import log

router = APIRouter(prefix="/ui/vector", tags=["Vector UI"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
async def vector_dashboard(
    request: Request,
    vector_client: IVectorClient = Depends(vector_client_dependency),
):
    """Show list of collections with management actions."""
    try:
        collections = await list_collections(vector_client)
    except Exception as e:
        log.error(f"⚠️ Failed to list collections: {e}")
        collections = []
    return templates.TemplateResponse(
        "vector_dashboard.html",
        {"request": request, "collections": collections},
    )


@router.post("/create")
async def create_collection_ui(
    request: Request,
    name: str = Form(...),
    vector_size: int = Form(768),
    distance: str = Form("cosine"),
    vector_client: IVectorClient = Depends(vector_client_dependency),
):
    """Handle creation via form."""
    try:
        await create_collection(
            vector_client=vector_client,
            name=name,
            vector_size=vector_size,
            distance=distance,
        )
        return RedirectResponse(url="/ui/vector", status_code=303)
    except Exception as e:
        log.error(f"❌ Failed to create collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete")
async def delete_collection_ui(
    request: Request,
    name: str = Form(...),
    vector_client: IVectorClient = Depends(vector_client_dependency),
    ):
    """Handle deletion via form."""
    try:
        await delete_collection(vector_client, name)
        return RedirectResponse(url="/ui/vector", status_code=303)
    except Exception as e:
        log.error(f"❌ Failed to delete collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recreate")
async def recreate_collection_ui(
    request: Request,
    name: str = Form(...),
    vector_size: int = Form(768),
    distance: str = Form("cosine"),
    vector_client: IVectorClient = Depends(vector_client_dependency),
):
    """Handle recreation via form."""
    try:
        await recreate_collection(
            vector_client=vector_client,
            name=name,
            vector_size=vector_size,
            distance=distance,
        )
        
        return RedirectResponse(url="/ui/vector", status_code=303)
    except Exception as e:
        log.error(f"❌ Failed to recreate collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


