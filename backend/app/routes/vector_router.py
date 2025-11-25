#app/routes/vector_router.py.py
"""
Vector management endpoints:
- list collections
- create collection
- delete collection
- recreate collection
"""
from fastapi import APIRouter, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from app.core.logger import log
from app.dependencies.vector_client import vector_client_dependency
from app.services.vector_services import(
    list_collections,
    create_collection,
    delete_collection,
    recreate_collection,
)
from app.core.config import cfg
from app.clients.base_client import IVectorClient

router = APIRouter(prefix="/api/vector", tags=["Vector Collections"])

#============================================================
@router.get("/list")
async def list_collections_route(
    vector_client: IVectorClient = Depends(vector_client_dependency),
):
    """List all available vector collections."""
    try:
        collections = await list_collections(vector_client)
        return{"collections": collections}
    
    except Exception as e:
        log.error(f"❌ Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#============================================================
@router.post("/create")
async def create_collection_route(
    vector_client: IVectorClient = Depends(vector_client_dependency),
    name: str = Form(..., description="Collection name"),
    vector_size: int | None = Form(None, description="Vector dimension"),
    distance: str |None = Form(None, description="Distance metric"),
):
    """
    Universal endpoint for creating vector collections.
    Works with both API and Web UI forms.
    Create a new vector collection in the configured backend.
    Example POST from fields:
        - name: "alice
        - vector_size: 1024
        -distance: "dot" | "coisine" | "euclid"
    """
    try:
        #fallback from config
        vector_size = vector_size or cfg.collection.size
        distance = distance or cfg.collection.distance
       
        result = await create_collection(
            vector_client=vector_client,
            name=name,
            vector_size=vector_size,
            distance=distance,
        )
        log.info(f"🧱 Collection creation result: {result}")
        return JSONResponse(content=result)
        
    except RuntimeError as e:
        log.error(f"Vector client not initialized: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    except Exception as e:
        log.error(f"❌ Failed to create collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}") 

#============================================================
@router.post("/delete")
async def delete_collection_route(
    vector_client: IVectorClient = Depends(vector_client_dependency),
    name: str = Form(..., description="Collection name"),
    ):
    """Delete exiting collection"""
    try:
        result = await delete_collection(
            vector_client,
            name=name
        )
        log.info(f"🗑️ Collection {name} deleted")
        return JSONResponse(content=result)
    
    except Exception as e:
        log.error(f"❌ Failed to delete collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))   


#============================================================
@router.post("/recreate")
async def recreate_collection_route(
    vector_client: IVectorClient = Depends(vector_client_dependency),
    name: str = Form(..., description="Collection name"),
    vector_size: int | None = Form(None, description="Vector dimension"),
    distance: str |None = Form(None, description="Distance metric"),
):
    #fallback from config
    vector_size = vector_size or cfg.collection.size
    distance = distance or cfg.collection.distance
    
    try:
        result = await create_collection(
                vector_client=vector_client,
                name=name,
                vector_size=vector_size,
                distance=distance,
            )
        log.info(f"♻️Collection recreation result: {result}")
        return JSONResponse(content=result)
    except Exception as e:
        log.error(f"❌ Failed to recreate collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#============================================================

#============================================================
