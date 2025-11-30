#rag_local/backend/app/main.py v1.0.1

from dotenv import load_dotenv
from fastapi import FastAPI
import os
from app.core.logger import log
from app.core.config import cfg
from app.clients.client_factory import ClientFactory

from app.dependencies.splitter_factory import get_splitter 
from app.routes import (   
    plot,    
    health,
    upload,    
    search_with_llm,    
    vector_router,    
    embed,
    llm,
    search,
    rag_ui,
    vector_ui,
    search_ui,    
)

load_dotenv()

app = FastAPI(title = cfg.app_name)

app.include_router(plot.router)
app.include_router(rag_ui.router)
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(search_with_llm.router)
app.include_router(vector_router.router)
app.include_router(vector_ui.router)
app.include_router(llm.router)
app.include_router(embed.router)
app.include_router(search.router)
app.include_router(search_ui.router)
#app.include_router(rag.router)


# Lifecycle managment  ==================
@app.on_event("startup")
async def startup_event():
    """
    Initialize core clients and ensure vector DB is ready.
    Skips initialization if running in CI.
    """    
    if os.getenv("CI") == "true":
        log.warning("🧪 CI mode detected — skipping Qdrant connection.")
        return

    
    try:
        # Initialize clients  =========
        clients = ClientFactory.build_all()
        app.state.embedding = clients["embedding"]
        app.state.llm = clients["llm"]
        app.state.vector = clients["vector"]
        app.state.reranker = clients["reranker"]
        
        log.info(f" All clients initializing successfully!")

        # Verify or create vector  collections ======
        collections = await app.state.vector.list_collections()
        collection_cfg = cfg.collection
        
        if collection_cfg.name not in collections or collection_cfg.recreate_if_exists:
            if collection_cfg.name in collections and collection_cfg.recreate_if_exists:
                log.warning(
                    f"Recreating collection '{collection_cfg.name}' as requested."
                )
            await app.state.vector.create_collection(
                name=collection_cfg.name,
                vector_size=collection_cfg.size,
                distance=collection_cfg.distance,               
            )
            log.info(f"Created vector collection '{collection_cfg.name}'.")
        else:
            log.info(f"🚀🚀Collection '{collection_cfg.name}' already exists")
            
    except Exception as e:
        log.warning(f"⚠ Skipping vector backend initialization: {e}")
    
    try:
        splitter = get_splitter()
        if splitter.semantic_chunker.is_available():
            log.info(
                f"SmartTextSplitter loaded model: {splitter.semantic_chunker.model_name}"
                )
        else:
            log.warning("SemanticChaker unavailable -> fallback to token-aware mode.")                                       

    except Exception as e:
        log.warning(f"Unknown error: {e}")    

                                                        
            

@app.on_event("shutdown")        
async def shutdown_event():
    """Gracefully close async clients"""
    try:        
        for client_name in ("embedding", "llm", "vector"):
            client = getattr(app.state, client_name, None)
            if client:
                close_func = getattr(client, "aclose", None)
                if callable(close_func):
                    await close_func()
                    log.info(f"Closed client: {client_name}")
        log.info("Application shutdown complete.")
    except Exception as  e:
        log.warning(f"!!!Error during shutdown: {e}")

