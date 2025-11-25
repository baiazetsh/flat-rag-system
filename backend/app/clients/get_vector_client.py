#app/clients/get_vector clients
#from app.core.config import cfg
from app.core.logger import log
from app.clients.client_factory import ClientFactory


# #==================================================
async def get_vector_client(app=None):
    """
    Universal accessor for the vector client.
    Works both inside FastAPI (via app.state.vector)
    and standalone (e.g., Celery, tests, CLI scripts).
    """
    if app and hasattr(app.state, "vector"):
        return app.state.vector    
    log.warning("⚠️ FastAPI app context not found — creating temporary vector client.")
    return ClientFactory.build_vector()
    