#app/clients/factory.py v1.0.1
"""
ClienFactory - central place to create all client instances
based on enviroment configuration (provider-agnostic design).
"""
from app.core.logger import log
from app.core.config import cfg
from app.clients.embedding_client import GenericEmbeddingClient
from app.clients.llm_client import GenericLLMClient
from app.clients.qdrant_vector_client import QdrantVectorClient


from app.clients.base_client import IEmbeddingClient, ILLMClient, IVectorClient


class ClientFactory:
    """
    Builds and stores all client instances (embedding, LLM, vector DB).
    Uses configuration values from cfg.*_provider.
    """

    @staticmethod
    def build_embedding() -> IEmbeddingClient:
        provider = cfg.embedding_provider.lower()
        log.info(f"🔧 Initializing embedding provider: {provider}")

        #extendable switch
        if provider in ("default", "generic"):
            return GenericEmbeddingClient()
    
        #elif provider == "openai":
            #from app.providers.openai.embedding_client import OpenAIEmbeddingClient
            #return OpenAIEmbeddingClient()
        #elif provider == "local":
            #from app.providers.local.embedding_client import LocalEmbeddingClient
            #return LocalEmbeddingClient()
    
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")


    @staticmethod
    def build_llm() -> ILLMClient:
        provider = cfg.llm_provider.lower()
        log.info(f"🔧 Initializing LLM client: {provider}")

        if provider in ("default", "generic"):
            return GenericLLMClient()
        
        else:
            raise ValueError(f"Unknown LLM provider : {provider}")

    
    @staticmethod
    def build_vector() -> IVectorClient:
        backend = cfg.vector_backend.lower()
        log.info(f"🔧 Initializing vector DB backend: {backend}")

        if backend in("qdrant", "default"):
            return QdrantVectorClient()
        
        
        else:
            raise ValueError(f"Unknown vector backend: {backend}")
        

    #  Aggregate builder ===============
    @staticmethod
    def build_all() -> dict[str, object]:
        """
         Returns a dict with all clients initialized.
        Example:
            {
                "embedding": <GenericEmbeddingClient>,
                "llm": <GenericLLMClient>,
                "vector": <QdrantVectorClient>,
            }
        """
        log.info(f" Building all client instances...")

        return {
            "embedding": ClientFactory.build_embedding(),
            "llm": ClientFactory.build_llm(),
            "vector": ClientFactory.build_vector(),
        }
    
    