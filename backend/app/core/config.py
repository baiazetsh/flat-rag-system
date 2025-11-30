#app/core/config.py v1.0.1

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel


# Nested collection config=========
class VectorCollectionConfig(BaseModel):
    name: str = "docs"  # default collection
    size: int = 1024    # vector dimension
    distance: str = "cosine" # similarity metric
    recreate_if_exists: bool = False   # userful for CI or dev mode
    score_threshold: float = 0.3


# main settings
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    app_name: str = "RAG Local API"

    #llm
    llm_provider: str = "generic"
    llm_url: str = "http://ollama:11434/api/generate"
    llm_model: str = "gpt-4" #gpt-4 is default value, if LLM_MODEL is not available
    
    #vector DB
    vector_backend: str = "qdrant"
    qdrant_url: str = "http://qdrant:6333"   
    collection: VectorCollectionConfig =VectorCollectionConfig()
    
    default_collection_name: str = "docs"
    
    #embedding
    embedding_url: str = "http://ollama:11434/api/embeddings"
    embedding_provider: str = "generic"
    embedding_model: str = "mxbai-embed-large"
    similarity_threshold: float = 0.75


    # Cache /misc
    redis_url: str = "redis://redis:6379/0"
    top_k: int = 30
    top_k_base: int = 30
    top_after_rrf: int = 10
    top_final: int = 3

    top_p: float = 0.8
    max_tokens: int = 2048
    repeat_penalty: float = 1.2
    seed: int = -1

    # selector GPU/CPU
    device:str = "cuda"
    device_for_reranker: str = "cpu"
    device_for_main_llm: str = "cuda"


    
    ollama_base_url: str = "http://ollama:11434"
    backend_url: str = "http://localhost:8000"

    # splitter defaults
    max_chunk_size: int = 1000
    overlap_sentences: int = 1
    max_chunks: int = 1000
    min_sentence_length: int = 10
    show_progress_bar: bool = False

    #reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_provider: str = "generic"
    reranker_url:str = "http://ollama:11434/api/embeddings"


cfg = Settings()