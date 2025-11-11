#app/clients.py
from qdrant_client import QdrantClient
import httpx
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

load_dotenv()

@dataclass
class QdrantConfig:
    host: str = os.getenv("QDRANT_HOST", "qdrant")
    port: int = int(os.getenv("QDRANT_PORT", 6333))
    score_threshold: float = float(os.getenv("QDRANT_SCORE_THRESHOLD", 0.3))
    collection: str = os.getenv("QDRANT_COLLECTION", "docs")

@dataclass
class OllamaConfig:
    """Global Ollama + Qdrant config."""
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    llm_model: str = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    

    @classmethod
    def reload(cls):
        """Reload .env values without restarting container."""        
        load_dotenv(override=True)
        return cls()
    
@dataclass
class SearchSettings:
    top_k: int = int(os.getenv("TOP_K", 3))    

# ========= GLOBAL CONFIG =========  
@dataclass
class GlobalConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    searchsettings: SearchSettings = field(default_factory=SearchSettings)

    def __repr__(self):
        return (
            f"Ollama({self.ollama.llm_model}), "
            f"Qdrant({self.qdrant.collection}:{self.qdrant.port}), "
            f"top_k={self.searchsettings.top_k}"
        )
    
# === Config instance ===    
cfg = GlobalConfig()

log = logging.getLogger(__name__)
log.info(f"⚙️ Loaded Ollama config: {cfg}")


# === QDRANT clients(real objects) ===
qdrant = QdrantClient(
    host=cfg.qdrant.host,
    port=cfg.qdrant.port,
    timeout=60.0
)

# === Shared HTTP client ===
http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
