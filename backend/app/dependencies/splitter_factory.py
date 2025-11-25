#app/dependencies/splitter_factory.py
from functools import lru_cache
from app.services.smart_text_splitter_service import SmartTextSplitter
from app.core.config import cfg

@lru_cache(maxsize=1)
def get_splitter() -> SmartTextSplitter:
    """Reusable splitter instance configured via .env"""
    splitter = SmartTextSplitter(
        max_chunk_size=cfg.max_chunk_size,
        overlap_sentences=cfg.overlap_sentences,
        strategy="auto",
        semantic_threshold=cfg.similarity_threshold,
        model_name=cfg.llm_model,
        preserve_formatting=False,
        return_metadata=False,
        max_chunks=cfg.max_chunks,
    )
    return splitter