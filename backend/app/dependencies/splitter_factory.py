#app/dependencies/splitter_factory.py

from functools import lru_cache

from app.splitting.splitter import SmartTextSplitter
from app.splitting.token_counter import TokenCounter
from app.splitting.semantic_chunker import SemanticChunker
from app.splitting.sentence_splitter import split_sentences
from app.splitting.metadata import ChunkMetadata

from app.core.config import cfg

@lru_cache(maxsize=1)
def get_splitter() -> SmartTextSplitter:
    """Reusable, cached SmartTextSplitter instance."""
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