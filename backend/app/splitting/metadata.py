#app/splitting/metadata.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ChunkMetadata:
    """Metadata for a single text chunk."""
    index: int
    char_count: int
    token_count: int
    sentence_count: int
    start_position: int
    end_position: int
    has_overlap: bool
    semantic_score: Optional[float] = None