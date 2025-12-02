# app/splitting/splitter.py
from typing import List, Optional, Literal, Tuple
import asyncio
import re
import numpy as np


from .token_counter import TokenCounter
from .semantic_chunker import SemanticChunker
from .sentence_splitter import split_sentences
from .metadata import ChunkMetadata


from app.core.logger import log
from app.core.config import cfg 

class SmartTextSplitter:
    """ Advanced text splitter combining multiple  strategies."""
   
    def __init__(
            self,
            max_chunk_size: int = 1000,
            max_tokens: Optional[int] = None,
            overlap_sentences: int = 1,
            min_sentences_length: int = 10,
            strategy: Literal["semantic", "token-aware", "sentence", "auto"] = "auto",
            preserve_formatting: bool = False,
            return_metadata: bool = False,
            max_chunks: Optional[int] = None,
            semantic_threshold: Optional[float] = None,
            model_name: Optional[str] = None,
    ):
        """
         SmartTextSplitter — orchestrates multi-strategy text chunking.

        Args:
            max_chunk_size: Maximum characters per chunk.
            max_tokens: Maximum tokens per chunk (fallback if None → chunk_size/4).
            overlap_sentences: Number of sentences to overlap between chunks.
            min_sentence_length: Filter too-short sentences.
            strategy: 'semantic', 'token-aware', 'sentence', or 'auto'.
            preserve_formatting: Preserve original line breaks if True.
            return_metadata: Return list of (chunk, metadata) tuples if True.
            max_chunks: Optional hard limit for total chunks (safety).
            semantic_threshold: Minimum cosine similarity to group sentences (default from cfg).
            model_name: LLM model for token counting (default from cfg).
        """
        self.max_chunk_size = max_chunk_size
        self.max_tokens = max_tokens or (max_chunk_size // 4)
        self.overlap_sentences = overlap_sentences
        self.min_sentences_length = min_sentences_length
        self.strategy = strategy
        self.preserve_formatting =  preserve_formatting
        self.return_metadata = return_metadata
        self.max_chunks = max_chunks or 1000

        self.semantic_threshold = semantic_threshold or cfg.similarity_threshold
        self.llm_model_name = cfg.llm_model

        self.token_counter = TokenCounter(model=self.llm_model_name)
        embedding_model = cfg.embedding_model if any (
            k in cfg.embedding_model.lower()
            for k in ["mpnet", "mini", "bge"]
        ) else None
        
        self.semantic_chunker = SemanticChunker(
            model_name=embedding_model,
            similarity_threshold=self.semantic_threshold
        )
        log.info(
            f"SmartTextSplitter -> llm={self.llm_model_name}, "
            f"embedding={embedding_model}, strategy={strategy}"
            )   
  
    
    # chunking strategies
    async def _split_semantic(self, sentences: List[str], pos: int):
        chunks, position = [], pos
        groups = await self.semantic_chunker.split(sentences, self.max_chunk_size)
        for idx, (group, sim) in enumerate(groups):
            text = " ".join(group)
            tokens = self.token_counter.count(text)
            meta = ChunkMetadata(
                index = idx,
                char_count = len(text),
                token_count = tokens,
                sentence_count = len(group),
                start_position = position,
                end_position = position + len(text),
                has_overlap = False,
                semantic_score = sim,
            )
            chunks.append((text, meta))
            position += len(text)
        return chunks
    

    async def _split_tokens(self, text:str, pos:int):
        sents = split_sentences(
            text,
            min_length=self.min_sentences_length,
        )
        chunks, current, tokens, position = [], [], 0, pos
        for s in sents:
            s_tokens = self.token_counter.count(s)
            if tokens + s_tokens > self.max_tokens and current:
                joined = " ".join(current)
                meta = ChunkMetadata(
                    index=len(chunks),
                    char_count=len(joined),
                    token_count=tokens,
                    sentence_count=len(current),
                    start_position=position,
                    end_position=position + len(joined),
                    has_overlap=self.overlap_sentences > 0,
                )
                chunks.append((joined, meta))                               
                position += len(joined)
                current, tokens = [s], s_tokens
            else:
                current.append(s)
                tokens += s_tokens
        if current:
            joined = " ".join(current)
            meta = ChunkMetadata(
                index=len(chunks),
                char_count=len(joined),
                token_count=tokens,
                sentence_count=len(current),
                start_position=position,
                end_position=position + len(joined),
                has_overlap=self.overlap_sentences > 0,                
            )
            chunks.append((joined, meta))
        return chunks
    

    # main split orchestrator
    async def split(self, text: str) -> List[str] | List[Tuple[str, ChunkMetadata]]:
        if not text or not isinstance(text, str):
            log.warning("Empty or invalid input.")
            return []
        
        text = text.strip()
        if not self.preserve_formatting:
            text = re.sub(r'[\t]+', ' ', text)

        sentences = split_sentences(
            text,
            min_length=self.min_sentences_length,
        )
        if not sentences:
            return []
        strategy = self._choose_strategy(sentences)
        log.info(f"Splitter strategy: {strategy}")

        if strategy == "semantic":
            chunks = await self._split_semantic(sentences, 0)
        elif strategy == "token-aware":
            chunks = await self._split_tokens(text, 0)
        else:
            chunks = await self._split_tokens(text, 0) # fallback

        if len(chunks) > self.max_chunks:
            log.warning(f"Triming chunks: {len(chunks)} -> {self.max_chunks}")
            chunks = chunks[:self.max_chunks]

        return chunks if self.return_metadata else [c for c, _ in chunks]
    

    def _choose_strategy(self, sentences: List[str]) ->  str:
        if self.strategy != "auto":
            return self.strategy
        if self.semantic_chunker.is_available() and len(sentences) > 5:
            return "semantic"
        if len(sentences) > 50:
            return "token-aware"
        return "sentence"

    

    def split_sync(
            self,
            text: str,
            #metadata: Optional[dict] = None,
    ) -> List[str]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.split(text))
    

    def get_stats(self, chunks: List[str]) -> dict:
        if not chunks:
            return {k: 0 for k in [
                "total_chunks", "total_chars", "total_tokens", "avg_chunk_chars",
                "avg_chunk_tokens", "min_chunk_size", "max_chunk_size",
                "min_token_count", "max_token_count"
            ]}
        sizes = [len(c) for c in chunks]
        tokens = [self.token_counter.count(c) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "total_chars": sum(sizes),
            "total_tokens": sum(tokens),
            "avg_chunk_chars": sum(sizes) // len(chunks),
            "avg_chunk_tokens": sum(tokens) // len(chunks),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "min_token_count": min(tokens),
            "max_token_count": max(tokens),
            "median_chunk_size": int(np.median(sizes)),
            "median_token_count": int(np.median(tokens)),
 
        }

