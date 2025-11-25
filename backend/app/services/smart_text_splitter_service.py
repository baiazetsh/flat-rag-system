# backend/app/services/smart_text_splitter_service.py v1.0.1
"""
Smart Text Splitter Service with Semantic and Token-Aware Chunking (optimized).

Combines multiple strategies:
- Semantic chunking: splits by meaning using embeddings
- Token-aware chunking: respects LLM token limits
- Traditional sentence-based splitting: fast fallback
"""
from __future__ import annotations
import re
import numpy as np
from typing import List, Optional, Literal, Tuple
from dataclasses import dataclass
import asyncio
from sentence_transformers import SentenceTransformer
from functools import lru_cache
try:
    from transformers import AutoTokenizer
    HF_AVAILABLE =True
except ImportError:
    HF_AVAILABLE = False

from app.core.logger import log
from app.core.config import cfg 

def map_ollama_to_hf(model_name: str) -> str:
    """
    Map Ollama model names (gemma:2b, qwen:1.5b) → HuggingFace names.
    Adjust manually if needed.
    """
    name = model_name.lower()

    if "gemma" in name:
        size = name.split(":")[-1]
        return f"google/gemma-{size}"
    if "qwen" in name:
        size = name.split(":")[-1]
        return f"Qwen/qwewn1.5-{size.upper()}"
    return model_name

@lru_cache(maxsize=1)
def load_hf_tokenizer(model_name: str):
    if not HF_AVAILABLE:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        log.info(f"HF Tokenizer loaded: {model_name}")
        return tok
    except Exception as e:
        log.warning(f"Failed to load HF tokenizer '{model_name}': {e}")
        return None


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

class TokenCounter:
    """Token counter using HuggingFace tokenizer or fallback heuristics."""

    def __init__(self, model: str | None = None):
        ollama_name = model or cfg.llm_model
        hf_name = map_ollama_to_hf(ollama_name)
        self.tokenizer = load_hf_tokenizer(hf_name)

        if self.tokenizer:
            log.info(f"Using HuggingFace tokenizer for '{hf_name}'")
        else:
            log.warning("HF tokenizer unavailable — using fallback token counting.")

    def count(self, text: str) -> int:
        if not text:
            return 0

        # BEST: HF tokenizer
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                log.warning(f"HF token count error: {e}")

        # FALLBACK (UTF-8 length heuristic, more accurate than yours)
        utf_len = len(text.encode("utf-8"))
        return max(1, utf_len // 2)       
  
        
class SemanticChunker:
    """
    Splits text by semantic similarity using sentence embeddings.
    Uses sentence-transformers for coherent chunk grouping.
    """

    def __init__(
            self,
            model_name: str | None = None,
            similarity_threshold: float = cfg.similarity_threshold,
            show_progress_bar: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.show_progress_bar = bool(cfg.show_progress_bar)
        self.model = None

        try:
            if model_name:
                self.model = SentenceTransformer(model_name)
            else:
                log.warning("No sentence-transformer model configured. Semantic mode disabled")
            log.info(f"SemanticChunker initialized with {model_name}")
        except ImportError:
            log.warning("sentence-transformers not installed. Semantic mode disabled.")
        except Exception as e:
            log.error(f"Failed to load embedding model: {e}")
            self.model = None


    def is_available(self) -> bool:
        return self.model is not None
    
    def  _calculate_similarity(self, emb1, emb2) -> float:
        from sklearn.metrics.pairwise import cosine_similarity
        
        return cosine_similarity(
            np.array(emb1).reshape(1, -1),
            np.array(emb2).reshape(1, -1)
        )[0][0]
    
    async def split(
            self,
            sentences: List[str],
            max_chunk_size: int
    ) -> List[Tuple[List[str], float]]:
        """Group sentences by semantic similarity."""
        #if not self.model:
         #   log.warning("SemanticChunker not initialized — skipping semantic grouping.")
         #   return [(sentences, 0.0)]
        #if not self.is_available() or len(sentences) <= 1:
        #    return [(sentences, 0.0)]
        if not self.model or len(sentences) <=1:
            log.warning("SemanticChunker not initialized — using fallback.")
            return [(sentences, 0.0)]

        
        try:
            embeddings = self.model.encode(
                sentences,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=cfg.show_progress_bar,
            )

            chunks = []
            current_chunk = [sentences[0]]
            similarities = []
            current_size = len(sentences[0])

            for i in range(1, len(sentences)):
                sent = sentences[i]
                sent_len = len(sent)
                sim = self._calculate_similarity(embeddings[i - 1], embeddings[i])

                if sim >= self.similarity_threshold and current_size + sent_len <= max_chunk_size:
                    current_chunk.append(sent)
                    current_size += sent_len
                    similarities.append(sim)
                else:
                    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
                    chunks.append((current_chunk, avg_sim))
                    current_chunk, similarities = [sent], []
                    current_size = sent_len

            if current_chunk:
                avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
                chunks.append((current_chunk, avg_sim))

                log.info(
                    f"Semantic chunking: {len(sentences)} sentences -> {len(chunks)} chunks."
                    f"Avg sim: {np.mean([s for _, s in chunks]):.2f}"                    
                )
                return chunks
            
        except Exception as e:
            log.error(f"Semantic  chunking failed: {e}")
            return [(sentences, 0.0)]


class SmartTextSplitter:
    """ Advanced text splitter combining multiple  strategies."""
    SENTENCE_BOUNDARIES = re.compile(
        r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ])|'
        r'(?<=[.!?]["\'»)])\s+(?=[A-ZА-ЯЁ])|'
        r'(?<=\n)\n+'
    )
    PARAGRAPH_BOUNDARIES = re.compile(r'\n\s*\n+')
    PROTECTED_PATTERNS = [
        re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|vs|etc|e\.g|i\.e)\.\s+', re.IGNORECASE),
        re.compile(r'\d+\.\d+'),
        re.compile(r'https?://[^\s]+'),
        re.compile(r'```[\s\S]*?```'),
        re.compile(r'`[^`]+`'),
    ]

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

    # sentence splitting
    def _split_sentences(self, text: str) -> List[str]:
        try:
            text, protected = self._protect_patterns(text)
            sentences = [self._restore_patterns(s, protected)
                            for s in self.SENTENCE_BOUNDARIES.split(text)
                            if len(s.strip()) >= self.min_sentences_length]
            return sentences or [text]
        except Exception as e:
            log.error(f"Semantic chunking failed ({e}), fallback to token-aware.")
            return [text]

    
    def _protect_patterns(self, text: str):
        protected = {}
        for i, pattern in enumerate(self.PROTECTED_PATTERNS):
            for m in pattern.finditer(text):
                ph = f"__P{i}_{len(protected)}__"
                protected[ph] = m.group(0)
                text = text.replace(m.group(0), ph, 1)
        return text, protected
    
    def _restore_patterns(self, text: str, protected: dict) -> str:
        for ph, orig in protected.items():
            text = text.replace(ph, orig)
        return text
    
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
        sents = self._split_sentences(text)
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
            log.warning("Empty or invalidinput.")
            return []
        
        text = text.strip()
        if not self.preserve_formatting:
            text = re.sub(r'[\t]+', ' ', text)

        sentences = self._split_sentences(text)
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


        
    
    
        

    


            