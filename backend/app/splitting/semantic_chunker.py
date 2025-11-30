#app/splitting/semantic_chunker.py
from sentence_transformers import SentenceTransformer
import numpy as np

from app.core.logger import log
from app.core.config import cfg 
from typing import List, Optional, Literal, Tuple

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