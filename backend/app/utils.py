#app/utils.py

import re


def chunk_text_by_sentences(text: str, max_chunk_size: int = 1000, overlap_sentences: int = 1):
        """
        Splits text into chunks by sentence, avoiding mid-phrase breaks.
        Returns a list of strings—chunks.
         """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current_chunk = []

        for sent in sentences:
            if sum(len(s) for s in current_chunk) + len(sent) <= max_chunk_size:
                current_chunk.append(sent)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                current_chunk.append(sent)

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


