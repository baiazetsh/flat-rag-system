#app/splitting/sentence_splitter.py
import re
from typing import List

from app.core.logger import log


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


def protect_patterns(text: str):
        protected = {}
        for i, pattern in enumerate(PROTECTED_PATTERNS):
            for m in pattern.finditer(text):
                ph = f"__P{i}_{len(protected)}__"
                protected[ph] = m.group(0)
                text = text.replace(m.group(0), ph, 1)
        return text, protected
    

def restore_patterns(text: str, protected: dict) -> str:
    for ph, orig in protected.items():
        text = text.replace(ph, orig)
    return text


# sentence splitting
def split_sentences(text: str, min_length: int = 10) -> List[str]:
    try:
        text, protected = protect_patterns(text)
        sentences = [
            restore_patterns(s, protected)
            for s in SENTENCE_BOUNDARIES.split(text)
            if len(s.strip()) >= min_length
        ]
        return sentences or [text]
    except Exception as e:
        log.error(f"Semantic splitting failed ({e}), fallback to token-aware.")
        return [text]