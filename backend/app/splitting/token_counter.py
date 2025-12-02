#app/splitting/token_counter.py

from functools import lru_cache

from app.core.config import cfg
from app.core.logger import log

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE =True
except ImportError as e:
    AutoTokenizer = None
    HF_AVAILABLE = False
    log.warning(f"Rransformers unavailable: {e}")
except Exception as e:
    AutoTokenizer  =None
    HF_AVAILABLE = False
    #PreTainedTokenizerBase = None
    log.warning(f"Unexpected error importing transformers: {e}")



def map_ollama_to_hf(model_name: str) -> str:
    """
    Map Ollama model names (gemma:2b, qwen:1.5b) → HuggingFace names.
    Adjust manually if needed.
    """
    name = model_name.lower()

    if "gemma" in name:
        size = name.split(":")[-1]
        return f"google/gemma-{size}"
    
    if "qwen2.5" in name:        
        size_part = name.split(":")[-1].split("-")[0]
        return f"Qwen/qwen2.5-{size_part.upper()}"
   
    if "qwen2" in name:
        size_part = name.split(":")[-1].split("-")[0]
        return f"Qwen/Qwen2-{size_part.upper()}"
    if "qwen" in name:
        size_part = name.split(":")[-1].split("-")[0]
        return f"Qwen/Qwen-{size_part.upper()}"
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