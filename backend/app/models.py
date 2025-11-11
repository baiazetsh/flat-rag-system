#app/models.py
from pydantic import BaseModel, Field
from typing import Optional
from app.clients import cfg

# ============================================================
#  Pydantic Models (app/models.py)
#
#  These models are defined in a separate module instead of
#  app/routes/base.py to:
#   • prevent overriding built-in Python functions (like str())
#     when multiple classes use annotations such as field: str;
#   • avoid the `'str' object is not callable` runtime error;
#   • safely use Field(default_factory=...) so configuration
#     values from cfg are evaluated at runtime, not import time;
#   • centralize all input data models (AskRequest, EmbedRequest,
#     SearchRequest, RAGRequest) for cleaner imports in base.py.
#
#  After this change:
#      from app.models import AskRequest, EmbedRequest, SearchRequest, RAGRequest
#  → the built-in str() remains intact, and cfg values are loaded correctly.
# ============================================================


class AskRequest(BaseModel):
    prompt: str


class EmbedRequest(BaseModel):
    text:str


class SearchRequest(BaseModel):
    query: str
    #top_k: int = Field(default_factory=lambda: cfg.searchsettings.top_k)
    top_k: int = 3
  

class RAGRequest(BaseModel):
    query: str
    #top_k: int = Field(default_factory=lambda: cfg.searchsettings.top_k)    
    #collection: str = Field(default_factory=lambda: cfg.qdrant.collection)
    top_k: Optional[int] = None
    collection: Optional[str] = None