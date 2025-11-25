# backend/app/tests/test_models.py
"""
Unit tests for Pydantic request/response models.
Tests validation, default values, and edge cases.
"""
import pytest
from pydantic import ValidationError
from app.models import AskRequest, EmbedRequest, SearchRequest, RAGRequest
from app.core.config import cfg


# ============================================================
# AskRequest Tests
# ============================================================
@pytest.mark.unit
class TestAskRequest:
    """Tests for AskRequest model."""

    def test_valid_request(self):
        """Test valid ask request."""
        request = AskRequest(prompt="What is AI?")
        assert request.prompt == "What is AI?"

    def test_missing_prompt(self):
        """Test missing required field."""
        with pytest.raises(ValidationError) as exc_info:
            AskRequest()
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("prompt",)
        assert errors[0]["type"] == "missing"
    
    def test_empty_prompt(self):
        """Test with empty prompt (allowed by model)."""
        request = AskRequest(prompt="")
        assert request.prompt == ""
    
    def test_prompt_with_special_chars(self):
        """Test prompt with special characters."""
        special_prompt = "What is π? Explain ∫ and ∂."
        request = AskRequest(prompt=special_prompt)
        assert request.prompt == special_prompt


# ============================================================
# EmbedRequest Tests
# ============================================================
@pytest.mark.unit
class TestEmbedRequest:
    """Tests for EmbedRequest model."""

    def test_valid_request(self):
        """Test valid embed request."""
        request = EmbedRequest(text="Sample text to embed")
        assert request.text == "Sample text to embed"

    def test_empty_text(self):
        """Test with empty text (allowed)."""
        request = EmbedRequest(text="")
        assert request.text == ""
    
    def test_missing_text(self):
        """Test missing required field."""
        with pytest.raises(ValidationError) as exc_info:
            EmbedRequest()
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("text",)
    
    def test_long_text(self):
        """Test with very long text."""
        long_text = "word " * 10000
        request = EmbedRequest(text=long_text)
        assert len(request.text) > 50000


# ============================================================
# SearchRequest Tests
# ============================================================
@pytest.mark.unit
class TestSearchRequest:
    """Tests for SearchRequest model."""

    def test_valid_request_with_top_k(self):
        """Test valid search request with explicit top_k."""
        request = SearchRequest(query="test query", top_k=5)
        assert request.query == "test query"
        assert request.top_k == 5

    def test_default_top_k_from_config(self):
        """Test default top_k value comes from cfg."""
        # IMPORTANT: Value comes from cfg.top_k (default 5 in .env)
        request = SearchRequest(query="test")
        assert request.top_k == cfg.top_k  # Uses actual config value
        assert isinstance(request.top_k, int)
        assert request.top_k > 0
    
    def test_missing_query(self):
        """Test missing required query field."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest()
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("query",) for e in errors)
    
    def test_top_k_validation(self):
        """Test top_k accepts valid integers."""
        request = SearchRequest(query="test", top_k=10)
        assert request.top_k == 10
        
        request = SearchRequest(query="test", top_k=1)
        assert request.top_k == 1
    
    def test_top_k_string_coercion(self):
        """Test Pydantic coerces string to int (if lax mode)."""
        # Pydantic v2 in lax mode will coerce "5" -> 5
        request = SearchRequest(query="test", top_k="5")
        assert request.top_k == 5
        assert isinstance(request.top_k, int)


# ============================================================
# RAGRequest Tests
# ============================================================
@pytest.mark.unit
class TestRAGRequest:
    """Tests for RAGRequest model."""

    def test_valid_request_full(self):
        """Test valid RAG request with all fields."""
        request = RAGRequest(
            query="What is Alice doing?",
            top_k=5,
            collection="alice_collection"
        )
        assert request.query == "What is Alice doing?"
        assert request.top_k == 5
        assert request.collection == "alice_collection"

    def test_default_values_are_none(self):
        """Test optional fields default to None."""
        request = RAGRequest(query="test query")
        assert request.query == "test query"
        assert request.top_k is None  # Optional field
        assert request.collection is None  # Optional field

    def test_missing_query(self):
        """Test missing required query field."""
        with pytest.raises(ValidationError) as exc_info:
            RAGRequest()
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("query",) for e in errors)

    @pytest.mark.parametrize("query,top_k,collection", [
        ("What is AI?", 3, "docs"),
        ("Who is Alice?", 5, "alice"),
        ("Test query", None, None),
        ("Another test", 10, "custom_collection"),
    ])
    def test_various_combinations(self, query, top_k, collection):
        """Test various parameter combinations."""
        request = RAGRequest(query=query, top_k=top_k, collection=collection)
        assert request.query == query
        assert request.top_k == top_k
        assert request.collection == collection
    
    def test_empty_query_string_allowed(self):
        """Test that empty query string is technically allowed by Pydantic."""
        # Model allows it, but service layer should reject it
        request = RAGRequest(query="")
        assert request.query == ""
    
    def test_top_k_coercion(self):
        """Test top_k type coercion."""
        request = RAGRequest(query="test", top_k="7")
        assert request.top_k == 7
        assert isinstance(request.top_k, int)
    
    def test_collection_as_string(self):
        """Test collection accepts any string."""
        request = RAGRequest(query="test", collection="my-collection-123")
        assert request.collection == "my-collection-123"


# ============================================================
# Edge Cases & Validation Tests
# ============================================================
@pytest.mark.unit
class TestModelEdgeCases:
    """Tests for edge cases across all models."""
    
    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored by default."""
        # Pydantic v2 ignores extra fields by default
        request = AskRequest(prompt="test", extra_field="ignored")
        assert request.prompt == "test"
        assert not hasattr(request, "extra_field")
    
    def test_model_serialization(self):
        """Test model can be serialized to dict/JSON."""
        request = RAGRequest(
            query="test",
            top_k=5,
            collection="docs"
        )
        
        # Test dict conversion
        data = request.model_dump()
        assert data == {
            "query": "test",
            "top_k": 5,
            "collection": "docs"
        }
        
        # Test JSON conversion
        json_str = request.model_dump_json()
        assert isinstance(json_str, str)
        assert "test" in json_str
    
    def test_model_from_dict(self):
        """Test creating model from dict."""
        data = {"query": "test query", "top_k": 5}
        request = SearchRequest(**data)
        assert request.query == "test query"
        assert request.top_k == 5
    
    def test_unicode_handling(self):
        """Test models handle Unicode text."""
        unicode_text = "Привет мир 🌍 こんにちは"
        request = AskRequest(prompt=unicode_text)
        assert request.prompt == unicode_text
        
        request = EmbedRequest(text=unicode_text)
        assert request.text == unicode_text


# ============================================================
# Config-Dependent Behavior Tests
# ============================================================
@pytest.mark.unit
class TestModelConfigDependency:
    """Tests for behavior that depends on cfg values."""
    
    def test_search_request_uses_runtime_config(self):
        """Test that SearchRequest.top_k uses runtime cfg value."""
        # This test verifies the model reads from cfg at runtime
        request = SearchRequest(query="test")
        
        # Value should match current config
        assert request.top_k == cfg.top_k
    
    def test_rag_request_optionals_independent_of_config(self):
        """Test RAGRequest optional fields are truly optional."""
        # These should be None, not cfg values
        request = RAGRequest(query="test")
        
        assert request.top_k is None
        assert request.collection is None
        
        # They don't auto-populate from config
        # (that happens in service layer, not model)