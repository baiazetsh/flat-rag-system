# backend/tests/test_models.py
import pytest
from pydantic import ValidationError
from app.models import AskRequest, EmbedRequest, SearchRequest, RAGRequest


class TestAskRequest:
    """Tests for AskRequest model"""

    def test_valid_request(self):
        """Test valid ask request"""
        request = AskRequest(prompt="What is AI?")
        assert request.prompt == "What is AI?"

    def test_missing_prompt(self):
        """Test missing required field"""
        with pytest.raises(ValidationError):
            AskRequest()


class TestEmbedRequest:
    """Tests for EmbedRequest model"""

    def test_valid_request(self):
        """Test valid embed request"""
        request = EmbedRequest(text="Sample text to embed")
        assert request.text == "Sample text to embed"

    def test_empty_text(self):
        """Test with empty text"""
        request = EmbedRequest(text="")
        assert request.text == ""


class TestSearchRequest:
    """Tests for SearchRequest model"""

    def test_valid_request(self):
        """Test valid search request"""
        request = SearchRequest(query="test query", top_k=5)
        assert request.query == "test query"
        assert request.top_k == 5

    def test_default_top_k(self):
        """Test default top_k value"""
        request = SearchRequest(query="test")
        assert request.top_k == 3


class TestRAGRequest:
    """Tests for RAGRequest model"""

    def test_valid_request_full(self):
        """Test valid RAG request with all fields"""
        request = RAGRequest(
            query="What is Alice doing?",
            top_k=5,
            collection="alice_collection"
        )
        assert request.query == "What is Alice doing?"
        assert request.top_k == 5
        assert request.collection == "alice_collection"

    def test_default_values(self):
        """Test default values for optional fields"""
        request = RAGRequest(query="test query")
        assert request.query == "test query"
        assert request.top_k is None
        assert request.collection is None

    @pytest.mark.parametrize("query,top_k,collection", [
        ("What is AI?", 3, "docs"),
        ("Who is Alice?", 5, "alice"),
        ("Test query", None, None),
    ])
    def test_various_combinations(self, query, top_k, collection):
        """Test various parameter combinations"""
        request = RAGRequest(query=query, top_k=top_k, collection=collection)
        assert request.query == query
        assert request.top_k == top_k
        assert request.collection == collection