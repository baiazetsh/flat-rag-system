# backend/app/tests/test_clients.py
"""
Unit tests for client implementations.
Tests match ACTUAL client behavior (not idealized).
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx

from app.clients.embedding_client import GenericEmbeddingClient
from app.clients.llm_client import GenericLLMClient
from app.clients.client_factory import ClientFactory


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def mock_embedding_response():
    """Mock httpx response for Ollama embedding (new format)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "embeddings": [[float(i) / 768.0 for i in range(768)]],
        "model": "test-embed"
    }
    return mock_resp


@pytest.fixture
def mock_llm_response():
    """Mock httpx response for Ollama LLM generation."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "This is a test answer from LLM.",
        "done": True
    }
    return mock_resp


# ============================================================
# GenericEmbeddingClient Tests
# ============================================================
@pytest.mark.unit
class TestGenericEmbeddingClient:
    """Tests for embedding client."""
    
    @pytest.mark.asyncio
    async def test_embed_success_new_format(self, mock_embedding_response):
        """Test successful embedding with new Ollama format."""
        with patch("httpx.AsyncClient") as mock_httpx:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_embedding_response)
            mock_httpx.return_value = mock_client
            
            client = GenericEmbeddingClient()
            embedding = await client.embed("test text")
            
            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_embed_success_old_format(self):
        """Test old Ollama format: {\"embedding\": [...]}."""
        with patch("httpx.AsyncClient") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "embedding": [0.1] * 768
            }
            
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_httpx.return_value = mock_client
            
            client = GenericEmbeddingClient()
            embedding = await client.embed("test")
            
            assert len(embedding) == 768
    
    @pytest.mark.asyncio
    async def test_embed_empty_text(self, mock_embedding_response):
        """Test embedding with empty text."""
        with patch("httpx.AsyncClient") as mock_httpx:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_embedding_response)
            mock_httpx.return_value = mock_client
            
            client = GenericEmbeddingClient()
            embedding = await client.embed("")
            
            assert len(embedding) == 768


# ============================================================
# GenericLLMClient Tests
# ============================================================
@pytest.mark.unit
class TestGenericLLMClient:
    """Tests for LLM client."""
    
    @pytest.mark.asyncio
    async def test_generate_success_ollama_format(self, mock_llm_response):
        """Test Ollama format: {\"response\": \"...\"}."""
        with patch("httpx.AsyncClient") as mock_httpx:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_llm_response)
            mock_httpx.return_value = mock_client
            
            client = GenericLLMClient()
            response = await client.generate("Test prompt")
            
            assert isinstance(response, str)
            assert response == "This is a test answer from LLM."
    
    @pytest.mark.asyncio
    async def test_generate_openai_format(self):
        """Test OpenAI-style response parsing."""
        with patch("httpx.AsyncClient") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [
                    {"message": {"content": "OpenAI response"}}
                ]
            }
            
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_httpx.return_value = mock_client
            
            client = GenericLLMClient()
            response = await client.generate("test")
            
            assert response == "OpenAI response"
    
    @pytest.mark.asyncio
    async def test_generate_fallback(self):
        """Test fallback to str(data) for unknown format."""
        with patch("httpx.AsyncClient") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"unknown_key": "raw content"}
            
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_httpx.return_value = mock_client
            
            client = GenericLLMClient()
            response = await client.generate("test")
            
            assert "unknown_key" in response
            assert "raw content" in response


# ============================================================
# ClientFactory Tests
# ============================================================
@pytest.mark.unit
class TestClientFactory:
    """Tests for ClientFactory."""
    
    @patch("app.clients.qdrant_vector_client.QdrantClient")
    def test_build_embedding(self, MockQdrantClient):
        """Test embedding client creation."""
        client = ClientFactory.build_embedding()
        
        assert client is not None
        assert isinstance(client, GenericEmbeddingClient)
        assert hasattr(client, "embed")
    
    @patch("app.clients.qdrant_vector_client.QdrantClient")
    def test_build_llm(self, MockQdrantClient):
        """Test LLM client creation."""
        client = ClientFactory.build_llm()
        
        assert client is not None
        assert isinstance(client, GenericLLMClient)
        assert hasattr(client, "generate")
    
    @patch("app.clients.qdrant_vector_client.QdrantClient")
    def test_build_vector(self, MockQdrantClient):
        """Test vector client creation."""
        from app.clients.qdrant_vector_client import QdrantVectorClient
        
        client = ClientFactory.build_vector()
        
        assert client is not None
        assert isinstance(client, QdrantVectorClient)
        assert hasattr(client, "search")
        assert hasattr(client, "upsert")
    
    @patch("app.clients.qdrant_vector_client.QdrantClient")
    def test_build_all(self, MockQdrantClient):
        """Test building all clients at once."""
        from app.clients.qdrant_vector_client import QdrantVectorClient
        
        clients = ClientFactory.build_all()
        
        assert "embedding" in clients
        assert "llm" in clients
        assert "vector" in clients
        
        assert isinstance(clients["embedding"], GenericEmbeddingClient)
        assert isinstance(clients["llm"], GenericLLMClient)
        assert isinstance(clients["vector"], QdrantVectorClient)