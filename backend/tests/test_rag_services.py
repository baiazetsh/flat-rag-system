# backend/tests/test_rag_services.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.services.rag_services import _get_embedding, generate_rag_answer


@pytest.mark.unit
class TestGetEmbedding:
    """Tests for _get_embedding function"""

    @pytest.mark.asyncio
    async def test_get_embedding_success(self, mock_ollama_embedding):
        """Test successful embedding generation"""
        with patch("app.services.rag_services.http_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_ollama_embedding)
            
            embedding, elapsed = await _get_embedding("test text")
            
            assert len(embedding) == 768
            assert elapsed > 0

    @pytest.mark.asyncio
    async def test_get_embedding_empty_response(self):
        """Test handling of empty embedding response"""
        with patch("app.services.rag_services.http_client") as mock_http:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"embedding": []}
            mock_http.post = AsyncMock(return_value=mock_response)
            
            with pytest.raises(Exception):
                await _get_embedding("test")


@pytest.mark.integration
class TestGenerateRAGAnswer:
    """Tests for full RAG pipeline"""

    @pytest.mark.asyncio
    async def test_rag_pipeline_no_results(self, mock_ollama_embedding):
        """Test RAG pipeline with no search results"""
        with patch("app.services.rag_services.http_client") as mock_http, \
             patch("app.services.rag_services.qdrant") as mock_qdrant:
            
            mock_http.post = AsyncMock(return_value=mock_ollama_embedding)
            mock_qdrant.search = MagicMock(return_value=[])
            
            result = await generate_rag_answer(
                query="Nonexistent query",
                collection="docs"
            )
            
            assert result["answer"] == "No relevant documents found."
            assert result["context_used"] == 0

    @pytest.mark.asyncio
    async def test_rag_pipeline_empty_query(self):
        """Test RAG pipeline with empty query"""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await generate_rag_answer(query="", collection="docs")