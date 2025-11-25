# backend/app/tests/test_services.py
"""
Unit tests for service layer (DI-based):
- embeddings_service
- llm_service
- vector_services
- generate_rag_answer
"""
import pytest
from unittest.mock import AsyncMock

from app.services.embeddings_service import get_embedding
from app.services.llm_service import generate_answer
from app.services import vector_services as vs
from app.services.generate_rag_answer import generate_rag_answer


# ============================================================
# Embeddings Service Tests
# ============================================================
@pytest.mark.unit
class TestEmbeddingsService:
    """Tests for embeddings service."""
    
    @pytest.mark.asyncio
    async def test_get_embedding_success(self, mock_embedding_client):
        """Test successful embedding generation."""
        vector = await get_embedding("test text", mock_embedding_client)
        
        assert isinstance(vector, list)
        assert len(vector) == 768
        mock_embedding_client.embed.assert_called_once_with("test text")
    
    @pytest.mark.asyncio
    async def test_get_embedding_empty_text(self, mock_embedding_client):
        """Test embedding with empty text."""
        vector = await get_embedding("", mock_embedding_client)
        
        assert isinstance(vector, list)
        assert len(vector) == 768


# ============================================================
# LLM Service Tests
# ============================================================
@pytest.mark.unit
class TestLLMService:
    """Tests for LLM service."""
    
    @pytest.mark.asyncio
    async def test_generate_answer_success(self, mock_llm_client):
        """Test successful answer generation."""
        answer = await generate_answer(mock_llm_client, "Test prompt")
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        mock_llm_client.generate.assert_called_once_with("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_answer_empty_prompt(self, mock_llm_client):
        """Test generation with empty prompt."""
        answer = await generate_answer(mock_llm_client, "")
        
        assert isinstance(answer, str)


# ============================================================
# Vector Services Tests
# ============================================================
@pytest.mark.unit
class TestVectorServices:
    """Tests for vector database services."""
    
    @pytest.mark.asyncio
    async def test_search_in_vector_db(self, mock_qdrant_client):
        """Test vector search."""
        vector = [0.1] * 768
        
        results = await vs.search_in_vector_db(
            vector_client=mock_qdrant_client,
            vector=vector,
            top_k=5,
            collection="docs"
        )
        
        assert len(results) == 2
        assert results[0]["text"] == "Test document 1"
        assert results[0]["score"] == 0.95
    
    @pytest.mark.asyncio
    async def test_list_collections(self, mock_qdrant_client):
        """Test listing collections."""
        collections = await vs.list_collections(mock_qdrant_client)
        
        assert isinstance(collections, list)
        assert "docs" in collections
        assert "test_collection" in collections
    
    @pytest.mark.asyncio
    async def test_create_collection_new(self, mock_qdrant_client):
        """Test creating new collection."""
        mock_qdrant_client.list_collections = AsyncMock(return_value=[])
        
        result = await vs.create_collection(
            vector_client=mock_qdrant_client,
            name="new_collection",
            vector_size=768,
            distance="cosine"
        )
        
        assert result["status"] == "created"
        assert result["collection"] == "new_collection"
        mock_qdrant_client.create_collection.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_collection_exists(self, mock_qdrant_client):
        """Test creating collection that already exists."""
        result = await vs.create_collection(
            vector_client=mock_qdrant_client,
            name="docs",  # Already exists
            vector_size=768,
            distance="cosine"
        )
        
        assert result["status"] == "exists"
        mock_qdrant_client.create_collection.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_delete_collection(self, mock_qdrant_client):
        """Test deleting collection."""
        result = await vs.delete_collection(mock_qdrant_client, "docs")
        
        assert result["status"] == "deleted"
        mock_qdrant_client.delete_collection.assert_called_once_with("docs")
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_collection(self, mock_qdrant_client):
        """Test deleting nonexistent collection."""
        mock_qdrant_client.list_collections = AsyncMock(return_value=[])
        
        result = await vs.delete_collection(mock_qdrant_client, "nonexistent")
        
        assert result["status"] == "not_found"
    
    @pytest.mark.asyncio
    async def test_upsert_vectors(self, mock_qdrant_client):
        """Test upserting vectors."""
        payloads = [
            {"text": "doc1", "chunk_index": 0},
            {"text": "doc2", "chunk_index": 1},
        ]
        vectors = [[0.1] * 768, [0.2] * 768]
        
        result = await vs.upsert_vectors(
            vector_client=mock_qdrant_client,
            collection="docs",
            payloads=payloads,
            vectors=vectors
        )
        
        assert result["status"] == "ok"
        assert result["count"] == 2
        mock_qdrant_client.upsert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upsert_vectors_mismatch(self, mock_qdrant_client):
        """Test error when payloads/vectors length mismatch."""
        payloads = [{"text": "doc1"}]
        vectors = [[0.1] * 768, [0.2] * 768]
        
        with pytest.raises(ValueError, match="length mismatch"):
            await vs.upsert_vectors(
                vector_client=mock_qdrant_client,
                collection="docs",
                payloads=payloads,
                vectors=vectors
            )


# ============================================================
# RAG Answer Generation Tests
# ============================================================
@pytest.mark.unit
class TestGenerateRAGAnswer:
    """Tests for complete RAG pipeline orchestrator."""
    
    @pytest.mark.asyncio
    async def test_rag_success(
        self,
        mock_qdrant_client,
        mock_embedding_client,
        mock_llm_client
    ):
        """Test successful RAG pipeline execution."""
        result = await generate_rag_answer(
            vector_client=mock_qdrant_client,
            embedding_client=mock_embedding_client,
            llm_client=mock_llm_client,
            query="What is Alice doing?",
            top_k=5,
            collection="docs"
        )
        
        assert "query" in result
        assert "answer" in result
        assert "context_used" in result
        assert result["context_used"] == 2
        assert result["query"] == "What is Alice doing?"
    
    @pytest.mark.asyncio
    async def test_rag_no_results(
        self,
        mock_qdrant_client,
        mock_embedding_client,
        mock_llm_client
    ):
        """Test RAG when no documents found."""
        mock_qdrant_client.search = AsyncMock(return_value=[])
        
        result = await generate_rag_answer(
            vector_client=mock_qdrant_client,
            embedding_client=mock_embedding_client,
            llm_client=mock_llm_client,
            query="Unknown query",
            top_k=5,
            collection="docs"
        )
        
        assert result["answer"] == "No relevant documents found."
        assert result["context_used"] == 0
    
    @pytest.mark.asyncio
    async def test_rag_empty_query(
        self,
        mock_qdrant_client,
        mock_embedding_client,
        mock_llm_client
    ):
        """Test RAG with empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await generate_rag_answer(
                vector_client=mock_qdrant_client,
                embedding_client=mock_embedding_client,
                llm_client=mock_llm_client,
                query="",
                top_k=5,
                collection="docs"
            )
    
    @pytest.mark.asyncio
    async def test_rag_uses_config_defaults(
        self,
        mock_qdrant_client,
        mock_embedding_client,
        mock_llm_client
    ):
        """Test RAG falls back to config when top_k/collection not provided."""
        result = await generate_rag_answer(
            vector_client=mock_qdrant_client,
            embedding_client=mock_embedding_client,
            llm_client=mock_llm_client,
            query="test",
            top_k=None,
            collection=None
        )
        
        assert "answer" in result
        assert result["collection"] == "docs"  # from cfg