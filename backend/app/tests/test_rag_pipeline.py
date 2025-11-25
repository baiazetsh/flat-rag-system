# backend/app/tests/test_rag_pipeline.py
"""
End-to-end tests for the RAG Pipeline:
- splitting
- embedding
- semantic search
- context building
- prompting
- LLM generation
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.services.rag_service import RAGService
from app.services.smart_text_splitter_service import SmartTextSplitter


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rag_pipeline_basic_flow(mock_app, mock_embedding_client, mock_llm_client, mock_qdrant_client):
    """Full RAG flow with mocked clients."""

    # Mock Qdrant search returning 2 chunks
    mock_qdrant_client.search = AsyncMock(return_value=[
        {"text": "Chunk A", "score": 0.95},
        {"text": "Chunk B", "score": 0.93},
    ])

    # Patch DI
    with patch("app.services.rag_service.get_embedding_client", return_value=mock_embedding_client), \
         patch("app.services.rag_service.get_llm_client", return_value=mock_llm_client), \
         patch("app.services.rag_service.get_vector_client", return_value=mock_qdrant_client):

        rag = RAGService()

        result = await rag.search_with_llm(
            app=mock_app,
            query="What is test?",
            top_k=2
        )

        assert "answer" in result
        assert result["answer"]
        assert "context" in result
        assert len(result["context"]) == 2
