# backend/app/tests/conftest.py
"""
Centralized test fixtures for RAG Local project.
Provides mocks for all external services (Ollama, Qdrant).
"""
import pytest
import asyncio
from typing import Generator
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import cfg


# ============================================================
# Event Loop
# ============================================================
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ============================================================
# FastAPI Test Client
# ============================================================
@pytest.fixture
def test_client() -> Generator:
    """Sync test client for FastAPI endpoints."""
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ============================================================
# Mock Embedding Client (IEmbeddingClient)
# ============================================================
@pytest.fixture
def mock_embedding_client():
    """Mock GenericEmbeddingClient."""
    mock = AsyncMock()
    mock.embed = AsyncMock(return_value=[0.1] * 768)
    return mock


# ============================================================
# Mock LLM Client (ILLMClient)
# ============================================================
@pytest.fixture
def mock_llm_client():
    """Mock GenericLLMClient."""
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value="This is a test LLM response.")
    return mock


# ============================================================
# Mock Vector Client (IVectorClient)
# ============================================================
@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantVectorClient."""
    mock = AsyncMock()
    
    # Mock search
    mock.search = AsyncMock(return_value=[
        {"text": "Test document 1", "score": 0.95},
        {"text": "Test document 2", "score": 0.85},
    ])
    
    # Mock list_collections
    mock.list_collections = AsyncMock(return_value=["docs", "test_collection"])
    
    # Mock create_collection
    mock.create_collection = AsyncMock(return_value=None)
    
    # Mock delete_collection
    mock.delete_collection = AsyncMock(return_value=None)
    
    # Mock upsert
    mock.upsert = AsyncMock(return_value=None)
    
    # Mock aclose
    mock.aclose = AsyncMock(return_value=None)
    
    return mock


# ============================================================
# Sample Data
# ============================================================
@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Alice was beginning to get very tired of sitting by her sister on the
    bank, and of having nothing to do. Once or twice she had peeped into
    the book her sister was reading, but it had no pictures or conversations
    in it. 'And what is the use of a book,' thought Alice 'without pictures
    or conversations?'
    """


@pytest.fixture
def sample_chunks():
    """Pre-split text chunks."""
    return [
        "Alice was beginning to get very tired of sitting by her sister.",
        "Once or twice she had peeped into the book her sister was reading.",
        "It had no pictures or conversations in it.",
    ]


@pytest.fixture
def sample_vectors():
    """Sample embedding vectors."""
    return [
        [0.1 * i] * 768 for i in range(3)
    ]


# ============================================================
# Config Reset
# ============================================================
@pytest.fixture(autouse=True)
def reset_config():
    """Reset config before each test."""
    original_collection = cfg.collection.name
    original_top_k = cfg.top_k
    yield
    cfg.collection.name = original_collection
    cfg.top_k = original_top_k