# backend/tests/conftest.py
import pytest
import asyncio
from typing import Generator
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from app.main import app
from app.clients import cfg

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_client() -> Generator:
    """Sync test client for FastAPI"""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_ollama_embedding():
    """Mock Ollama embedding response"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "embedding": [0.1] * 768  # 768-dim vector
    }
    return mock_response

@pytest.fixture
def mock_ollama_generate():
    """Mock Ollama generate response"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": "This is a test answer from LLM."
    }
    return mock_response

@pytest.fixture
def mock_qdrant_search():
    """Mock Qdrant search results"""
    mock_hits = []
    for i in range(2):
        mock_hit = MagicMock()
        mock_hit.score = 0.95 - (i * 0.1)
        mock_hit.payload = {
            "text": f"Test document {i+1}",
            "source": "test.txt"
        }
        mock_hit.id = str(i+1)
        mock_hits.append(mock_hit)
    return mock_hits

@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """
    Alice was beginning to get very tired of sitting by her sister on the
    bank, and of having nothing to do. Once or twice she had peeped into
    the book her sister was reading, but it had no pictures or conversations
    in it.
    """

@pytest.fixture(autouse=True)
def reset_config():
    """Reset config before each test"""
    original_collection = cfg.qdrant.collection
    yield
    cfg.qdrant.collection = original_collection