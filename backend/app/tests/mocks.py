# backend/app/tests/conftest.py (add near the bottom)

from .mocks import (
    FakeEmbeddingClient,
    FakeLLMClient,
    FakeVectorClient,
    make_chunks,
)


@pytest.fixture
def fake_embedding_client():
    """Reusable fake embedding client (no network)."""
    return FakeEmbeddingClient(dim=768, value=0.1)


@pytest.fixture
def fake_llm_client():
    """Reusable fake LLM client (no network)."""
    return FakeLLMClient(response_text="This is a fake LLM response.")


@pytest.fixture
def fake_vector_client():
    """Reusable in-memory vector client."""
    return FakeVectorClient()


@pytest.fixture
def sample_chunks():
    """Common chunks set used across multiple tests."""
    return make_chunks(n=3, base_text="Sample chunk", start_score=0.95, step=0.03)
