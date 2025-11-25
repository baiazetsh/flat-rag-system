# app/tests/test_upload.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.api
def test_upload_missing_file(test_client):
    response = test_client.post("/upload/upload_docs", data={})
    assert response.status_code == 422  # FastAPI возвращает 422, если File() отсутствует


@pytest.mark.api
@pytest.mark.asyncio
async def test_upload_valid_txt(test_client):
    # Создаём мок-клиенты
    mock_embedding_client = AsyncMock()
    mock_embedding_client.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

    mock_vector_client = AsyncMock()
    # upsert_vectors вызывает vector_client.upsert — мокаем, чтобы не падало
    mock_vector_client.upsert = AsyncMock()

    # Мокаем зависимости как функции, возвращающие клиентов
    with patch("app.routes.upload.embedding_client_dependency", return_value=mock_embedding_client), \
         patch("app.routes.upload.vector_client_dependency", return_value=mock_vector_client), \
         patch("app.dependencies.splitter_factory.get_splitter") as mock_get_splitter:

        mock_splitter = AsyncMock()
        mock_splitter.split = AsyncMock(return_value=["Chunk 1", "Chunk 2"])
        mock_splitter.get_stats = AsyncMock(return_value={"total_chunks": 2})
        mock_get_splitter.return_value = mock_splitter

        response = test_client.post(
            "/upload/upload_docs",
            files={"file": ("test.txt", b"Hello world")},
            data={
                "collection": "docs",
                "chunk_size": "1000",
                "overlap": "1",
                "max_chunks": "100",
            }
        )

        assert response.status_code in (200, 201)
        assert "filename" in response.json()