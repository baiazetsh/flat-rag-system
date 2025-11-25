# backend/app/tests/test_upload.py
"""
Tests for /api/upload:
- file upload
- validation
- vectorization pipeline triggers
"""

import pytest
from unittest.mock import patch, AsyncMock


@pytest.mark.api
def test_upload_missing_file(test_client):
    response = test_client.post("/api/upload", files={})
    assert response.status_code == 400


@pytest.mark.api
def test_upload_valid_txt(test_client):
    with patch("app.routes.upload.handle_uploaded_file", return_value=True):
        response = test_client.post(
            "/api/upload",
            files={"file": ("test.txt", b"Hello world")}
        )

        assert response.status_code in (200, 201)
