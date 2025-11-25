# backend/app/tests/test_llm_client.py
"""
Tests for LLM Client edge cases:
- malformed response
- empty generation
- network error
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.clients.ollama_llm_client import OllamaLLMClient


@pytest.mark.asyncio
async def test_llm_malformed_response():
    client = OllamaLLMClient()

    bad_response = MagicMock()
    bad_response.json.return_value = {}  # missing "response"
    bad_response.status_code = 200

    with pytest.raises(ValueError):
        await client._parse_response(bad_response)


@pytest.mark.asyncio
async def test_llm_empty_generation():
    client = OllamaLLMClient()

    resp = MagicMock()
    resp.json.return_value = {"response": ""}
    resp.status_code = 200

    result = await client._parse_response(resp)
    assert result == ""
