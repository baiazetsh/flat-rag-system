# backend/tests/__init__.py
"""
Test suite for RAG Local project.

This package contains:
- Unit tests: Fast, isolated tests for utils and models
- Integration tests: Tests for RAG pipeline with mocked services
- API tests: Tests for FastAPI endpoints

To run all tests:
    pytest tests/ -v

To run specific category:
    pytest tests/ -m unit
    pytest tests/ -m integration
    pytest tests/ -m api

To run with coverage:
    pytest tests/ --cov=app --cov-report=term-missing
"""
