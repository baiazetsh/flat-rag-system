import pytest
from app.services.prompt_builder import build_rag_prompt, build_rag_prompt_advanced


# Фикстуры для переиспользования данных
@pytest.fixture
def sample_context():
    return "Alice is in the garden. She is watering roses."


@pytest.fixture
def sample_query():
    return "What is Alice doing in the garden?"


# ——————— Тесты для базового промпта ———————

def test_build_rag_prompt_includes_context_and_query(sample_context, sample_query):
    prompt = build_rag_prompt(sample_context, sample_query)
    
    assert sample_context in prompt
    assert sample_query in prompt


def test_build_rag_prompt_contains_strict_instructions(sample_context, sample_query):
    prompt = build_rag_prompt(sample_context, sample_query)
    
    # Ключевые элементы инструкции
    assert "Answer STRICTLY based on the context" in prompt
    assert "Answer not found in the documents." in prompt
    assert "ANSWER:" in prompt


def test_build_rag_prompt_handles_empty_context(sample_query):
    prompt = build_rag_prompt("", sample_query)
    
    assert "CONTEXT:\n" in prompt  # пустой контекст
    assert sample_query in prompt


# ——————— Тесты для продвинутого промпта ———————

def test_build_rag_prompt_advanced_without_metadata(sample_context, sample_query):
    prompt = build_rag_prompt_advanced(sample_context, sample_query)
    
    assert sample_context in prompt
    assert sample_query in prompt
    assert "STEP 1 - ANALYZE" in prompt
    assert "Document metadata:" not in prompt  # метаданных нет


def test_build_rag_prompt_advanced_with_metadata(sample_context, sample_query):
    metadata = {"source": "garden_notes.txt", "page": 42}
    prompt = build_rag_prompt_advanced(sample_context, sample_query, metadata=metadata)
    
    assert sample_context in prompt
    assert sample_query in prompt
    # Проверяем, что метаданные присутствуют как часть строки
    assert "Document metadata: {'source': 'garden_notes.txt', 'page': 42}" in prompt


def test_both_prompts_end_with_answer_marker(sample_context, sample_query):
    basic = build_rag_prompt(sample_context, sample_query)
    advanced = build_rag_prompt_advanced(sample_context, sample_query)
    
    assert basic.strip().endswith("ANSWER:")
    assert advanced.strip().endswith("ANSWER:")