import pytest
# Assuming the prompt_builder.py file is located in app/services/
from app.services.prompt_builder import build_rag_prompt, build_rag_prompt_advanced


# --- Test Data ---
CONTEXT_SAMPLE = "The first law of thermodynamics states that energy cannot be created or destroyed, only transferred or changed from one form to another. This is also known as the Law of Conservation of Energy."
QUERY_SAMPLE = "What is the First Law of Thermodynamics?"
METADATA_SAMPLE = {"source": "textbook_ch4.pdf", "page": 15}


# --- Tests for Basic Prompt Builder (build_rag_prompt) ---

@pytest.mark.unit
def test_build_rag_prompt_structure():
    """Test the basic function structure: checks for instructions, context, and query inclusion."""
    prompt = build_rag_prompt(
        context=CONTEXT_SAMPLE,
        query=QUERY_SAMPLE,
    )

    # Check for core persona and instruction phrases
    assert "precise scientific assistant" in prompt
    assert "Answer STRICTLY based on the context" in prompt
    assert 'If the answer isn\'t in the context, respond: "Answer not found in the documents."' in prompt

    # Check for data inclusion and prompt delimiters
    assert CONTEXT_SAMPLE in prompt
    assert QUERY_SAMPLE in prompt
    assert prompt.endswith("\nANSWER:")
    assert "CONTEXT:\n" in prompt
    assert "QUESTION: What is the First Law of Thermodynamics?" in prompt


@pytest.mark.unit
def test_build_rag_prompt_empty_context():
    """Test: the function should handle an empty context string correctly."""
    prompt = build_rag_prompt(
        context="",
        query=QUERY_SAMPLE,
    )

    assert QUERY_SAMPLE in prompt
    # Check that the CONTEXT block is empty but present
    assert "CONTEXT:\n\nQUESTION:" in prompt
    assert prompt.endswith("\nANSWER:")


# --- Tests for Advanced Prompt Builder (build_rag_prompt_advanced) ---

@pytest.mark.unit
def test_build_rag_prompt_advanced_structure():
    """Test the advanced function without metadata: checks for Chain-of-Thought steps."""
    prompt = build_rag_prompt_advanced(
        context=CONTEXT_SAMPLE,
        query=QUERY_SAMPLE,
        metadata=None,
    )

    # Check for Chain-of-Thought (CoT) instructions
    assert "expert scientific assistant" in prompt
    assert "STEP 1 - ANALYZE" in prompt
    assert "STEP 2 - SYNTHESIZE" in prompt
    assert "STEP 4 - RESPOND" in prompt

    assert CONTEXT_SAMPLE in prompt
    assert QUERY_SAMPLE in prompt
    assert prompt.endswith("\nANSWER:")
    # Metadata should be absent
    assert "Document metadata" not in prompt


@pytest.mark.unit
def test_build_rag_prompt_advanced_with_metadata():
    """Test the advanced function with metadata: checks for correct metadata formatting."""
    prompt = build_rag_prompt_advanced(
        context=CONTEXT_SAMPLE,
        query=QUERY_SAMPLE,
        metadata=METADATA_SAMPLE,
    )

    # Check for the presence of the formatted metadata string
    expected_metadata_str = "\nDocument metadata: {'source': 'textbook_ch4.pdf', 'page': 15}\n"
    assert expected_metadata_str in prompt
    assert CONTEXT_SAMPLE in prompt
    assert QUERY_SAMPLE in prompt


@pytest.mark.unit
def test_build_rag_prompt_advanced_empty_context_and_no_metadata():
    """Test: advanced function with empty context and no metadata."""
    prompt = build_rag_prompt_advanced(
        context="",
        query="Just a query?",
        metadata=None,
    )

    assert "Just a query?" in prompt
    # Ensure that metadata and context are correctly absent in the body
    assert "CONTEXT:\n\nQUESTION: Just a query?" in prompt
    assert "Document metadata" not in prompt
    assert "STEP 1 - ANALYZE" in prompt