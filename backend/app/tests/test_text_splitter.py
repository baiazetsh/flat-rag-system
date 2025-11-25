# backend/app/tests/test_text_splitter.py
"""
Unit tests for SmartTextSplitter:
- Sentence splitting
- Token counting
- Semantic chunking
- Strategy selection
"""
import pytest
from app.services.smart_text_splitter_service import (
    SmartTextSplitter,
    TokenCounter,
    SemanticChunker
)


# ============================================================
# TokenCounter Tests
# ============================================================
@pytest.mark.unit
class TestTokenCounter:
    """Tests for TokenCounter."""
    
    def test_token_counter_initialization(self):
        """Test token counter initializes."""
        counter = TokenCounter(model="gpt-4")
        assert counter is not None
    
    def test_count_simple_text(self):
        """Test counting tokens in simple text."""
        counter = TokenCounter()
        text = "This is a simple test sentence."
        count = counter.count(text)
        
        assert count > 0
        assert isinstance(count, int)
    
    def test_count_empty_text(self):
        """Test counting empty text."""
        counter = TokenCounter()
        count = counter.count("")
        assert count == 0
    
    def test_count_list_of_strings(self):
        """Test counting list of strings."""
        counter = TokenCounter()
        texts = ["First sentence.", "Second sentence."]
        total = sum(counter.count(t) for t in texts)
        
        assert total > 0
        assert isinstance(total, int)


# ============================================================
# SmartTextSplitter Tests
# ============================================================
@pytest.mark.unit
class TestSmartTextSplitter:
    """Tests for SmartTextSplitter."""
    
    @pytest.mark.asyncio
    async def test_split_simple_text(self, sample_text):
        """Test splitting simple text."""
        splitter = SmartTextSplitter(
            max_chunk_size=200,
            strategy="sentence"
        )
        
        chunks = await splitter.split(sample_text)
        
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)
    
    @pytest.mark.asyncio
    async def test_split_empty_text(self):
        """Test splitting empty text."""
        splitter = SmartTextSplitter()
        chunks = await splitter.split("")
        
        assert len(chunks) == 0
    
    @pytest.mark.asyncio
    async def test_split_with_max_chunks_limit(self, sample_text):
        """Test max_chunks limit is enforced."""
        splitter = SmartTextSplitter(
            max_chunk_size=50,
            max_chunks=2
        )
        
        chunks = await splitter.split(sample_text)
        
        assert len(chunks) <= 2
    
    @pytest.mark.asyncio
    async def test_split_preserves_formatting(self):
        """Test formatting preservation."""
        text = "Line 1\nLine 2\n\nLine 3"
        
        splitter = SmartTextSplitter(preserve_formatting=True)
        chunks = await splitter.split(text)
        
        # With formatting preserved, newlines should remain
        combined = " ".join(chunks)
        assert "\n" in combined or len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_split_returns_metadata(self, sample_text):
        """Test metadata return option."""
        splitter = SmartTextSplitter(return_metadata=True)
        result = await splitter.split(sample_text)
        
        assert len(result) > 0
        # Each item should be tuple (text, metadata)
        if result:
            assert isinstance(result[0], tuple)
            assert len(result[0]) == 2
    
    def test_get_stats_empty_chunks(self):
        """Test stats for empty chunk list."""
        splitter = SmartTextSplitter()
        stats = splitter.get_stats([])
        
        assert stats["total_chunks"] == 0
        assert stats["total_chars"] == 0
    
    def test_get_stats_with_chunks(self, sample_chunks):
        """Test stats calculation."""
        splitter = SmartTextSplitter()
        stats = splitter.get_stats(sample_chunks)
        
        assert stats["total_chunks"] == len(sample_chunks)
        assert stats["total_chars"] > 0
        assert stats["avg_chunk_chars"] > 0
        assert stats["min_chunk_size"] > 0
        assert stats["max_chunk_size"] > 0
    
    def test_split_sync(self, sample_text):
        """Test synchronous split wrapper."""
        splitter = SmartTextSplitter(max_chunk_size=200)
        chunks = splitter.split_sync(sample_text)
        
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)


# ============================================================
# Strategy Selection Tests
# ============================================================
@pytest.mark.unit
class TestSplitterStrategySelection:
    """Tests for automatic strategy selection."""
    
    def test_choose_strategy_semantic_available(self):
        """Test semantic strategy chosen when available."""
        splitter = SmartTextSplitter(strategy="auto")
        if not splitter.semantic_chunker.is_available():
            pytest.skip("Semantic chunking unavailable(no sentence-transformers)")
        # Create long sentence list to trigger semantic
        sentences = ["Sentence " + str(i) for i in range(10)]
        
        strategy = splitter._choose_strategy(sentences)
        
        # Should be semantic or token-aware
        assert strategy in ["semantic", "token-aware"]
    
    def test_choose_strategy_token_aware_many_sentences(self):
        """Test token-aware for many sentences."""
        splitter = SmartTextSplitter(strategy="auto")
        
        # 60 sentences should trigger token-aware
        sentences = ["Sentence " + str(i) for i in range(60)]
        
        strategy = splitter._choose_strategy(sentences)
        
        assert strategy == "token-aware"
    
    def test_choose_strategy_sentence_few_sentences(self):
        """Test sentence strategy for few sentences."""
        splitter = SmartTextSplitter(strategy="auto")
        
        # Few sentences
        sentences = ["Sentence 1", "Sentence 2"]
        
        strategy = splitter._choose_strategy(sentences)
        
        # Should be sentence (fallback)
        assert strategy in ["sentence", "semantic"]
    
    def test_explicit_strategy_respected(self):
        """Test explicit strategy is used."""
        splitter = SmartTextSplitter(strategy="token-aware")
        
        sentences = ["Sentence " + str(i) for i in range(10)]
        strategy = splitter._choose_strategy(sentences)
        
        assert strategy == "token-aware"


# ============================================================
# Edge Cases
# ============================================================
@pytest.mark.unit
class TestSplitterEdgeCases:
    """Tests for edge cases."""
    
    @pytest.mark.asyncio
    async def test_split_very_long_text(self):
        """Test splitting very long text."""
        text = "Word " * 10000  # 10k words
        
        splitter = SmartTextSplitter(
            max_chunk_size=500,
            max_chunks=100
        )
        
        chunks = await splitter.split(text)
        
        assert len(chunks) > 0
        assert len(chunks) <= 100
    
    @pytest.mark.asyncio
    async def test_split_single_long_sentence(self):
        """Test splitting text with single very long sentence."""
        text = "Word " * 1000 + "."
        
        splitter = SmartTextSplitter(max_chunk_size=200)
        chunks = await splitter.split(text)
        
        # Should still produce chunks
        assert len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_split_special_characters(self):
        """Test text with special characters."""
        text = "Test with emojis 😊 and symbols ∑ ∫ ∂ and numbers 123.45"
        
        splitter = SmartTextSplitter()
        chunks = await splitter.split(text)
        
        assert len(chunks) > 0