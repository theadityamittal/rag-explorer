import pytest
from unittest.mock import Mock, patch
from src.core.rag import (
    _trim, _stem_simple, _tokens, _keyword_overlap, 
    _overlap_ratio, _similarity_from_distance, _confidence,
    _to_citations, _format_context, answer_question
)


class TestRagUtilities:
    """Test utility functions in the RAG module."""

    def test_trim(self):
        """Test text trimming function."""
        assert _trim("short", 10) == "short"
        assert _trim("this is a long text", 10) == "this is a … "
        assert _trim("exactly ten", 10) == "exactly ten"
        assert _trim("", 10) == ""

    def test_stem_simple(self):
        """Test simple stemming function."""
        assert _stem_simple("running") == "run"
        assert _stem_simple("played") == "play"
        assert _stem_simple("cats") == "cat"
        assert _stem_simple("bus") == "bus"  # doesn't end with 's' removal pattern
        assert _stem_simple("class") == "class"  # ends with 'ss'
        assert _stem_simple("go") == "go"  # too short

    def test_tokens(self):
        """Test tokenization function."""
        text = "This is a TEST with numbers123 and symbols!"
        tokens = _tokens(text)
        
        # Should be lowercase, alphanumeric, length > 2, no stop words
        assert "test" in tokens
        assert "numbers123" in tokens
        assert "symbols" in tokens
        assert "this" not in tokens  # stop word
        assert "is" not in tokens   # stop word and length <= 2

    def test_keyword_overlap(self):
        """Test keyword overlap calculation."""
        question = "How to install Python packages?"
        text = "Install Python packages using pip install command"
        
        overlap = _keyword_overlap(question, text)
        assert overlap > 0  # Should have some overlap
        
        # Test with no overlap
        no_overlap = _keyword_overlap("cats dogs", "birds fish")
        assert no_overlap == 0

    def test_overlap_ratio(self):
        """Test overlap ratio calculation."""
        question = "install package"
        text = "install package using pip"
        
        ratio = _overlap_ratio(question, text)
        assert 0 <= ratio <= 1
        assert ratio > 0  # Should have overlap
        
        # Test with empty question
        empty_ratio = _overlap_ratio("", "some text")
        assert empty_ratio == 0.0

    def test_similarity_from_distance(self):
        """Test distance to similarity conversion."""
        assert _similarity_from_distance(0) == 1.0
        assert _similarity_from_distance(1) == 0.5
        assert 0 < _similarity_from_distance(10) < 1
        assert _similarity_from_distance("invalid") == 0.5


class TestRagConfidence:
    """Test confidence calculation."""

    def test_confidence_empty_hits(self):
        """Test confidence with no hits."""
        assert _confidence([], "test question") == 0.0

    def test_confidence_calculation(self):
        """Test confidence calculation with hits."""
        hits = [
            {"text": "install package using pip", "distance": 0.3},
            {"text": "other content", "distance": 0.8}
        ]
        question = "how to install package"
        
        confidence = _confidence(hits, question)
        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)

    def test_confidence_no_distance(self):
        """Test confidence when hits have no distance."""
        hits = [{"text": "some text", "distance": None}]
        confidence = _confidence(hits, "test")
        assert 0 <= confidence <= 1


class TestRagCitations:
    """Test citation generation."""

    def test_to_citations(self):
        """Test citation generation from hits."""
        hits = [
            {"text": "Long text content for testing citations", 
             "meta": {"path": "test.md", "chunk_id": 0}},
            {"text": "Another text chunk", 
             "meta": {"path": "test2.md", "chunk_id": 1}}
        ]
        
        citations = _to_citations(hits, take=2)
        
        assert len(citations) == 2
        assert citations[0]["rank"] == 1
        assert citations[0]["path"] == "test.md"
        assert citations[0]["chunk_id"] == 0
        assert len(citations[0]["preview"]) <= 200

    def test_to_citations_empty(self):
        """Test citation generation with empty hits."""
        citations = _to_citations([], take=3)
        assert citations == []

    def test_format_context(self):
        """Test context formatting for LLM."""
        hits = [
            {"text": "First chunk content", 
             "meta": {"path": "doc1.md", "chunk_id": 0}},
            {"text": "Second chunk content", 
             "meta": {"path": "doc2.md", "chunk_id": 1}}
        ]
        
        context = _format_context(hits)
        
        assert "[1] (doc1.md)" in context
        assert "[2] (doc2.md)" in context
        assert "First chunk content" in context
        assert "Second chunk content" in context


class TestAnswerQuestion:
    """Test the main answer_question function."""

    @patch('src.core.rag.retrieve')
    @patch('src.core.rag.llm_chat')
    def test_answer_question_high_confidence(self, mock_llm, mock_retrieve):
        """Test answering with high confidence."""
        # Mock retrieve to return relevant hits
        mock_retrieve.return_value = [
            {"text": "Install packages using pip install", 
             "meta": {"path": "docs.md", "chunk_id": 0}, 
             "distance": 0.2}
        ]
        
        # Mock LLM response
        mock_llm.return_value = "Use pip install to install packages."
        
        with patch('src.core.rag.MIN_CONF', 0.1):  # Set low threshold
            result = answer_question("How to install packages?")
        
        assert "answer" in result
        assert "citations" in result
        assert "confidence" in result
        assert result["answer"] == "Use pip install to install packages."
        assert len(result["citations"]) > 0

    @patch('src.core.rag.retrieve')
    def test_answer_question_low_confidence(self, mock_retrieve):
        """Test refusal with low confidence."""
        # Mock retrieve to return irrelevant hits
        mock_retrieve.return_value = [
            {"text": "Completely unrelated content", 
             "meta": {"path": "docs.md", "chunk_id": 0}, 
             "distance": 0.9}
        ]
        
        with patch('src.core.rag.MIN_CONF', 0.8):  # Set high threshold
            result = answer_question("How to install packages?")
        
        assert result["answer"] == "I don't have enough information in the docs to answer that."

    @patch('src.core.rag.retrieve')
    @patch('src.core.rag.llm_chat')
    def test_answer_question_empty_llm_response(self, mock_llm, mock_retrieve):
        """Test handling of empty LLM response."""
        mock_retrieve.return_value = [
            {"text": "Some relevant content", 
             "meta": {"path": "docs.md", "chunk_id": 0}, 
             "distance": 0.3}
        ]
        
        # Mock empty LLM response
        mock_llm.return_value = ""
        
        with patch('src.core.rag.MIN_CONF', 0.1):  # Set low threshold
            result = answer_question("Test question?")
        
        # Should fallback to extractive answer
        assert "answer" in result
        assert len(result["answer"]) > 0
        assert result["answer"] != "I don't have enough information in the docs to answer that."

    @patch('src.core.rag.retrieve')
    def test_answer_question_with_domains(self, mock_retrieve):
        """Test answering with domain filtering."""
        mock_retrieve.return_value = [
            {"text": "Domain-specific content", 
             "meta": {"path": "docs.md", "chunk_id": 0}, 
             "distance": 0.3}
        ]
        
        with patch('src.core.rag.MIN_CONF', 0.8):  # High threshold for refusal
            result = answer_question("Test question?", domains=["example.com"])
        
        # Should call retrieve with domains parameter
        mock_retrieve.assert_called_once_with("Test question?", k=5, domains=["example.com"])