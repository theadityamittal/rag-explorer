"""
Unit tests for UnifiedRAGEngine service.

Tests the core RAG (Retrieval Augmented Generation) engine that coordinates
document retrieval, embedding generation, and response generation.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from tests.base import BaseEngineTest
from support_deflect_bot.engine import UnifiedRAGEngine


class TestUnifiedRAGEngine(BaseEngineTest):
    """Test the UnifiedRAGEngine service."""
    
    @pytest.fixture
    def rag_engine(self):
        """Create UnifiedRAGEngine instance for testing."""
        mock_registry = {}
        return UnifiedRAGEngine(provider_registry=mock_registry)
    
    @pytest.mark.unit
    @pytest.mark.engine
    def test_init_creates_correct_attributes(self, rag_engine):
        """Test RAG engine initialization creates correct attributes."""
        # Assert
        assert hasattr(rag_engine, 'provider_registry')
        assert hasattr(rag_engine, 'metrics')
        assert hasattr(rag_engine, 'system_prompt')
        assert hasattr(rag_engine, '_stop_words')
        
        # Check metrics structure
        expected_metrics = ["queries_processed", "successful_answers", "refusals", 
                          "provider_failures", "average_confidence", "last_query_time"]
        for metric in expected_metrics:
            assert metric in rag_engine.metrics
            
    @pytest.mark.unit
    @pytest.mark.engine
    def test_system_prompt_contains_key_instructions(self, rag_engine):
        """Test system prompt contains critical instructions."""
        prompt = rag_engine.system_prompt
        
        # Assert key instructions are present
        assert "support deflection assistant" in prompt.lower()
        assert "use ONLY the provided Context" in prompt
        assert "refuse ONLY if" in prompt
        assert "I don't have enough information" in prompt
        
    @pytest.mark.unit
    @pytest.mark.engine
    @patch('support_deflect_bot.engine.rag_engine.get_default_registry')
    def test_answer_question_with_high_confidence(self, mock_registry, rag_engine):
        """Test answering question with high confidence result."""
        # Arrange
        question = "How do I install the Support Deflect Bot?"
        mock_hits = [
            {
                "content": "To install the Support Deflect Bot, run pip install -e .",
                "source": "getting_started.md",
                "distance": 0.1,
                "section": "Installation"
            },
            {
                "content": "Clone the repository first, then install dependencies",
                "source": "setup.md", 
                "distance": 0.2,
                "section": "Setup"
            }
        ]
        
        with patch.object(rag_engine, 'search_documents', return_value=mock_hits):
            with patch.object(rag_engine, 'calculate_confidence', return_value=0.92):
                with patch.object(rag_engine, '_generate_answer', return_value="Install by running pip install -e ."):
                    # Act
                    result = rag_engine.answer_question(question)
                    
                    # Assert
                    assert result["confidence"] > 0.8
                    assert "install" in result["answer"].lower()
                    assert len(result["citations"]) > 0
                    assert result["citations"][0]["source"] == "getting_started.md"
                    assert rag_engine.metrics["queries_processed"] > 0
                    assert rag_engine.metrics["last_query_time"] is not None
                    
    @pytest.mark.unit 
    @pytest.mark.engine
    @patch('support_deflect_bot.engine.rag_engine.get_default_registry')
    def test_answer_question_with_low_confidence(self, mock_registry, rag_engine):
        """Test answering question with low confidence - should refuse."""
        # Arrange
        question = "What is the meaning of life?"
        mock_hits = [
            {
                "content": "This is not related to the bot at all",
                "source": "unrelated.md",
                "distance": 0.9,
                "section": "Other"
            }
        ]
        
        with patch.object(rag_engine, 'search_documents', return_value=mock_hits):
            with patch.object(rag_engine, 'calculate_confidence', return_value=0.15):
                # Act
                result = rag_engine.answer_question(question)
                
                # Assert
                assert result["confidence"] < 0.25
                assert "I don't have enough information" in result["answer"]
                assert len(result["citations"]) == 0
                assert rag_engine.metrics["refusals"] > 0
                
    @pytest.mark.unit
    @pytest.mark.engine
    @patch('support_deflect_bot.engine.rag_engine.get_default_registry')
    def test_search_documents_with_domain_filter(self, mock_registry, rag_engine):
        """Test document search with domain filtering."""
        # This would test the search_documents method
        # For now, we'll test it as a mock since it depends on ChromaDB
        with patch('chromadb.Client') as mock_chroma:
            mock_collection = Mock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
            mock_collection.query.return_value = {
                'documents': [["Test document content"]],
                'metadatas': [[{"source": "test.md", "section": "Test"}]],
                'distances': [[0.1]]
            }
            
            # Act
            result = rag_engine.search_documents("test query", domains=["test.md"])
            
            # Assert - This will be more meaningful once search_documents is fully implemented
            # For now, just ensure it returns expected structure
            assert isinstance(result, list)
            
    @pytest.mark.unit
    @pytest.mark.engine
    def test_calculate_confidence_with_relevant_content(self, rag_engine):
        """Test confidence calculation with highly relevant content."""
        # Arrange
        hits = [
            {
                "content": "Install the bot using pip install command",
                "distance": 0.1
            },
            {
                "content": "To install, run pip install -e . in your terminal",
                "distance": 0.15
            }
        ]
        question = "How do I install the bot?"
        
        # Act
        confidence = rag_engine.calculate_confidence(hits, question)
        
        # Assert
        assert confidence > 0.5  # Should be high due to keyword overlap and low distance
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_calculate_confidence_with_irrelevant_content(self, rag_engine):
        """Test confidence calculation with irrelevant content."""
        # Arrange 
        hits = [
            {
                "content": "The weather is nice today",
                "distance": 0.8
            },
            {
                "content": "Cats are fluffy animals",
                "distance": 0.9
            }
        ]
        question = "How do I install the bot?"
        
        # Act
        confidence = rag_engine.calculate_confidence(hits, question)
        
        # Assert
        assert confidence < 0.3  # Should be low due to no keyword overlap and high distance
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_get_metrics_returns_correct_structure(self, rag_engine):
        """Test metrics retrieval returns correct structure."""
        # Act
        metrics = rag_engine.get_metrics()
        
        # Assert
        expected_keys = ["queries_processed", "successful_answers", "refusals", 
                        "provider_failures", "average_confidence", "last_query_time"]
        for key in expected_keys:
            assert key in metrics
        assert isinstance(metrics["queries_processed"], int)
        assert isinstance(metrics["average_confidence"], float)
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_validate_providers_checks_availability(self, rag_engine):
        """Test provider validation checks availability."""
        # Arrange
        rag_engine.provider_registry = {
            "test_provider": Mock(is_available=Mock(return_value=True)),
            "unavailable_provider": Mock(is_available=Mock(return_value=False))
        }
        
        # Act
        result = rag_engine.validate_providers()
        
        # Assert
        assert "providers" in result
        # The implementation details depend on the actual validate_providers method
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_build_user_prompt_formats_correctly(self, rag_engine):
        """Test user prompt building formats question and context correctly."""
        # Arrange
        question = "How do I configure the bot?"
        context = "Context: Configuration is done via environment variables."
        
        # Act
        prompt = rag_engine._build_user_prompt(question, context)
        
        # Assert
        assert question in prompt
        assert context in prompt
        assert "Context:" in prompt
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_format_context_limits_chunks_correctly(self, rag_engine):
        """Test context formatting limits chunks correctly."""
        # Arrange
        hits = [
            {"content": f"Chunk {i} content", "source": f"doc{i}.md"} 
            for i in range(10)
        ]
        
        # Act
        context = rag_engine._format_context(hits)
        
        # Assert
        assert isinstance(context, str)
        assert len(context) > 0
        # Should contain source information
        assert "doc" in context
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_trim_text_respects_limit(self, rag_engine):
        """Test text trimming respects character limit."""
        # Arrange
        long_text = "a" * 1000
        limit = 500
        
        # Act
        trimmed = rag_engine._trim_text(long_text, limit)
        
        # Assert
        assert len(trimmed) <= limit
        assert "..." in trimmed  # Should indicate truncation
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_to_citations_returns_correct_format(self, rag_engine):
        """Test citation generation returns correct format."""
        # Arrange
        hits = [
            {
                "source": "test1.md",
                "content": "Test content 1",
                "section": "Section 1",
                "distance": 0.1
            },
            {
                "source": "test2.md", 
                "content": "Test content 2",
                "section": "Section 2",
                "distance": 0.2
            }
        ]
        
        # Act
        citations = rag_engine._to_citations(hits, take=2)
        
        # Assert
        assert len(citations) <= 2
        assert all("source" in citation for citation in citations)
        assert citations[0]["source"] == "test1.md"
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_similarity_from_distance_converts_correctly(self, rag_engine):
        """Test distance to similarity conversion."""
        # Act
        similarity_high = rag_engine._similarity_from_distance(0.1)  # Low distance = high similarity
        similarity_low = rag_engine._similarity_from_distance(0.9)   # High distance = low similarity
        
        # Assert
        assert similarity_high > similarity_low
        assert 0.0 <= similarity_high <= 1.0
        assert 0.0 <= similarity_low <= 1.0
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_tokens_splits_text_correctly(self, rag_engine):
        """Test text tokenization splits correctly."""
        # Arrange
        text = "How do I install the bot?"
        
        # Act
        tokens = rag_engine._tokens(text)
        
        # Assert
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "install" in tokens
        assert "bot" in tokens
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_keyword_overlap_calculates_correctly(self, rag_engine):
        """Test keyword overlap calculation."""
        # Arrange
        question = "How do I install the bot?"
        text_relevant = "To install the bot, run pip install"
        text_irrelevant = "The weather is nice today"
        
        # Act
        overlap_relevant = rag_engine._keyword_overlap(question, text_relevant)
        overlap_irrelevant = rag_engine._keyword_overlap(question, text_irrelevant)
        
        # Assert
        assert overlap_relevant > overlap_irrelevant
        assert overlap_relevant >= 2  # Should match "install" and "bot"
        assert overlap_irrelevant == 0  # No matching keywords
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_overlap_ratio_calculates_correctly(self, rag_engine):
        """Test overlap ratio calculation."""
        # Arrange
        question = "install bot"
        text_full_match = "install bot completely"
        text_partial_match = "install the system"
        
        # Act
        ratio_full = rag_engine._overlap_ratio(question, text_full_match)
        ratio_partial = rag_engine._overlap_ratio(question, text_partial_match)
        
        # Assert
        assert ratio_full > ratio_partial
        assert 0.0 <= ratio_full <= 1.0
        assert 0.0 <= ratio_partial <= 1.0
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_update_average_confidence_updates_correctly(self, rag_engine):
        """Test average confidence tracking updates correctly."""
        # Arrange
        initial_avg = rag_engine.metrics["average_confidence"]
        
        # Act
        rag_engine._update_average_confidence(0.8)
        first_update = rag_engine.metrics["average_confidence"]
        
        rag_engine._update_average_confidence(0.6)
        second_update = rag_engine.metrics["average_confidence"]
        
        # Assert
        assert first_update != initial_avg
        assert second_update != first_update
        assert 0.0 <= second_update <= 1.0


class TestUnifiedRAGEngineIntegration(BaseEngineTest):
    """Integration tests for RAG engine with mocked external dependencies."""
    
    @pytest.mark.unit
    @pytest.mark.engine
    @patch('chromadb.Client')
    @patch('support_deflect_bot.engine.rag_engine.get_default_registry')
    def test_full_rag_pipeline_with_mocks(self, mock_registry, mock_chroma):
        """Test complete RAG pipeline with all external dependencies mocked."""
        # Arrange
        mock_registry.return_value = {
            "test_provider": Mock(
                is_available=Mock(return_value=True),
                generate_response=Mock(return_value="Test LLM response")
            )
        }
        
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_collection.query.return_value = {
            'documents': [["Install using pip install -e ."]],
            'metadatas': [[{"source": "install.md", "section": "Installation"}]], 
            'distances': [[0.1]]
        }
        
        # Act
        rag_engine = UnifiedRAGEngine()
        result = rag_engine.answer_question("How do I install?")
        
        # Assert
        assert "answer" in result
        assert "confidence" in result
        assert "citations" in result
        assert isinstance(result["confidence"], float)
        
    @pytest.mark.unit
    @pytest.mark.engine 
    def test_error_handling_with_provider_failure(self, rag_engine):
        """Test RAG engine handles provider failures gracefully."""
        # Arrange
        failing_provider = Mock()
        failing_provider.generate_response.side_effect = Exception("Provider failed")
        rag_engine.provider_registry = {"failing_provider": failing_provider}
        
        with patch.object(rag_engine, 'search_documents', return_value=[]):
            # Act & Assert
            # This should handle the error gracefully, depending on implementation
            result = rag_engine.answer_question("test question")
            assert "answer" in result  # Should still return a result structure