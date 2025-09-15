"""Unit tests for the UnifiedRAGEngine."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from support_deflect_bot.engine.rag_engine import UnifiedRAGEngine
from support_deflect_bot.core.providers import ProviderType, ProviderError, ProviderUnavailableError


class TestUnifiedRAGEngine:
    """Test suite for UnifiedRAGEngine class."""

    @pytest.fixture
    def mock_provider_registry(self):
        """Create a mock provider registry."""
        registry = Mock()

        # Mock LLM provider
        llm_provider = Mock()
        llm_provider.get_config.return_value = Mock(name="test_llm")
        llm_provider.chat.return_value = "This is a test answer."

        # Mock embedding provider
        embedding_provider = Mock()
        embedding_provider.get_config.return_value = Mock(name="test_embedding")
        embedding_provider.embed_texts.return_value = [[0.1, 0.2, 0.3, 0.4]]

        registry.build_fallback_chain.side_effect = lambda provider_type: {
            ProviderType.LLM: [llm_provider],
            ProviderType.EMBEDDING: [embedding_provider]
        }[provider_type]

        return registry

    @pytest.fixture
    def rag_engine(self, mock_provider_registry):
        """Create a RAG engine instance with mocked providers."""
        return UnifiedRAGEngine(provider_registry=mock_provider_registry)

    def test_initialization(self, rag_engine):
        """Test RAG engine initialization."""
        assert rag_engine.provider_registry is not None
        assert rag_engine.metrics["queries_processed"] == 0
        assert rag_engine.metrics["successful_answers"] == 0
        assert rag_engine.metrics["refusals"] == 0
        assert rag_engine.metrics["provider_failures"] == 0
        assert rag_engine.metrics["average_confidence"] == 0.0
        assert rag_engine.metrics["last_query_time"] is None

    @patch('support_deflect_bot.engine.rag_engine.query_by_embedding')
    def test_answer_question_success(self, mock_query, rag_engine):
        """Test successful question answering."""
        # Mock document search results
        mock_hits = [
            {
                "text": "This is a test document with relevant information.",
                "distance": 0.2,
                "meta": {"path": "test.md", "chunk_id": "1"}
            }
        ]
        mock_query.return_value = mock_hits

        result = rag_engine.answer_question("What is this about?")

        assert result["answer"] == "This is a test answer."
        assert result["confidence"] > 0.0
        assert len(result["citations"]) > 0
        assert result["metadata"]["chunks_found"] == 1
        assert rag_engine.metrics["queries_processed"] == 1
        assert rag_engine.metrics["successful_answers"] == 1

    @patch('support_deflect_bot.engine.rag_engine.query_by_embedding')
    def test_answer_question_low_confidence(self, mock_query, rag_engine):
        """Test question answering with low confidence refusal."""
        # Mock low-confidence search results
        mock_hits = [
            {
                "text": "Unrelated content",
                "distance": 0.9,  # High distance = low similarity
                "meta": {"path": "test.md", "chunk_id": "1"}
            }
        ]
        mock_query.return_value = mock_hits

        result = rag_engine.answer_question("What is this about?")

        assert result["answer"] == "I don't have enough information in the docs to answer that."
        assert result["confidence"] < 0.5
        assert result["metadata"]["reason"] == "low_confidence"
        assert rag_engine.metrics["refusals"] == 1

    @patch('support_deflect_bot.engine.rag_engine.query_by_embedding')
    def test_answer_question_no_documents(self, mock_query, rag_engine):
        """Test question answering with no documents found."""
        mock_query.return_value = []

        result = rag_engine.answer_question("What is this about?")

        assert result["answer"] == "I don't have enough information in the docs to answer that."
        assert result["confidence"] == 0.0
        assert result["metadata"]["reason"] == "low_confidence"
        assert rag_engine.metrics["refusals"] == 1

    @patch('support_deflect_bot.engine.rag_engine.query_by_embedding')
    def test_answer_question_provider_failure(self, mock_query, rag_engine):
        """Test question answering with provider failure."""
        # Mock provider failure
        mock_query.side_effect = Exception("Provider failed")

        result = rag_engine.answer_question("What is this about?")

        assert "error" in result["answer"].lower()
        assert result["confidence"] == 0.0
        assert "error" in result["metadata"]
        assert rag_engine.metrics["provider_failures"] == 1

    def test_search_documents_success(self, rag_engine):
        """Test successful document search."""
        with patch('support_deflect_bot.engine.rag_engine.query_by_embedding') as mock_query:
            mock_hits = [
                {
                    "text": "Test document",
                    "distance": 0.3,
                    "meta": {"path": "test.md", "chunk_id": "1"}
                }
            ]
            mock_query.return_value = mock_hits

            results = rag_engine.search_documents("test query", k=5)

            assert len(results) == 1
            assert results[0]["text"] == "Test document"
            mock_query.assert_called_once()

    def test_search_documents_with_domains(self, rag_engine):
        """Test document search with domain filtering."""
        with patch('support_deflect_bot.engine.rag_engine.query_by_embedding') as mock_query:
            mock_query.return_value = []

            rag_engine.search_documents("test query", domains=["example.com"])

            # Verify domain filter was applied
            call_args = mock_query.call_args
            assert call_args[1]["where"] == {"host": {"$in": ["example.com"]}}

    def test_search_documents_embedding_failure(self, rag_engine):
        """Test document search with embedding provider failure."""
        # Mock embedding provider failure
        embedding_provider = rag_engine.provider_registry.build_fallback_chain(ProviderType.EMBEDDING)[0]
        embedding_provider.embed_texts.side_effect = ProviderError("Embedding failed")

        results = rag_engine.search_documents("test query")

        assert results == []

    def test_calculate_confidence(self, rag_engine):
        """Test confidence calculation."""
        hits = [
            {
                "text": "This is about testing and questions",
                "distance": 0.2,
                "meta": {"path": "test.md", "chunk_id": "1"}
            }
        ]

        confidence = rag_engine.calculate_confidence(hits, "What is testing about?")

        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)

    def test_calculate_confidence_no_hits(self, rag_engine):
        """Test confidence calculation with no hits."""
        confidence = rag_engine.calculate_confidence([], "test question")
        assert confidence == 0.0

    def test_get_metrics(self, rag_engine):
        """Test metrics retrieval."""
        with patch.object(rag_engine, '_check_provider_status') as mock_provider_status, \
             patch.object(rag_engine, '_check_database_status') as mock_db_status:

            mock_provider_status.return_value = {"llm_providers": 1, "embedding_providers": 1}
            mock_db_status.return_value = {"connected": True, "document_count": 100}

            metrics = rag_engine.get_metrics()

            assert "queries_processed" in metrics
            assert "successful_answers" in metrics
            assert "provider_status" in metrics
            assert "database_status" in metrics

    def test_validate_providers_success(self, rag_engine):
        """Test provider validation with working providers."""
        status = rag_engine.validate_providers()

        assert "llm_test_llm" in status
        assert "embedding_test_embedding" in status
        assert status["llm_test_llm"] is True
        assert status["embedding_test_embedding"] is True

    def test_validate_providers_failure(self, rag_engine):
        """Test provider validation with failing providers."""
        # Mock provider failures
        llm_provider = rag_engine.provider_registry.build_fallback_chain(ProviderType.LLM)[0]
        embedding_provider = rag_engine.provider_registry.build_fallback_chain(ProviderType.EMBEDDING)[0]

        llm_provider.chat.side_effect = Exception("LLM failed")
        embedding_provider.embed_texts.side_effect = Exception("Embedding failed")

        status = rag_engine.validate_providers()

        assert status["llm_test_llm"] is False
        assert status["embedding_test_embedding"] is False

    def test_generate_answer_fallback(self, rag_engine):
        """Test answer generation with provider fallback."""
        # Mock first provider failure, second success
        llm_providers = rag_engine.provider_registry.build_fallback_chain(ProviderType.LLM)

        # Add a second provider that works
        backup_provider = Mock()
        backup_provider.get_config.return_value = Mock(name="backup_llm")
        backup_provider.chat.return_value = "Backup answer"
        llm_providers.append(backup_provider)

        # Make first provider fail
        llm_providers[0].chat.side_effect = ProviderError("Primary failed")

        answer = rag_engine._generate_answer("Test prompt")

        assert answer == "Backup answer"

    def test_keyword_overlap_calculation(self, rag_engine):
        """Test keyword overlap calculation."""
        question = "How to install the package?"
        text = "Installation guide for the package setup"

        overlap = rag_engine._keyword_overlap(question, text)
        ratio = rag_engine._overlap_ratio(question, text)

        assert overlap > 0
        assert 0.0 <= ratio <= 1.0

    def test_similarity_from_distance(self, rag_engine):
        """Test similarity calculation from distance."""
        # Lower distance should give higher similarity
        sim_low = rag_engine._similarity_from_distance(0.1)
        sim_high = rag_engine._similarity_from_distance(0.9)

        assert sim_low > sim_high
        assert 0.0 <= sim_low <= 1.0
        assert 0.0 <= sim_high <= 1.0

    def test_text_trimming(self, rag_engine):
        """Test text trimming functionality."""
        long_text = "A" * 1000
        trimmed = rag_engine._trim_text(long_text, 100)

        assert len(trimmed) <= 104  # 100 + " … "
        assert trimmed.endswith(" … ")

    def test_citations_formatting(self, rag_engine):
        """Test citation formatting."""
        hits = [
            {
                "text": "First document content",
                "meta": {"path": "doc1.md", "chunk_id": "1"}
            },
            {
                "text": "Second document content",
                "meta": {"path": "doc2.md", "chunk_id": "2"}
            }
        ]

        citations = rag_engine._to_citations(hits, take=2)

        assert len(citations) == 2
        assert citations[0]["rank"] == 1
        assert citations[1]["rank"] == 2
        assert citations[0]["path"] == "doc1.md"
        assert citations[1]["path"] == "doc2.md"

    def test_context_formatting(self, rag_engine):
        """Test context formatting for LLM."""
        hits = [
            {
                "text": "Document content here",
                "meta": {"path": "test.md", "chunk_id": "1"}
            }
        ]

        context = rag_engine._format_context(hits)

        assert "[1]" in context
        assert "(test.md)" in context
        assert "Document content here" in context

    def test_user_prompt_building(self, rag_engine):
        """Test user prompt construction."""
        question = "What is this?"
        context = "[1] (test.md)\nTest content"

        prompt = rag_engine._build_user_prompt(question, context)

        assert "Question: What is this?" in prompt
        assert "Context (numbered citations):" in prompt
        assert "Test content" in prompt
        assert "Instructions:" in prompt

    def test_metrics_update(self, rag_engine):
        """Test metrics updating during operations."""
        initial_queries = rag_engine.metrics["queries_processed"]

        with patch('support_deflect_bot.engine.rag_engine.query_by_embedding') as mock_query:
            mock_query.return_value = [
                {
                    "text": "Test content",
                    "distance": 0.3,
                    "meta": {"path": "test.md", "chunk_id": "1"}
                }
            ]

            rag_engine.answer_question("test question")

            assert rag_engine.metrics["queries_processed"] == initial_queries + 1
            assert rag_engine.metrics["last_query_time"] is not None
            assert isinstance(rag_engine.metrics["last_query_time"], str)

    @patch('support_deflect_bot.engine.rag_engine.get_client')
    @patch('support_deflect_bot.engine.rag_engine.get_collection')
    def test_database_status_check(self, mock_get_collection, mock_get_client, rag_engine):
        """Test database status checking."""
        mock_collection = Mock()
        mock_collection.count.return_value = 150
        mock_get_collection.return_value = mock_collection

        status = rag_engine._check_database_status()

        assert status["connected"] is True
        assert status["document_count"] == 150
        assert "collection" in status

    @patch('support_deflect_bot.engine.rag_engine.get_client')
    def test_database_status_check_failure(self, mock_get_client, rag_engine):
        """Test database status checking with failure."""
        mock_get_client.side_effect = Exception("Database connection failed")

        status = rag_engine._check_database_status()

        assert status["connected"] is False
        assert "error" in status

    def test_stemming_functionality(self, rag_engine):
        """Test simple stemming functionality."""
        assert rag_engine._stem_simple("running") == "run"
        assert rag_engine._stem_simple("worked") == "work"
        assert rag_engine._stem_simple("testing") == "test"
        assert rag_engine._stem_simple("cars") == "car"
        assert rag_engine._stem_simple("bus") == "bus"  # Should not stem "us" ending

    def test_token_extraction(self, rag_engine):
        """Test token extraction and filtering."""
        text = "How to install the package quickly?"
        tokens = rag_engine._tokens(text)

        # Should exclude stop words like "how", "to", "the"
        assert "how" not in tokens
        assert "the" not in tokens
        assert "install" in tokens
        assert "package" in tokens
        assert "quickly" in tokens or "quick" in tokens  # May be stemmed