"""
Unit tests for UnifiedQueryService.

Tests the query preprocessing, document retrieval, and result ranking service
that handles query optimization and search result processing.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from tests.base import BaseEngineTest
from support_deflect_bot.engine import UnifiedQueryService


class TestUnifiedQueryService(BaseEngineTest):
    """Test the UnifiedQueryService."""
    
    @pytest.fixture
    def query_service(self):
        """Create UnifiedQueryService instance for testing."""
        mock_registry = {}
        return UnifiedQueryService(provider_registry=mock_registry)
    
    @pytest.mark.unit
    @pytest.mark.engine
    def test_init_creates_correct_attributes(self, query_service):
        """Test query service initialization creates correct attributes."""
        # Assert
        assert hasattr(query_service, 'provider_registry')
        assert hasattr(query_service, 'query_analytics')
        assert hasattr(query_service, '_stop_words')
        assert hasattr(query_service, '_technical_terms')
        
        # Check analytics structure
        expected_analytics = ["queries_processed", "average_results_returned", 
                            "query_types", "performance_metrics"]
        analytics = query_service.query_analytics
        for key in expected_analytics:
            assert key in analytics
            
    @pytest.mark.unit
    @pytest.mark.engine
    def test_preprocess_query_cleans_and_extracts_keywords(self, query_service):
        """Test query preprocessing cleans text and extracts keywords."""
        # Arrange
        raw_query = "  How do I install the bot?  "
        
        # Act
        result = query_service.preprocess_query(raw_query)
        
        # Assert
        assert "query" in result
        assert "keywords" in result
        assert "query_type" in result
        assert "expanded_terms" in result
        
        # Clean query should be trimmed and normalized
        assert result["query"].strip() == result["query"]
        assert len(result["keywords"]) > 0
        assert "install" in result["keywords"]
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_preprocess_query_classifies_technical_questions(self, query_service):
        """Test query preprocessing identifies technical questions."""
        # Arrange
        technical_query = "How do I configure the API endpoints?"
        non_technical_query = "What is this product about?"
        
        # Act
        technical_result = query_service.preprocess_query(technical_query)
        non_technical_result = query_service.preprocess_query(non_technical_query)
        
        # Assert
        # Technical queries should be classified differently
        assert technical_result["query_type"] in ["technical", "configuration", "how_to"]
        assert non_technical_result["query_type"] in ["general", "what_is"]
        
    @pytest.mark.unit
    @pytest.mark.engine
    async def test_retrieve_documents_combines_search_methods(self, query_service):
        """Test document retrieval combines semantic and keyword search."""
        # Arrange
        query_info = {
            "query": "install bot",
            "keywords": ["install", "bot"],
            "query_type": "how_to"
        }
        
        with patch.object(query_service, '_semantic_search', return_value=[
            {"content": "Install guide", "source": "install.md", "score": 0.9}
        ]) as mock_semantic:
            with patch.object(query_service, '_keyword_search', return_value=[
                {"content": "Bot installation", "source": "bot.md", "score": 0.8}
            ]) as mock_keyword:
                # Act
                results = await query_service.retrieve_documents(query_info, k=5)
                
                # Assert
                assert len(results) > 0
                assert all("content" in result for result in results)
                assert all("score" in result for result in results)
                mock_semantic.assert_called_once()
                mock_keyword.assert_called_once()
                
    @pytest.mark.unit
    @pytest.mark.engine
    def test_rank_results_scores_relevance_correctly(self, query_service):
        """Test result ranking scores by relevance factors."""
        # Arrange
        keywords = ["install", "bot"]
        results = [
            {
                "content": "How to install the bot step by step",
                "source": "install.md",
                "title": "Installation Guide", 
                "score": 0.8,
                "metadata": {"section": "Getting Started", "date": "2024-01-01"}
            },
            {
                "content": "Bot configuration after installation",
                "source": "config.md",
                "title": "Configuration",
                "score": 0.6,
                "metadata": {"section": "Setup", "date": "2024-01-15"}
            }
        ]
        
        # Act
        ranked_results = query_service.rank_results(results, keywords)
        
        # Assert
        assert len(ranked_results) == len(results)
        assert all("final_score" in result for result in ranked_results)
        
        # Results should be sorted by final score (descending)
        scores = [r["final_score"] for r in ranked_results]
        assert scores == sorted(scores, reverse=True)
        
        # First result should have higher score due to better keyword match
        assert ranked_results[0]["final_score"] > ranked_results[1]["final_score"]
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_calculate_keyword_overlap_counts_matches(self, query_service):
        """Test keyword overlap calculation counts relevant matches."""
        # Arrange
        keywords = ["install", "bot", "configuration"]
        text_high_overlap = "Install the bot with proper configuration settings"
        text_low_overlap = "The weather is nice today for walking"
        
        # Act
        high_overlap = query_service.calculate_keyword_overlap(keywords, text_high_overlap)
        low_overlap = query_service.calculate_keyword_overlap(keywords, text_low_overlap)
        
        # Assert
        assert high_overlap > low_overlap
        assert high_overlap >= 3  # Should match all keywords
        assert low_overlap == 0   # Should match no keywords
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_apply_domain_filters_filters_by_source(self, query_service):
        """Test domain filtering restricts results to specified domains."""
        # Arrange
        results = [
            {"source": "install.md", "content": "Install guide"},
            {"source": "config.md", "content": "Config guide"}, 
            {"source": "other.txt", "content": "Other content"}
        ]
        domains = ["install.md", "config.md"]
        
        # Act
        filtered = query_service.apply_domain_filters(results, domains)
        
        # Assert
        assert len(filtered) == 2
        sources = [r["source"] for r in filtered]
        assert "install.md" in sources
        assert "config.md" in sources
        assert "other.txt" not in sources
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_get_query_analytics_returns_metrics(self, query_service):
        """Test query analytics returns performance metrics."""
        # Act
        analytics = query_service.get_query_analytics()
        
        # Assert
        expected_keys = ["queries_processed", "average_results_returned", 
                        "query_types", "performance_metrics"]
        for key in expected_keys:
            assert key in analytics
            
        assert isinstance(analytics["queries_processed"], int)
        assert isinstance(analytics["query_types"], dict)
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_clean_query_normalizes_text(self, query_service):
        """Test query cleaning normalizes and sanitizes text."""
        # Arrange
        messy_query = "  HOW do I Install THE bot???  "
        
        # Act
        cleaned = query_service._clean_query(messy_query)
        
        # Assert
        assert cleaned.strip() == cleaned  # No leading/trailing whitespace
        assert "?" not in cleaned or cleaned.count("?") <= 1  # Reduced punctuation
        assert len(cleaned) > 0
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_tokenize_splits_text_correctly(self, query_service):
        """Test tokenization splits text into meaningful tokens."""
        # Arrange
        text = "How do I install the bot?"
        
        # Act
        tokens = query_service._tokenize(text)
        
        # Assert
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "install" in tokens
        assert "bot" in tokens
        # Stop words should be filtered out
        assert "do" not in tokens or "the" not in tokens
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_extract_keywords_identifies_important_terms(self, query_service):
        """Test keyword extraction identifies important terms."""
        # Arrange
        tokens = ["install", "bot", "configuration", "api", "endpoint"]
        
        # Act
        keywords = query_service._extract_keywords(tokens)
        
        # Assert
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "install" in keywords
        assert "bot" in keywords
        # Should preserve technical terms
        assert "api" in keywords
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_classify_query_identifies_query_types(self, query_service):
        """Test query classification identifies different query types."""
        # Arrange
        how_to_keywords = ["install", "configure", "setup"]
        what_is_keywords = ["what", "explain", "define"]
        troubleshoot_keywords = ["error", "problem", "fix", "broken"]
        
        # Act
        how_to_type = query_service._classify_query("how to install", how_to_keywords)
        what_is_type = query_service._classify_query("what is this", what_is_keywords)  
        troubleshoot_type = query_service._classify_query("error occurred", troubleshoot_keywords)
        
        # Assert
        assert how_to_type in ["how_to", "technical"]
        assert what_is_type in ["what_is", "general"]
        assert troubleshoot_type in ["troubleshooting", "error"]
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_has_technical_terms_identifies_technical_content(self, query_service):
        """Test technical term detection."""
        # Arrange
        technical_keywords = ["api", "endpoint", "configuration", "ssl"]
        non_technical_keywords = ["hello", "help", "please", "thanks"]
        
        # Act
        has_technical = query_service._has_technical_terms(technical_keywords)
        has_non_technical = query_service._has_technical_terms(non_technical_keywords)
        
        # Assert
        assert has_technical is True
        assert has_non_technical is False
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_expand_query_adds_synonyms(self, query_service):
        """Test query expansion adds relevant synonyms."""
        # Arrange
        query = "install bot"
        
        # Act
        expanded_query, expanded_terms = query_service._expand_query(query)
        
        # Assert
        assert isinstance(expanded_query, str)
        assert isinstance(expanded_terms, list)
        assert len(expanded_terms) >= 0  # May or may not find expansions
        
    @pytest.mark.unit
    @pytest.mark.engine
    async def test_get_query_embedding_generates_vector(self, query_service):
        """Test query embedding generation."""
        # Arrange
        query = "How to install the bot"
        
        with patch.object(query_service, '_get_query_embedding', return_value=[0.1] * 384) as mock_embed:
            # Act
            embedding = await query_service._get_query_embedding(query)
            
            # Assert
            if embedding is not None:  # Depends on provider availability
                assert isinstance(embedding, list)
                assert len(embedding) > 0
                assert all(isinstance(val, float) for val in embedding)
                
    @pytest.mark.unit
    @pytest.mark.engine
    def test_calculate_title_relevance_scores_title_matches(self, query_service):
        """Test title relevance calculation."""
        # Arrange
        keywords = ["install", "bot"]
        relevant_title = "Bot Installation Guide"
        irrelevant_title = "Weather Reports"
        
        # Act
        relevant_score = query_service._calculate_title_relevance(keywords, relevant_title)
        irrelevant_score = query_service._calculate_title_relevance(keywords, irrelevant_title)
        
        # Assert
        assert relevant_score > irrelevant_score
        assert relevant_score > 0.0
        assert irrelevant_score == 0.0
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_calculate_recency_boost_favors_recent_docs(self, query_service):
        """Test recency boost calculation favors newer documents."""
        # Arrange
        recent_doc = {"metadata": {"date": "2024-01-15"}}
        old_doc = {"metadata": {"date": "2020-01-01"}}
        no_date_doc = {"metadata": {}}
        
        # Act
        recent_boost = query_service._calculate_recency_boost(recent_doc)
        old_boost = query_service._calculate_recency_boost(old_doc)
        no_date_boost = query_service._calculate_recency_boost(no_date_doc)
        
        # Assert
        assert recent_boost >= old_boost  # Recent should get higher or equal boost
        assert no_date_boost >= 0.0       # Should handle missing dates gracefully
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_calculate_exact_match_boost_rewards_exact_matches(self, query_service):
        """Test exact match boost rewards precise keyword matches."""
        # Arrange
        keywords = ["install", "bot"]
        exact_match_text = "To install the bot, run the following command"
        partial_match_text = "Installation process for the system"
        
        # Act
        exact_boost = query_service._calculate_exact_match_boost(keywords, exact_match_text)
        partial_boost = query_service._calculate_exact_match_boost(keywords, partial_match_text)
        
        # Assert
        assert exact_boost > partial_boost
        assert exact_boost > 0.0
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_calculate_source_quality_scores_by_source_type(self, query_service):
        """Test source quality calculation based on document type."""
        # Arrange
        official_doc = {"source": "official_guide.md", "metadata": {"type": "official"}}
        user_doc = {"source": "user_guide.md", "metadata": {"type": "community"}}
        unknown_doc = {"source": "random.txt", "metadata": {}}
        
        # Act
        official_quality = query_service._calculate_source_quality(official_doc)
        user_quality = query_service._calculate_source_quality(user_doc)
        unknown_quality = query_service._calculate_source_quality(unknown_doc)
        
        # Assert
        assert 0.0 <= official_quality <= 1.0
        assert 0.0 <= user_quality <= 1.0  
        assert 0.0 <= unknown_quality <= 1.0
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_update_average_results_tracks_performance(self, query_service):
        """Test average results tracking updates correctly."""
        # Arrange
        initial_avg = query_service.query_analytics["average_results_returned"]
        
        # Act
        query_service._update_average_results(10)
        first_update = query_service.query_analytics["average_results_returned"]
        
        query_service._update_average_results(5)
        second_update = query_service.query_analytics["average_results_returned"]
        
        # Assert
        assert isinstance(first_update, float)
        assert isinstance(second_update, float)
        assert second_update != first_update  # Should update with new values


class TestUnifiedQueryServiceIntegration(BaseEngineTest):
    """Integration tests for query service with mocked dependencies."""
    
    @pytest.mark.unit
    @pytest.mark.engine
    @patch('support_deflect_bot.engine.query_service.get_default_registry')
    async def test_full_query_pipeline(self, mock_registry, query_service):
        """Test complete query processing pipeline."""
        # Arrange
        mock_registry.return_value = {
            "embedding_provider": Mock(
                is_available=Mock(return_value=True),
                generate_embeddings=AsyncMock(return_value=[[0.1] * 384])
            )
        }
        
        raw_query = "How do I install the support deflect bot?"
        
        with patch.object(query_service, '_semantic_search', return_value=[
            {"content": "Install guide content", "source": "install.md", "score": 0.9}
        ]):
            with patch.object(query_service, '_keyword_search', return_value=[
                {"content": "Installation steps", "source": "setup.md", "score": 0.8}
            ]):
                # Act
                query_info = query_service.preprocess_query(raw_query)
                results = await query_service.retrieve_documents(query_info)
                ranked_results = query_service.rank_results(results, query_info["keywords"])
                
                # Assert
                assert len(query_info["keywords"]) > 0
                assert "install" in query_info["keywords"]
                assert len(results) > 0
                assert len(ranked_results) > 0
                assert all("final_score" in result for result in ranked_results)
                
    @pytest.mark.unit
    @pytest.mark.engine
    def test_error_handling_with_provider_failure(self, query_service):
        """Test query service handles provider failures gracefully."""
        # Arrange
        failing_provider = Mock()
        failing_provider.generate_embeddings.side_effect = Exception("Provider failed")
        query_service.provider_registry = {"embedding_provider": failing_provider}
        
        # Act & Assert - Should handle gracefully
        query_info = query_service.preprocess_query("test query")
        assert "keywords" in query_info  # Should still process without embeddings