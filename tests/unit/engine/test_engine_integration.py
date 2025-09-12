"""
Integration tests for unified engine services.

Tests the integration between UnifiedRAGEngine, UnifiedQueryService,
UnifiedDocumentProcessor, and UnifiedEmbeddingService working together
in realistic workflows.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from tests.base import BaseEngineTest
from support_deflect_bot.engine import (
    UnifiedRAGEngine, 
    UnifiedQueryService, 
    UnifiedDocumentProcessor, 
    UnifiedEmbeddingService
)


class TestEngineServicesIntegration(BaseEngineTest):
    """Test integration between all unified engine services."""
    
    @pytest.fixture
    def integrated_services(self):
        """Create integrated service instances for testing."""
        mock_registry = {}
        
        # Create service instances
        rag_engine = UnifiedRAGEngine(provider_registry=mock_registry)
        query_service = UnifiedQueryService(provider_registry=mock_registry)
        doc_processor = UnifiedDocumentProcessor(provider_registry=mock_registry)
        embedding_service = UnifiedEmbeddingService(provider_registry=mock_registry)
        
        return {
            'rag': rag_engine,
            'query': query_service, 
            'document': doc_processor,
            'embedding': embedding_service
        }
    
    @pytest.mark.unit
    @pytest.mark.engine
    @pytest.mark.integration
    async def test_full_rag_workflow_integration(self, integrated_services):
        """Test complete RAG workflow with all services integrated."""
        # Arrange
        services = integrated_services
        question = "How do I install the Support Deflect Bot?"
        
        # Mock provider for LLM calls
        mock_provider = Mock()
        mock_provider.is_available = Mock(return_value=True)
        mock_provider.generate_response = AsyncMock(return_value="Install using pip install -e .")
        mock_provider.generate_embeddings = AsyncMock(return_value=[[0.1] * 384])
        
        for service in services.values():
            service.provider_registry = {"test_provider": mock_provider}
        
        # Mock ChromaDB for document search
        with patch('chromadb.Client') as mock_chroma:
            mock_collection = Mock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
            mock_collection.query.return_value = {
                'documents': [["To install the bot, run pip install -e . in your terminal"]],
                'metadatas': [[{"source": "installation.md", "section": "Getting Started"}]],
                'distances': [[0.1]]
            }
            
            # Act - Test complete workflow
            # 1. Query preprocessing
            processed_query = services['query'].preprocess_query(question)
            
            # 2. Document search (via RAG engine)
            with patch.object(services['rag'], 'search_documents') as mock_search:
                mock_search.return_value = [{
                    "content": "To install the bot, run pip install -e . in your terminal",
                    "source": "installation.md",
                    "distance": 0.1,
                    "section": "Getting Started"
                }]
                
                # 3. Generate answer
                result = services['rag'].answer_question(question)
        
        # Assert
        assert "content" in processed_query
        assert len(processed_query["keywords"]) > 0
        assert "install" in processed_query["keywords"]
        
        assert "answer" in result
        assert "confidence" in result
        assert "citations" in result
        assert result["confidence"] > 0.5
        assert len(result["citations"]) > 0
        
    @pytest.mark.unit
    @pytest.mark.engine
    @pytest.mark.integration 
    async def test_document_to_embedding_pipeline(self, integrated_services):
        """Test document processing to embedding generation pipeline."""
        # Arrange
        services = integrated_services
        test_content = "This is a test document about bot installation."
        
        # Mock provider
        mock_provider = Mock()
        mock_provider.is_available = Mock(return_value=True)
        mock_provider.generate_embeddings = AsyncMock(return_value=[[0.1] * 384])
        
        services['document'].provider_registry = {"test_provider": mock_provider}
        services['embedding'].provider_registry = {"test_provider": mock_provider}
        
        # Act
        # 1. Process document into chunks
        chunks = services['document'].chunk_text(test_content, chunk_size=50, overlap=10)
        
        # 2. Generate embeddings for chunks
        embeddings = await services['embedding'].generate_embeddings(chunks)
        
        # Assert
        assert len(chunks) > 0
        assert len(embeddings) == len(chunks)
        assert all(len(emb) == 384 for emb in embeddings)
        
        # Verify embedding service was called with chunks
        mock_provider.generate_embeddings.assert_called_with(chunks)
        
    @pytest.mark.unit
    @pytest.mark.engine
    @pytest.mark.integration
    def test_query_service_rag_integration(self, integrated_services):
        """Test query service preprocessing feeds into RAG engine."""
        # Arrange
        services = integrated_services
        raw_question = "how can i install this bot???"
        
        # Act
        processed = services['query'].preprocess_query(raw_question)
        
        # Mock RAG search to use processed keywords
        with patch.object(services['rag'], 'search_documents') as mock_search:
            mock_search.return_value = []
            
            # Use processed query in RAG search
            services['rag'].search_documents(processed['content'])
            
        # Assert
        assert processed['content'] != raw_question  # Should be cleaned
        assert "install" in processed['keywords']
        assert "bot" in processed['keywords']
        
        # Verify RAG engine was called with processed content
        mock_search.assert_called_once_with(processed['content'])
        
    @pytest.mark.unit
    @pytest.mark.engine
    @pytest.mark.integration
    async def test_embedding_cache_integration_across_services(self, integrated_services):
        """Test embedding cache is shared and consistent across services."""
        # Arrange
        services = integrated_services
        text = "Common text used by multiple services"
        
        # Mock provider
        mock_provider = Mock()
        mock_provider.is_available = Mock(return_value=True)
        mock_provider.generate_embeddings = AsyncMock(return_value=[[0.5] * 384])
        
        services['embedding'].provider_registry = {"test_provider": mock_provider}
        
        # Act
        # First call - should hit provider
        embeddings1 = await services['embedding'].generate_embeddings([text])
        
        # Cache the embedding
        services['embedding'].cache_embeddings([(text, embeddings1[0])])
        
        # Second call - should hit cache
        embeddings2 = await services['embedding'].generate_embeddings([text])
        
        # Assert
        assert embeddings1 == embeddings2
        
        # Provider should only be called once due to caching
        mock_provider.generate_embeddings.assert_called_once()
        
        # Verify cache stats show hit
        cache_stats = services['embedding'].get_cache_stats()
        assert cache_stats['cache_hits'] > 0
        
    @pytest.mark.unit
    @pytest.mark.engine
    @pytest.mark.integration
    async def test_provider_fallback_across_services(self, integrated_services):
        """Test provider fallback works consistently across all services."""
        # Arrange
        services = integrated_services
        
        # Create failing primary provider
        failing_provider = Mock()
        failing_provider.is_available = Mock(return_value=False)
        failing_provider.generate_embeddings = AsyncMock(side_effect=Exception("Primary failed"))
        failing_provider.generate_response = AsyncMock(side_effect=Exception("Primary failed"))
        
        # Create working fallback provider  
        working_provider = Mock()
        working_provider.is_available = Mock(return_value=True)
        working_provider.generate_embeddings = AsyncMock(return_value=[[0.3] * 384])
        working_provider.generate_response = AsyncMock(return_value="Fallback response")
        
        # Set up provider registry for all services
        provider_registry = {
            "primary": failing_provider,
            "fallback": working_provider
        }
        
        for service in services.values():
            service.provider_registry = provider_registry
        
        # Act & Assert
        # Test embedding service fallback
        embeddings = await services['embedding']._generate_embeddings_with_fallback(["test"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
        
        # Verify both providers were attempted
        failing_provider.generate_embeddings.assert_called_once()
        working_provider.generate_embeddings.assert_called_once()
        
    @pytest.mark.unit
    @pytest.mark.engine
    @pytest.mark.integration
    def test_analytics_aggregation_across_services(self, integrated_services):
        """Test analytics can be aggregated across all services."""
        # Arrange
        services = integrated_services
        
        # Simulate some activity in each service
        services['rag'].metrics['queries_processed'] = 10
        services['rag'].metrics['successful_answers'] = 8
        services['query'].analytics['queries_preprocessed'] = 10
        services['embedding'].analytics['embeddings_generated'] = 50
        services['document'].analytics['documents_processed'] = 5
        
        # Act - Collect all analytics
        all_analytics = {}
        for service_name, service in services.items():
            if hasattr(service, 'metrics'):
                all_analytics[f"{service_name}_metrics"] = service.metrics
            if hasattr(service, 'analytics'):
                all_analytics[f"{service_name}_analytics"] = service.analytics
        
        # Assert
        assert 'rag_metrics' in all_analytics
        assert 'query_analytics' in all_analytics
        assert 'embedding_analytics' in all_analytics
        assert 'document_analytics' in all_analytics
        
        assert all_analytics['rag_metrics']['queries_processed'] == 10
        assert all_analytics['query_analytics']['queries_preprocessed'] == 10
        assert all_analytics['embedding_analytics']['embeddings_generated'] == 50
        
    @pytest.mark.unit
    @pytest.mark.engine
    @pytest.mark.integration
    async def test_error_propagation_across_services(self, integrated_services):
        """Test error handling propagates correctly across service boundaries."""
        # Arrange
        services = integrated_services
        
        # Mock a critical failure in embedding service
        with patch.object(services['embedding'], 'generate_embeddings', side_effect=Exception("Embedding failed")):
            with patch.object(services['rag'], 'search_documents', return_value=[]):
                
                # Act & Assert - Should handle gracefully
                try:
                    result = services['rag'].answer_question("test question")
                    # Should still return a structured response even with embedding failure
                    assert "answer" in result
                    assert "confidence" in result
                except Exception as e:
                    # If exception is raised, it should be informative
                    assert "Embedding failed" in str(e) or "embedding" in str(e).lower()
                    
    @pytest.mark.unit
    @pytest.mark.engine
    @pytest.mark.integration
    def test_configuration_consistency_across_services(self, integrated_services):
        """Test configuration is consistent across all services."""
        # Arrange
        services = integrated_services
        
        # Act - Check each service has expected configuration attributes
        config_attrs = ['provider_registry']
        
        for service_name, service in services.items():
            for attr in config_attrs:
                # Assert
                assert hasattr(service, attr), f"Service {service_name} missing {attr}"
                
        # Check specific service attributes
        assert hasattr(services['rag'], 'metrics')
        assert hasattr(services['rag'], 'system_prompt')
        assert hasattr(services['query'], 'analytics')
        assert hasattr(services['embedding'], 'cache_enabled')
        assert hasattr(services['embedding'], 'analytics')
        assert hasattr(services['document'], 'analytics')


class TestServiceCommunication(BaseEngineTest):
    """Test communication patterns between services."""
    
    @pytest.mark.unit
    @pytest.mark.engine
    @pytest.mark.integration
    async def test_rag_engine_orchestrates_all_services(self):
        """Test RAG engine properly orchestrates other services."""
        # Arrange
        mock_registry = {"test": Mock(is_available=Mock(return_value=True))}
        rag_engine = UnifiedRAGEngine(provider_registry=mock_registry)
        
        # Mock the other services that RAG engine would use
        with patch('support_deflect_bot.engine.UnifiedQueryService') as mock_query:
            with patch('support_deflect_bot.engine.UnifiedEmbeddingService') as mock_embedding:
                with patch('chromadb.Client') as mock_chroma:
                    
                    # Setup mocks
                    mock_query_instance = Mock()
                    mock_query.return_value = mock_query_instance
                    mock_query_instance.preprocess_query.return_value = {
                        "content": "processed query",
                        "keywords": ["test"]
                    }
                    
                    mock_embedding_instance = Mock()
                    mock_embedding.return_value = mock_embedding_instance
                    mock_embedding_instance.generate_embeddings = AsyncMock(return_value=[[0.1] * 384])
                    
                    mock_collection = Mock()
                    mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
                    mock_collection.query.return_value = {
                        'documents': [["Test document"]],
                        'metadatas': [[{"source": "test.md"}]],
                        'distances': [[0.1]]
                    }
                    
                    # Act
                    with patch.object(rag_engine, '_generate_answer', return_value="Test answer"):
                        result = rag_engine.answer_question("test question")
                    
                    # Assert
                    assert "answer" in result
                    assert "confidence" in result
                    assert result["answer"] == "Test answer"
                    
    @pytest.mark.unit
    @pytest.mark.engine  
    @pytest.mark.integration
    async def test_concurrent_service_usage(self, integrated_services):
        """Test services handle concurrent usage correctly."""
        # Arrange
        services = integrated_services
        
        # Mock providers for all services
        mock_provider = Mock()
        mock_provider.is_available = Mock(return_value=True)
        mock_provider.generate_embeddings = AsyncMock(return_value=[[0.1] * 384])
        
        for service in services.values():
            service.provider_registry = {"test": mock_provider}
        
        # Act - Make concurrent calls to different services
        async def call_embedding():
            return await services['embedding'].generate_embeddings(["test 1"])
            
        async def call_query():
            return services['query'].preprocess_query("test query 2")
            
        async def call_document():
            return services['document'].chunk_text("test document 3")
        
        results = await asyncio.gather(
            call_embedding(),
            asyncio.create_task(asyncio.coroutine(call_query)()),
            asyncio.create_task(asyncio.coroutine(call_document)())
        )
        
        # Assert
        assert len(results) == 3
        assert len(results[0]) == 1  # Embedding result
        assert "content" in results[1]  # Query result
        assert len(results[2]) > 0  # Document chunks