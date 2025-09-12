"""
Unit tests for UnifiedEmbeddingService.

Tests the embedding service that handles vector generation, caching,
provider management, and embedding dimension handling.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, mock_open
from tests.base import BaseEngineTest
from support_deflect_bot.engine import UnifiedEmbeddingService


class TestUnifiedEmbeddingService(BaseEngineTest):
    """Test the UnifiedEmbeddingService."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create UnifiedEmbeddingService instance for testing."""
        mock_registry = {}
        return UnifiedEmbeddingService(provider_registry=mock_registry, cache_enabled=True)
    
    @pytest.mark.unit
    @pytest.mark.engine
    def test_init_creates_correct_attributes(self, embedding_service):
        """Test embedding service initialization creates correct attributes."""
        # Assert
        assert hasattr(embedding_service, 'provider_registry')
        assert hasattr(embedding_service, 'cache_enabled')
        assert hasattr(embedding_service, 'cache_path')
        assert hasattr(embedding_service, 'embedding_cache')
        assert hasattr(embedding_service, 'analytics')
        
        # Check analytics structure
        expected_analytics = ["embeddings_generated", "cache_hits", "cache_misses",
                            "provider_calls", "average_dimension", "total_tokens_processed"]
        for key in expected_analytics:
            assert key in embedding_service.analytics
            
    @pytest.mark.unit
    @pytest.mark.engine
    async def test_generate_embeddings_single_text(self, embedding_service):
        """Test embedding generation for single text."""
        # Arrange
        text = "This is a test document for embedding generation."
        mock_provider = Mock()
        mock_provider.is_available = Mock(return_value=True)
        mock_provider.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3] * 128])  # 384-dim
        
        embedding_service.provider_registry = {"test_provider": mock_provider}
        
        # Act
        embeddings = await embedding_service.generate_embeddings([text])
        
        # Assert
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
        assert all(isinstance(val, float) for val in embeddings[0])
        mock_provider.generate_embeddings.assert_called_once_with([text])
        
    @pytest.mark.unit
    @pytest.mark.engine
    async def test_generate_embeddings_batch_texts(self, embedding_service):
        """Test embedding generation for multiple texts."""
        # Arrange
        texts = [
            "First test document",
            "Second test document", 
            "Third test document"
        ]
        mock_provider = Mock()
        mock_provider.is_available = Mock(return_value=True)
        mock_provider.generate_embeddings = AsyncMock(return_value=[
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384
        ])
        
        embedding_service.provider_registry = {"test_provider": mock_provider}
        
        # Act
        embeddings = await embedding_service.generate_embeddings(texts)
        
        # Assert
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        mock_provider.generate_embeddings.assert_called_once_with(texts)
        assert embedding_service.analytics["embeddings_generated"] >= 3
        
    @pytest.mark.unit
    @pytest.mark.engine
    async def test_batch_embed_processes_large_batches(self, embedding_service):
        """Test batch embedding processing handles large text batches."""
        # Arrange
        texts = [f"Document {i} content" for i in range(100)]  # Large batch
        batch_size = 10
        
        mock_provider = Mock()
        mock_provider.is_available = Mock(return_value=True)
        mock_provider.generate_embeddings = AsyncMock(return_value=[[0.1] * 384] * 10)
        
        embedding_service.provider_registry = {"test_provider": mock_provider}
        
        # Act
        embeddings = await embedding_service.batch_embed(texts, batch_size=batch_size)
        
        # Assert
        assert len(embeddings) == 100
        # Should have made 10 batch calls (100 texts / 10 per batch)
        assert mock_provider.generate_embeddings.call_count == 10
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_get_embedding_dimension_returns_correct_dimension(self, embedding_service):
        """Test embedding dimension retrieval."""
        # Arrange
        mock_provider = Mock()
        mock_provider.embedding_dimension = 1536  # OpenAI dimension
        embedding_service.provider_registry = {"openai": mock_provider}
        
        # Act
        dimension = embedding_service.get_embedding_dimension("openai")
        
        # Assert
        assert dimension == 1536
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_get_embedding_dimension_fallback_to_default(self, embedding_service):
        """Test embedding dimension fallback to default when provider not specified."""
        # Act
        dimension = embedding_service.get_embedding_dimension()
        
        # Assert
        assert isinstance(dimension, int)
        assert dimension > 0  # Should return a valid dimension
        
    @pytest.mark.unit
    @pytest.mark.engine
    async def test_validate_providers_tests_all_providers(self, embedding_service):
        """Test provider validation tests all available providers."""
        # Arrange
        working_provider = Mock()
        working_provider.is_available = Mock(return_value=True)
        working_provider.generate_embeddings = AsyncMock(return_value=[[0.1] * 384])
        
        broken_provider = Mock()
        broken_provider.is_available = Mock(return_value=False)
        broken_provider.generate_embeddings = AsyncMock(side_effect=Exception("Provider failed"))
        
        embedding_service.provider_registry = {
            "working": working_provider,
            "broken": broken_provider
        }
        
        # Act
        validation = await embedding_service.validate_providers()
        
        # Assert
        assert "working" in validation
        assert "broken" in validation
        assert validation["working"]["available"] is True
        assert validation["broken"]["available"] is False
        assert "error" in validation["broken"]
        
    @pytest.mark.unit
    @pytest.mark.engine
    async def test_cache_embeddings_stores_and_retrieves(self, embedding_service):
        """Test embedding caching stores and retrieves embeddings."""
        # Arrange
        text_embedding_pairs = [
            ("test text 1", [0.1] * 384),
            ("test text 2", [0.2] * 384)
        ]
        
        # Act - Store embeddings
        embedding_service.cache_embeddings(text_embedding_pairs)
        
        # Verify cached embeddings can be retrieved
        cached_1 = embedding_service.embedding_cache.get(embedding_service._get_cache_key("test text 1"))
        cached_2 = embedding_service.embedding_cache.get(embedding_service._get_cache_key("test text 2"))
        
        # Assert
        assert cached_1 is not None
        assert cached_2 is not None
        assert cached_1 == [0.1] * 384
        assert cached_2 == [0.2] * 384
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_get_cache_stats_returns_metrics(self, embedding_service):
        """Test cache statistics returns comprehensive metrics."""
        # Arrange - Add some cache entries
        embedding_service.embedding_cache = {
            "key1": [0.1] * 384,
            "key2": [0.2] * 384,
            "key3": [0.3] * 384
        }
        embedding_service.analytics["cache_hits"] = 10
        embedding_service.analytics["cache_misses"] = 5
        
        # Act
        stats = embedding_service.get_cache_stats()
        
        # Assert
        expected_keys = ["cache_size", "cache_hits", "cache_misses", 
                        "hit_rate", "memory_usage"]
        for key in expected_keys:
            assert key in stats
            
        assert stats["cache_size"] == 3
        assert stats["cache_hits"] == 10
        assert stats["cache_misses"] == 5
        assert stats["hit_rate"] == 10 / 15  # 10 hits out of 15 total
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_clear_cache_empties_cache(self, embedding_service):
        """Test cache clearing empties the cache."""
        # Arrange
        embedding_service.embedding_cache = {
            "key1": [0.1] * 384,
            "key2": [0.2] * 384
        }
        
        # Act
        success = embedding_service.clear_cache()
        
        # Assert
        assert success is True
        assert len(embedding_service.embedding_cache) == 0
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_get_analytics_returns_comprehensive_metrics(self, embedding_service):
        """Test analytics returns comprehensive performance metrics."""
        # Arrange
        embedding_service.analytics.update({
            "embeddings_generated": 100,
            "cache_hits": 25,
            "provider_calls": 75,
            "total_tokens_processed": 10000
        })
        
        # Act
        analytics = embedding_service.get_analytics()
        
        # Assert
        expected_keys = ["embeddings_generated", "cache_hits", "cache_misses",
                        "provider_calls", "cache_efficiency", "tokens_per_call"]
        for key in expected_keys:
            assert key in analytics
            
        assert analytics["embeddings_generated"] == 100
        assert analytics["cache_efficiency"] > 0  # Should calculate efficiency
        
    @pytest.mark.unit
    @pytest.mark.engine
    async def test_generate_embeddings_with_fallback_tries_multiple_providers(self, embedding_service):
        """Test embedding generation with provider fallback."""
        # Arrange
        primary_provider = Mock()
        primary_provider.is_available = Mock(return_value=False)
        primary_provider.generate_embeddings = AsyncMock(side_effect=Exception("Primary failed"))
        
        fallback_provider = Mock() 
        fallback_provider.is_available = Mock(return_value=True)
        fallback_provider.generate_embeddings = AsyncMock(return_value=[[0.1] * 384])
        
        embedding_service.provider_registry = {
            "primary": primary_provider,
            "fallback": fallback_provider
        }
        
        # Act
        embeddings = await embedding_service._generate_embeddings_with_fallback(["test text"])
        
        # Assert
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
        # Should have tried primary first, then fallback
        primary_provider.generate_embeddings.assert_called_once()
        fallback_provider.generate_embeddings.assert_called_once()
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_get_cache_key_generates_consistent_keys(self, embedding_service):
        """Test cache key generation is consistent for same text."""
        # Arrange
        text = "This is a test text for cache key generation"
        
        # Act
        key1 = embedding_service._get_cache_key(text)
        key2 = embedding_service._get_cache_key(text)
        key3 = embedding_service._get_cache_key("Different text")
        
        # Assert
        assert key1 == key2  # Same text should generate same key
        assert key1 != key3  # Different text should generate different key
        assert isinstance(key1, str)
        assert len(key1) > 0
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_get_default_dimension_returns_valid_dimension(self, embedding_service):
        """Test default dimension returns a valid embedding dimension."""
        # Act
        dimension = embedding_service._get_default_dimension()
        
        # Assert
        assert isinstance(dimension, int)
        assert dimension in [384, 768, 1536, 3072]  # Common embedding dimensions
        
    @pytest.mark.unit
    @pytest.mark.engine
    @patch('builtins.open', new_callable=mock_open, read_data='{"key1": [0.1, 0.2, 0.3]}')
    @patch('os.path.exists')
    def test_load_cache_loads_from_file(self, mock_exists, mock_file, embedding_service):
        """Test cache loading from file."""
        # Arrange
        mock_exists.return_value = True
        
        # Act
        embedding_service._load_cache()
        
        # Assert
        assert len(embedding_service.embedding_cache) > 0
        mock_file.assert_called_once()
        
    @pytest.mark.unit
    @pytest.mark.engine
    @patch('os.path.exists')
    def test_load_cache_handles_missing_file(self, mock_exists, embedding_service):
        """Test cache loading handles missing cache file."""
        # Arrange
        mock_exists.return_value = False
        
        # Act
        embedding_service._load_cache()
        
        # Assert
        # Should not raise error and cache should remain empty
        assert len(embedding_service.embedding_cache) == 0
        
    @pytest.mark.unit
    @pytest.mark.engine
    @patch('builtins.open', new_callable=mock_open)
    def test_save_cache_writes_to_file(self, mock_file, embedding_service):
        """Test cache saving writes to file."""
        # Arrange
        embedding_service.embedding_cache = {
            "key1": [0.1, 0.2, 0.3],
            "key2": [0.4, 0.5, 0.6]
        }
        
        # Act
        embedding_service._save_cache()
        
        # Assert
        mock_file.assert_called_once()
        # Verify JSON was written
        handle = mock_file()
        handle.write.assert_called()


class TestUnifiedEmbeddingServiceIntegration(BaseEngineTest):
    """Integration tests for embedding service with mocked dependencies."""
    
    @pytest.mark.unit
    @pytest.mark.engine
    async def test_full_embedding_pipeline_with_caching(self):
        """Test complete embedding pipeline with caching."""
        # Arrange
        mock_provider = Mock()
        mock_provider.is_available = Mock(return_value=True)
        mock_provider.generate_embeddings = AsyncMock(return_value=[
            [0.1] * 384,
            [0.2] * 384
        ])
        
        embedding_service = UnifiedEmbeddingService(
            provider_registry={"test": mock_provider},
            cache_enabled=True
        )
        
        texts = ["First document", "Second document"]
        
        # Act - First call should hit provider
        embeddings1 = await embedding_service.generate_embeddings(texts)
        # Second call should hit cache
        embeddings2 = await embedding_service.generate_embeddings(texts)
        
        # Assert
        assert len(embeddings1) == 2
        assert len(embeddings2) == 2
        assert embeddings1 == embeddings2  # Should be identical from cache
        
        # Provider should only be called once due to caching
        mock_provider.generate_embeddings.assert_called_once()
        
        # Check cache stats
        stats = embedding_service.get_cache_stats()
        assert stats["cache_hits"] > 0
        
    @pytest.mark.unit
    @pytest.mark.engine
    async def test_error_handling_with_all_providers_failing(self):
        """Test embedding service handles all providers failing."""
        # Arrange
        failing_provider1 = Mock()
        failing_provider1.is_available = Mock(return_value=False)
        failing_provider1.generate_embeddings = AsyncMock(side_effect=Exception("Provider 1 failed"))
        
        failing_provider2 = Mock()
        failing_provider2.is_available = Mock(return_value=False)
        failing_provider2.generate_embeddings = AsyncMock(side_effect=Exception("Provider 2 failed"))
        
        embedding_service = UnifiedEmbeddingService(provider_registry={
            "provider1": failing_provider1,
            "provider2": failing_provider2
        })
        
        # Act & Assert
        with pytest.raises(Exception):  # Should raise an exception when all providers fail
            await embedding_service.generate_embeddings(["test text"])
            
    @pytest.mark.unit
    @pytest.mark.engine
    async def test_performance_with_concurrent_requests(self, embedding_service):
        """Test embedding service handles concurrent requests correctly."""
        # Arrange
        mock_provider = Mock()
        mock_provider.is_available = Mock(return_value=True)
        mock_provider.generate_embeddings = AsyncMock(return_value=[[0.1] * 384])
        
        embedding_service.provider_registry = {"test": mock_provider}
        
        # Act - Make multiple concurrent requests
        tasks = [
            embedding_service.generate_embeddings([f"Document {i}"])
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert len(results) == 10
        assert all(len(result) == 1 for result in results)  # Each request returned 1 embedding
        assert all(len(result[0]) == 384 for result in results)  # Each embedding has correct dimension