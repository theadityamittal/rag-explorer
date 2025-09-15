"""Unit tests for the embeddings module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from support_deflect_bot.data.embeddings import (
    embed_texts,
    embed_one,
    _embed_texts_new_system,
    _embed_texts_ollama_direct
)


class TestEmbeddings:
    """Test suite for embeddings module functions."""

    def test_embed_texts_empty_list(self):
        """Test embed_texts with empty input list."""
        result = embed_texts([])
        assert result == []

    @patch('support_deflect_bot.data.embeddings.USE_NEW_SYSTEM', True)
    @patch('support_deflect_bot.data.embeddings._embed_texts_new_system')
    def test_embed_texts_uses_new_system(self, mock_new_system):
        """Test that embed_texts uses new system when available."""
        mock_new_system.return_value = [[0.1, 0.2, 0.3]]
        texts = ["test text"]

        result = embed_texts(texts, batch_size=5)

        mock_new_system.assert_called_once_with(texts, 5)
        assert result == [[0.1, 0.2, 0.3]]

    @patch('support_deflect_bot.data.embeddings.USE_NEW_SYSTEM', False)
    @patch('support_deflect_bot.data.embeddings._embed_texts_ollama_direct')
    def test_embed_texts_uses_ollama_direct(self, mock_ollama_direct):
        """Test that embed_texts uses Ollama direct when new system unavailable."""
        mock_ollama_direct.return_value = [[0.1, 0.2, 0.3]]
        texts = ["test text"]

        result = embed_texts(texts, batch_size=5)

        mock_ollama_direct.assert_called_once_with(texts, 5)
        assert result == [[0.1, 0.2, 0.3]]

    @patch('support_deflect_bot.data.embeddings.get_default_registry')
    def test_embed_texts_new_system_success(self, mock_get_registry):
        """Test successful embedding with new provider system."""
        # Mock provider registry and chain
        mock_provider = Mock()
        mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_provider.get_config.return_value = Mock(name="test_provider")

        mock_registry = Mock()
        mock_registry.build_fallback_chain.return_value = [mock_provider]
        mock_get_registry.return_value = mock_registry

        texts = ["text1", "text2"]
        result = _embed_texts_new_system(texts, batch_size=10)

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_provider.embed_texts.assert_called_once_with(texts, batch_size=10)

    @patch('support_deflect_bot.data.embeddings.get_default_registry')
    def test_embed_texts_new_system_provider_fallback(self, mock_get_registry):
        """Test provider fallback in new system."""
        # Mock first provider that fails
        mock_provider1 = Mock()
        mock_provider1.embed_texts.side_effect = Exception("Provider 1 failed")
        mock_provider1.get_config.return_value = Mock(name="provider1")

        # Mock second provider that succeeds
        mock_provider2 = Mock()
        mock_provider2.embed_texts.return_value = [[0.1, 0.2, 0.3]]
        mock_provider2.get_config.return_value = Mock(name="provider2")

        mock_registry = Mock()
        mock_registry.build_fallback_chain.return_value = [mock_provider1, mock_provider2]
        mock_get_registry.return_value = mock_registry

        texts = ["test text"]
        result = _embed_texts_new_system(texts, batch_size=10)

        assert result == [[0.1, 0.2, 0.3]]
        mock_provider1.embed_texts.assert_called_once()
        mock_provider2.embed_texts.assert_called_once()

    @patch('support_deflect_bot.data.embeddings.get_default_registry')
    def test_embed_texts_new_system_all_providers_fail(self, mock_get_registry):
        """Test behavior when all providers fail."""
        # Mock provider that fails
        mock_provider = Mock()
        mock_provider.embed_texts.side_effect = Exception("Provider failed")
        mock_provider.get_config.return_value = Mock(name="failed_provider")

        mock_registry = Mock()
        mock_registry.build_fallback_chain.return_value = [mock_provider]
        mock_get_registry.return_value = mock_registry

        texts = ["text1", "text2"]
        result = _embed_texts_new_system(texts, batch_size=10)

        # Should return zero vectors
        assert len(result) == 2
        assert all(len(vec) == 768 for vec in result)
        assert all(all(x == 0.0 for x in vec) for vec in result)

    @patch('support_deflect_bot.data.embeddings.ollama')
    def test_embed_texts_ollama_direct_success(self, mock_ollama):
        """Test successful Ollama direct embedding."""
        mock_ollama.embeddings.side_effect = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]}
        ]

        texts = ["text1", "text2"]
        result = _embed_texts_ollama_direct(texts, batch_size=10)

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert mock_ollama.embeddings.call_count == 2

    @patch('support_deflect_bot.data.embeddings.ollama')
    def test_embed_texts_ollama_direct_empty_text(self, mock_ollama):
        """Test Ollama direct with empty text."""
        texts = ["", "   ", "valid text"]
        mock_ollama.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        result = _embed_texts_ollama_direct(texts, batch_size=10)

        assert len(result) == 3
        # First two should be zero vectors
        assert result[0] == [0.0] * 768
        assert result[1] == [0.0] * 768
        # Third should be the actual embedding
        assert result[2] == [0.1, 0.2, 0.3]
        # Ollama should only be called once for the valid text
        mock_ollama.embeddings.assert_called_once()

    @patch('support_deflect_bot.data.embeddings.ollama')
    def test_embed_texts_ollama_direct_failure(self, mock_ollama):
        """Test Ollama direct with embedding failure."""
        mock_ollama.embeddings.side_effect = Exception("Ollama failed")

        texts = ["text1", "text2"]
        result = _embed_texts_ollama_direct(texts, batch_size=10)

        # Should return zero vectors for failed embeddings
        assert len(result) == 2
        assert all(len(vec) == 768 for vec in result)
        assert all(all(x == 0.0 for x in vec) for vec in result)

    @patch('support_deflect_bot.data.embeddings.ollama')
    def test_embed_texts_ollama_direct_batching(self, mock_ollama):
        """Test Ollama direct with small batch size."""
        mock_ollama.embeddings.side_effect = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
            {"embedding": [0.7, 0.8, 0.9]}
        ]

        texts = ["text1", "text2", "text3"]
        result = _embed_texts_ollama_direct(texts, batch_size=2)

        assert len(result) == 3
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

    @patch('support_deflect_bot.data.embeddings.embed_texts')
    def test_embed_one_success(self, mock_embed_texts):
        """Test successful single text embedding."""
        mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]

        result = embed_one("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_embed_texts.assert_called_once_with(["test text"], batch_size=1)

    def test_embed_one_empty_text(self):
        """Test embed_one with empty text."""
        result = embed_one("")
        assert len(result) == 768
        assert all(x == 0.0 for x in result)

    def test_embed_one_whitespace_text(self):
        """Test embed_one with whitespace-only text."""
        result = embed_one("   \n\t  ")
        assert len(result) == 768
        assert all(x == 0.0 for x in result)

    @patch('support_deflect_bot.data.embeddings.embed_texts')
    def test_embed_one_failure(self, mock_embed_texts):
        """Test embed_one with failure."""
        mock_embed_texts.side_effect = Exception("Embedding failed")

        result = embed_one("test text")

        # Should return zero vector on failure
        assert len(result) == 768
        assert all(x == 0.0 for x in result)

    @patch('support_deflect_bot.data.embeddings.get_default_registry')
    def test_embed_texts_new_system_provider_error_types(self, mock_get_registry):
        """Test that different provider error types are handled correctly."""
        from support_deflect_bot.core.providers import ProviderError, ProviderUnavailableError

        # Mock provider that raises ProviderError
        mock_provider1 = Mock()
        mock_provider1.embed_texts.side_effect = ProviderError("Provider error")
        mock_provider1.get_config.return_value = Mock(name="provider1")

        # Mock provider that raises ProviderUnavailableError
        mock_provider2 = Mock()
        mock_provider2.embed_texts.side_effect = ProviderUnavailableError("Provider unavailable")
        mock_provider2.get_config.return_value = Mock(name="provider2")

        # Mock provider that succeeds
        mock_provider3 = Mock()
        mock_provider3.embed_texts.return_value = [[0.1, 0.2, 0.3]]
        mock_provider3.get_config.return_value = Mock(name="provider3")

        mock_registry = Mock()
        mock_registry.build_fallback_chain.return_value = [mock_provider1, mock_provider2, mock_provider3]
        mock_get_registry.return_value = mock_registry

        texts = ["test text"]
        result = _embed_texts_new_system(texts, batch_size=10)

        assert result == [[0.1, 0.2, 0.3]]
        # All providers should be tried
        mock_provider1.embed_texts.assert_called_once()
        mock_provider2.embed_texts.assert_called_once()
        mock_provider3.embed_texts.assert_called_once()

    @patch('support_deflect_bot.data.embeddings.ollama')
    def test_embed_texts_ollama_partial_batch_failure(self, mock_ollama):
        """Test Ollama direct with partial batch failure."""
        # First call succeeds, second call fails
        mock_ollama.embeddings.side_effect = [
            {"embedding": [0.1, 0.2, 0.3]},
            Exception("Second embedding failed")
        ]

        texts = ["text1", "text2"]
        result = _embed_texts_ollama_direct(texts, batch_size=1)

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.0] * 768  # Zero vector for failed embedding

    def test_embed_texts_batch_size_parameter(self):
        """Test that batch_size parameter is passed correctly."""
        with patch('support_deflect_bot.data.embeddings.USE_NEW_SYSTEM', True), \
             patch('support_deflect_bot.data.embeddings._embed_texts_new_system') as mock_new_system:

            mock_new_system.return_value = []
            embed_texts(["test"], batch_size=15)

            mock_new_system.assert_called_once_with(["test"], 15)

    @patch('support_deflect_bot.data.embeddings.get_default_registry')
    def test_embed_texts_new_system_empty_chain(self, mock_get_registry):
        """Test behavior with empty provider chain."""
        mock_registry = Mock()
        mock_registry.build_fallback_chain.return_value = []  # Empty chain
        mock_get_registry.return_value = mock_registry

        texts = ["test text"]
        result = _embed_texts_new_system(texts, batch_size=10)

        # Should return zero vectors when no providers available
        assert len(result) == 1
        assert len(result[0]) == 768
        assert all(x == 0.0 for x in result[0])

    @patch('support_deflect_bot.data.embeddings.ollama')
    def test_embed_texts_ollama_large_batch(self, mock_ollama):
        """Test Ollama direct with batch processing."""
        # Mock multiple responses for batched calls
        responses = [{"embedding": [0.1 * i, 0.2 * i, 0.3 * i]} for i in range(1, 6)]
        mock_ollama.embeddings.side_effect = responses

        texts = [f"text{i}" for i in range(1, 6)]
        result = _embed_texts_ollama_direct(texts, batch_size=2)

        assert len(result) == 5
        # Check that all embeddings are different (not zero vectors)
        assert all(sum(vec) != 0 for vec in result)

    def test_zero_vector_dimension_consistency(self):
        """Test that zero vectors have consistent dimensions."""
        # Test empty text cases
        result1 = embed_one("")
        result2 = embed_one("   ")

        assert len(result1) == 768
        assert len(result2) == 768
        assert len(result1) == len(result2)

        # Test failure cases
        with patch('support_deflect_bot.data.embeddings.embed_texts') as mock_embed:
            mock_embed.side_effect = Exception("Failed")
            result3 = embed_one("test")

        assert len(result3) == 768