import pytest
from unittest.mock import Mock, patch
from src.data.embeddings import embed_texts, embed_one


class TestEmbedTexts:
    """Test the embed_texts function."""

    @patch('src.data.embeddings.ollama.embeddings')
    def test_embed_texts_success(self, mock_ollama):
        """Test successful text embedding."""
        # Mock Ollama response
        mock_ollama.return_value = {"embedding": [0.1] * 768}
        
        texts = ["Hello world", "Test document"]
        result = embed_texts(texts)
        
        assert len(result) == 2
        assert all(len(emb) == 768 for emb in result)
        assert all(isinstance(emb[0], float) for emb in result)
        assert mock_ollama.call_count == 2

    @patch('src.data.embeddings.ollama.embeddings')
    def test_embed_texts_empty_input(self, mock_ollama):
        """Test embedding empty text list."""
        result = embed_texts([])
        assert result == []
        mock_ollama.assert_not_called()

    @patch('src.data.embeddings.ollama.embeddings')
    def test_embed_texts_empty_strings(self, mock_ollama):
        """Test embedding list with empty strings."""
        texts = ["", "   ", "valid text"]
        mock_ollama.return_value = {"embedding": [0.1] * 768}
        
        result = embed_texts(texts)
        
        assert len(result) == 3
        # First two should be zero vectors, third should be from ollama
        assert result[0] == [0.0] * 768
        assert result[1] == [0.0] * 768
        assert result[2] == [0.1] * 768
        # Ollama should only be called for valid text
        mock_ollama.assert_called_once()

    @patch('src.data.embeddings.ollama.embeddings')
    @patch('src.data.embeddings.logging.error')
    def test_embed_texts_api_error(self, mock_log, mock_ollama):
        """Test handling of Ollama API errors."""
        # Mock Ollama to raise an exception
        mock_ollama.side_effect = Exception("API Error")
        
        texts = ["test text"]
        result = embed_texts(texts, batch_size=1)
        
        assert len(result) == 1
        assert result[0] == [0.0] * 768  # Should return zero vector fallback
        mock_log.assert_called_once()

    @patch('src.data.embeddings.ollama.embeddings')
    def test_embed_texts_batch_processing(self, mock_ollama):
        """Test batch processing functionality."""
        mock_ollama.return_value = {"embedding": [0.1] * 768}
        
        texts = ["text1", "text2", "text3", "text4", "text5"]
        result = embed_texts(texts, batch_size=2)
        
        assert len(result) == 5
        # Should process in 3 batches: 2, 2, 1
        assert mock_ollama.call_count == 5

    @patch('src.data.embeddings.ollama.embeddings')
    def test_embed_texts_mixed_success_failure(self, mock_ollama):
        """Test partial success in batch processing."""
        # First call succeeds, second fails
        mock_ollama.side_effect = [
            {"embedding": [0.1] * 768},
            Exception("API Error"),
            {"embedding": [0.2] * 768}
        ]
        
        texts = ["text1", "text2", "text3"]
        
        with patch('src.data.embeddings.logging.error'):
            result = embed_texts(texts, batch_size=1)
        
        assert len(result) == 3
        assert result[0] == [0.1] * 768  # Success
        assert result[1] == [0.0] * 768  # Fallback due to error
        assert result[2] == [0.2] * 768  # Success


class TestEmbedOne:
    """Test the embed_one function."""

    @patch('src.data.embeddings.embed_texts')
    def test_embed_one_success(self, mock_embed_texts):
        """Test successful single text embedding."""
        mock_embed_texts.return_value = [[0.1] * 768]
        
        result = embed_one("test text")
        
        assert len(result) == 768
        assert isinstance(result[0], float)
        mock_embed_texts.assert_called_once_with(["test text"], batch_size=1)

    def test_embed_one_empty_text(self):
        """Test embedding empty text."""
        result = embed_one("")
        assert result == [0.0] * 768

    def test_embed_one_whitespace_only(self):
        """Test embedding whitespace-only text."""
        result = embed_one("   \n\t   ")
        assert result == [0.0] * 768

    @patch('src.data.embeddings.embed_texts')
    @patch('src.data.embeddings.logging.error')
    def test_embed_one_error_handling(self, mock_log, mock_embed_texts):
        """Test error handling in single text embedding."""
        mock_embed_texts.side_effect = Exception("Embedding failed")
        
        result = embed_one("test text")
        
        assert result == [0.0] * 768
        mock_log.assert_called_once()

    @patch('src.data.embeddings.embed_texts')
    def test_embed_one_valid_text(self, mock_embed_texts):
        """Test embedding valid non-empty text."""
        mock_embed_texts.return_value = [[0.5] * 768]
        
        result = embed_one("This is a valid test document.")
        
        assert result == [0.5] * 768
        mock_embed_texts.assert_called_once_with(["This is a valid test document."], batch_size=1)