"""Unit tests for Google Gemini provider."""

import os
from unittest.mock import patch, MagicMock, Mock

import pytest

from support_deflect_bot.core.providers.implementations.google_gemini import (
    GoogleGeminiFreeProvider,
    GoogleGeminiPaidProvider
)
from support_deflect_bot.core.providers.base import (
    ProviderError,
    ProviderUnavailableError,
    ProviderRateLimitError
)


class TestGoogleGeminiFreeProvider:
    """Test cases for Google Gemini Free provider."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api_key = "test_google_api_key"
    
    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        assert provider.api_key == self.api_key
        assert provider.default_llm_model == "gemini-2.5-flash-lite"
        assert provider.default_embedding_model == "gemini-embedding-001"
        
        config = provider.get_config()
        assert config.name == "google_gemini_free"
        assert config.provider_type.value == "combined"
        assert config.models_available == ["gemini-2.5-flash-lite", "gemini-embedding-001"]
    
    def test_provider_initialization_no_api_key(self):
        """Test provider initialization without API key."""
        provider = GoogleGeminiFreeProvider()
        
        assert provider.api_key is None
        assert not provider.is_available()
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_is_available_with_api_key(self, mock_genai):
        """Test availability check with API key."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        # Mock successful model list
        mock_genai.list_models.return_value = [
            Mock(name="models/gemini-2.5-flash-lite"),
            Mock(name="models/gemini-embedding-001")
        ]
        
        assert provider.is_available()
        mock_genai.configure.assert_called_with(api_key=self.api_key)
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_is_available_api_error(self, mock_genai):
        """Test availability check with API error."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        # Mock API error
        mock_genai.list_models.side_effect = Exception("API Error")
        
        assert not provider.is_available()
    
    def test_is_available_no_api_key(self):
        """Test availability check without API key."""
        provider = GoogleGeminiFreeProvider()
        
        assert not provider.is_available()
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_chat_success(self, mock_genai):
        """Test successful chat completion."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "This is a test response from Gemini"
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        
        result = provider.chat(
            system_prompt="You are a helpful assistant",
            user_prompt="Hello, how are you?",
            temperature=0.7,
            max_tokens=100
        )
        
        assert result == "This is a test response from Gemini"
        
        # Verify the model was called correctly
        mock_genai.GenerativeModel.assert_called_with("gemini-2.5-flash-lite")
        mock_genai.GenerativeModel.return_value.generate_content.assert_called_once()
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_chat_custom_model(self, mock_genai):
        """Test chat with custom model override."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        mock_response = Mock()
        mock_response.text = "Custom model response"
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        
        result = provider.chat(
            system_prompt="System prompt",
            user_prompt="User prompt",
            model="gemini-2.5-pro"
        )
        
        assert result == "Custom model response"
        mock_genai.GenerativeModel.assert_called_with("gemini-2.5-pro")
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_chat_rate_limit_error(self, mock_genai):
        """Test chat with rate limit error."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        # Mock rate limit error
        error = Exception("Resource has been exhausted")
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = error
        
        with pytest.raises(ProviderRateLimitError):
            provider.chat("system", "user")
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_chat_api_error(self, mock_genai):
        """Test chat with generic API error."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        # Mock generic error
        error = Exception("Invalid API key")
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = error
        
        with pytest.raises(ProviderError) as exc_info:
            provider.chat("system", "user")
        
        assert "Gemini API error" in str(exc_info.value)
    
    def test_chat_no_api_key(self):
        """Test chat without API key."""
        provider = GoogleGeminiFreeProvider()
        
        with pytest.raises(ProviderUnavailableError):
            provider.chat("system", "user")
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_embed_texts_success(self, mock_genai):
        """Test successful text embedding."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        # Mock successful embedding response
        mock_response = Mock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4]
        mock_genai.embed_content.return_value = mock_response
        
        texts = ["Hello world", "Test embedding"]
        result = provider.embed_texts(texts)
        
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3, 0.4]
        assert result[1] == [0.1, 0.2, 0.3, 0.4]
        
        # Verify embed_content was called for each text
        assert mock_genai.embed_content.call_count == 2
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_embed_texts_empty_list(self, mock_genai):
        """Test embedding empty text list."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        result = provider.embed_texts([])
        
        assert result == []
        mock_genai.embed_content.assert_not_called()
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_embed_texts_with_empty_strings(self, mock_genai):
        """Test embedding with empty strings."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        texts = ["Hello", "", "  ", "World"]
        
        # Mock response for non-empty texts
        mock_response = Mock()
        mock_response.embedding = [0.1, 0.2, 0.3]
        mock_genai.embed_content.return_value = mock_response
        
        result = provider.embed_texts(texts)
        
        assert len(result) == 4
        assert result[0] == [0.1, 0.2, 0.3]  # "Hello"
        assert len(result[1]) == 768  # Empty string -> zero vector
        assert all(x == 0.0 for x in result[1])
        assert len(result[2]) == 768  # Whitespace -> zero vector
        assert all(x == 0.0 for x in result[2])
        assert result[3] == [0.1, 0.2, 0.3]  # "World"
        
        # Should only call for non-empty texts
        assert mock_genai.embed_content.call_count == 2
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_embed_texts_batch_processing(self, mock_genai):
        """Test batch processing of embeddings."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        # Create many texts to test batching
        texts = [f"Text {i}" for i in range(15)]
        
        mock_response = Mock()
        mock_response.embedding = [0.1] * 768
        mock_genai.embed_content.return_value = mock_response
        
        result = provider.embed_texts(texts, batch_size=5)
        
        assert len(result) == 15
        assert all(len(embedding) == 768 for embedding in result)
        
        # Should be called once for each text
        assert mock_genai.embed_content.call_count == 15
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_embed_texts_error_handling(self, mock_genai):
        """Test error handling in embedding."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        # Mock error for some embeddings
        def side_effect(*args, **kwargs):
            if "error" in str(args):
                raise Exception("Embedding error")
            mock_response = Mock()
            mock_response.embedding = [0.1, 0.2]
            return mock_response
        
        mock_genai.embed_content.side_effect = side_effect
        
        texts = ["good text", "error text", "another good text"]
        result = provider.embed_texts(texts)
        
        assert len(result) == 3
        assert result[0] == [0.1, 0.2]  # Success
        assert len(result[1]) == 768 and all(x == 0.0 for x in result[1])  # Error -> zero vector
        assert result[2] == [0.1, 0.2]  # Success
    
    def test_embed_texts_no_api_key(self):
        """Test embedding without API key."""
        provider = GoogleGeminiFreeProvider()
        
        with pytest.raises(ProviderUnavailableError):
            provider.embed_texts(["test"])
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        # Default model
        dim = provider.get_embedding_dimension()
        assert dim == 768
        
        # Custom model
        dim = provider.get_embedding_dimension("text-embedding-004")
        assert dim == 768
        
        # Unknown model should return default
        dim = provider.get_embedding_dimension("unknown-model")
        assert dim == 768
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_count_tokens(self, mock_genai):
        """Test token counting functionality."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        mock_genai.GenerativeModel.return_value.count_tokens.return_value.total_tokens = 10
        
        count = provider.count_tokens("Hello world test")
        
        assert count == 10
        mock_genai.GenerativeModel.return_value.count_tokens.assert_called_once_with("Hello world test")
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_count_tokens_error(self, mock_genai):
        """Test token counting with error."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        mock_genai.GenerativeModel.return_value.count_tokens.side_effect = Exception("Count error")
        
        count = provider.count_tokens("test")
        
        # Should return reasonable estimate on error
        assert count > 0
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_health_check(self, mock_genai):
        """Test health check functionality."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        # Mock successful health check
        mock_response = Mock()
        mock_response.text = "I'm working fine"
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        
        result = provider.health_check()
        
        assert result is True
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_health_check_failure(self, mock_genai):
        """Test health check with failure."""
        provider = GoogleGeminiFreeProvider(api_key=self.api_key)
        
        # Mock health check error
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = Exception("Health check failed")
        
        result = provider.health_check()
        
        assert result is False


class TestGoogleGeminiPaidProvider:
    """Test cases for Google Gemini Paid provider."""
    
    def test_provider_initialization(self):
        """Test paid provider initialization."""
        provider = GoogleGeminiPaidProvider(api_key="test_key")
        
        config = provider.get_config()
        assert config.name == "google_gemini_paid"
        assert config.tier.value == "paid"
        assert config.gdpr_compliant is True
        assert config.regions_supported == ['global']
        
        # Should have higher rate limits than free tier
        assert config.rate_limit_rpm > 100
        assert config.rate_limit_tpm > 100000


class TestEnvironmentVariableIntegration:
    """Test environment variable integration."""
    
    def test_provider_uses_env_api_key(self):
        """Test that provider uses API key from environment."""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'env_test_key'}):
            provider = GoogleGeminiFreeProvider()
            
            # Should automatically pick up API key from environment
            assert provider.api_key == 'env_test_key'
    
    def test_provider_prefers_explicit_api_key(self):
        """Test that explicit API key overrides environment."""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'env_key'}):
            provider = GoogleGeminiFreeProvider(api_key='explicit_key')
            
            # Should use explicit key, not environment
            assert provider.api_key == 'explicit_key'
    
    @patch('support_deflect_bot.core.providers.implementations.google_gemini.genai')
    def test_model_override_from_settings(self, mock_genai):
        """Test that model overrides work from settings."""
        # Mock settings import
        with patch('support_deflect_bot.core.providers.implementations.google_gemini.GOOGLE_LLM_MODEL', 'custom-gemini-model'):
            with patch('support_deflect_bot.core.providers.implementations.google_gemini.GOOGLE_EMBEDDING_MODEL', 'custom-embedding-model'):
                provider = GoogleGeminiFreeProvider(api_key="test")
                
                assert provider.default_llm_model == 'custom-gemini-model'
                assert provider.default_embedding_model == 'custom-embedding-model'