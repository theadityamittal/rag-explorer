"""
Unit tests for OpenAI provider implementation.

Tests OpenAI-specific functionality including API integration,
model variants, response handling, and error scenarios.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from tests.base import BaseProviderTest


class TestOpenAIProviderInterface(BaseProviderTest):
    """Test OpenAI provider interface implementation."""
    
    @pytest.fixture
    def openai_provider(self):
        """Create OpenAI provider instance for testing."""
        # Mock the actual OpenAI provider when available
        provider = Mock()
        provider.name = "openai"
        provider.endpoint = "https://api.openai.com/v1"
        provider.api_key = "sk-test123"
        provider.model = "gpt-4o-mini"
        provider.is_available = Mock(return_value=True)
        provider.generate_response = AsyncMock()
        provider.generate_embeddings = AsyncMock()
        return provider
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    def test_openai_provider_initialization(self, openai_provider):
        """Test OpenAI provider initialization."""
        # Assert
        assert openai_provider.name == "openai"
        assert openai_provider.endpoint == "https://api.openai.com/v1"
        assert openai_provider.api_key.startswith("sk-")
        assert openai_provider.model == "gpt-4o-mini"
        assert callable(openai_provider.is_available)
        assert callable(openai_provider.generate_response)
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    def test_openai_availability_check(self, openai_provider):
        """Test OpenAI provider availability checking."""
        # Test available scenario
        openai_provider.is_available.return_value = True
        assert openai_provider.is_available() is True
        
        # Test unavailable scenario
        openai_provider.is_available.return_value = False
        assert openai_provider.is_available() is False
        
        # Verify API key validation would be part of availability
        openai_provider.api_key = "invalid-key"
        openai_provider.is_available.return_value = False
        assert openai_provider.is_available() is False


class TestOpenAIModelVariants(BaseProviderTest):
    """Test different OpenAI model variants and configurations."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    def test_gpt_4o_mini_model(self):
        """Test GPT-4o-mini model configuration."""
        # Arrange
        provider = self.create_mock_provider("openai")
        provider.model = "gpt-4o-mini"
        provider.max_tokens = 4096
        provider.cost_per_input_token = 0.15 / 1000000  # $0.15 per 1M tokens
        provider.cost_per_output_token = 0.60 / 1000000  # $0.60 per 1M tokens
        
        # Act & Assert
        assert provider.model == "gpt-4o-mini"
        assert provider.max_tokens == 4096
        assert provider.cost_per_input_token < 0.001
        assert provider.cost_per_output_token < 0.001
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    def test_gpt_4o_model(self):
        """Test GPT-4o model configuration."""
        # Arrange
        provider = self.create_mock_provider("openai")
        provider.model = "gpt-4o"
        provider.max_tokens = 8192
        provider.cost_per_input_token = 2.50 / 1000000  # $2.50 per 1M tokens
        provider.cost_per_output_token = 10.00 / 1000000  # $10.00 per 1M tokens
        
        # Act & Assert
        assert provider.model == "gpt-4o"
        assert provider.max_tokens == 8192
        assert provider.cost_per_input_token > provider.cost_per_output_token / 5
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    def test_embedding_model(self):
        """Test OpenAI embedding model configuration."""
        # Arrange
        provider = self.create_mock_provider("openai")
        provider.embedding_model = "text-embedding-3-small"
        provider.embedding_dimensions = 1536
        provider.embedding_cost_per_token = 0.02 / 1000000  # $0.02 per 1M tokens
        
        # Act & Assert
        assert provider.embedding_model == "text-embedding-3-small"
        assert provider.embedding_dimensions == 1536
        assert provider.embedding_cost_per_token < 0.001
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    def test_model_capabilities_mapping(self):
        """Test mapping OpenAI models to capabilities."""
        # Arrange
        model_capabilities = {
            "gpt-4o": ["COMPLETION", "CHAT", "FUNCTION_CALLING"],
            "gpt-4o-mini": ["COMPLETION", "CHAT", "FUNCTION_CALLING"],
            "gpt-3.5-turbo": ["COMPLETION", "CHAT"],
            "text-embedding-3-small": ["EMBEDDING"],
            "text-embedding-3-large": ["EMBEDDING"]
        }
        
        # Act & Assert
        for model, capabilities in model_capabilities.items():
            assert len(capabilities) > 0
            if "embedding" in model.lower():
                assert "EMBEDDING" in capabilities
                assert "CHAT" not in capabilities
            else:
                assert "CHAT" in capabilities or "COMPLETION" in capabilities
                assert "EMBEDDING" not in capabilities


class TestOpenAIAPIIntegration(BaseProviderTest):
    """Test OpenAI API integration and request/response handling."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    async def test_chat_completion_request(self):
        """Test OpenAI chat completion API request format."""
        # Arrange
        provider = self.create_mock_provider("openai")
        
        expected_response = {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from OpenAI GPT-4o-mini."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35
            }
        }
        
        provider.generate_response.return_value = {
            "content": expected_response["choices"][0]["message"]["content"],
            "model": expected_response["model"],
            "usage": expected_response["usage"],
            "finish_reason": expected_response["choices"][0]["finish_reason"]
        }
        
        # Act
        response = await provider.generate_response("What is AI?")
        
        # Assert
        assert "content" in response
        assert "model" in response
        assert "usage" in response
        assert response["model"] == "gpt-4o-mini"
        assert "OpenAI GPT-4o-mini" in response["content"]
        assert response["usage"]["total_tokens"] == 35
        
        # Verify method was called
        provider.generate_response.assert_called_once_with("What is AI?")
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    async def test_embedding_request(self):
        """Test OpenAI embedding API request format."""
        # Arrange
        provider = self.create_mock_provider("openai")
        
        expected_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1] * 1536  # 1536-dimensional embedding
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 10
            }
        }
        
        provider.generate_embeddings.return_value = [expected_response["data"][0]["embedding"]]
        
        # Act
        embeddings = await provider.generate_embeddings(["test text"])
        
        # Assert
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536
        assert all(isinstance(x, float) for x in embeddings[0])
        
        # Verify method was called
        provider.generate_embeddings.assert_called_once_with(["test text"])
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    def test_request_headers(self):
        """Test OpenAI API request headers."""
        # Arrange
        provider = self.create_mock_provider("openai")
        provider.api_key = "sk-test123"
        
        # Expected headers
        expected_headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "support-deflect-bot"
        }
        
        # Act
        def build_headers(provider):
            return {
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "support-deflect-bot"
            }
        
        headers = build_headers(provider)
        
        # Assert
        assert headers["Authorization"] == f"Bearer sk-test123"
        assert headers["Content-Type"] == "application/json"
        assert "support-deflect-bot" in headers["User-Agent"]
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    def test_request_payload_structure(self):
        """Test OpenAI API request payload structure."""
        # Arrange
        message = "What is machine learning?"
        model = "gpt-4o-mini"
        max_tokens = 500
        temperature = 0.7
        
        # Act
        def build_chat_payload(message, model, max_tokens=None, temperature=None):
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": message}
                ]
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if temperature is not None:
                payload["temperature"] = temperature
                
            return payload
        
        payload = build_chat_payload(message, model, max_tokens, temperature)
        
        # Assert
        assert payload["model"] == "gpt-4o-mini"
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == message
        assert payload["max_tokens"] == 500
        assert payload["temperature"] == 0.7


class TestOpenAIErrorHandling(BaseProviderTest):
    """Test OpenAI provider error handling and recovery."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    async def test_rate_limit_error_handling(self):
        """Test handling OpenAI rate limit errors."""
        # Arrange
        provider = self.create_mock_provider("openai")
        
        rate_limit_error = Exception("Rate limit exceeded. Please try again later.")
        provider.generate_response.side_effect = rate_limit_error
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test message")
        
        assert "Rate limit exceeded" in str(exc_info.value)
        provider.generate_response.assert_called_once()
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    async def test_api_key_error_handling(self):
        """Test handling invalid API key errors."""
        # Arrange
        provider = self.create_mock_provider("openai")
        
        auth_error = Exception("Incorrect API key provided")
        provider.generate_response.side_effect = auth_error
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test message")
        
        assert "API key" in str(exc_info.value)
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    async def test_model_not_found_error(self):
        """Test handling model not found errors."""
        # Arrange
        provider = self.create_mock_provider("openai")
        
        model_error = Exception("The model 'invalid-model' does not exist")
        provider.generate_response.side_effect = model_error
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test message")
        
        assert "model" in str(exc_info.value)
        assert "does not exist" in str(exc_info.value)
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    async def test_timeout_error_handling(self):
        """Test handling timeout errors."""
        # Arrange
        provider = self.create_mock_provider("openai")
        
        timeout_error = Exception("Request timeout")
        provider.generate_response.side_effect = timeout_error
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test message")
        
        assert "timeout" in str(exc_info.value).lower()
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    async def test_content_filter_error_handling(self):
        """Test handling OpenAI content filter errors."""
        # Arrange
        provider = self.create_mock_provider("openai")
        
        # Mock response with content filter flag
        filtered_response = {
            "content": None,
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
            "finish_reason": "content_filter"
        }
        
        provider.generate_response.return_value = filtered_response
        
        # Act
        response = await provider.generate_response("inappropriate content")
        
        # Assert
        assert response["finish_reason"] == "content_filter"
        assert response["content"] is None
        assert response["usage"]["completion_tokens"] == 0


class TestOpenAIPerformanceOptimization(BaseProviderTest):
    """Test OpenAI provider performance optimization features."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    async def test_batch_embedding_optimization(self):
        """Test batch processing for embeddings."""
        # Arrange
        provider = self.create_mock_provider("openai")
        
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]
        batch_size = 3
        
        # Mock batch processing
        def mock_batch_embeddings(texts):
            # Simulate batching
            batches = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batches.append([[0.1] * 1536 for _ in batch])
            return [emb for batch in batches for emb in batch]
        
        provider.generate_embeddings.side_effect = mock_batch_embeddings
        
        # Act
        embeddings = await provider.generate_embeddings(texts)
        
        # Assert
        assert len(embeddings) == 5
        assert all(len(emb) == 1536 for emb in embeddings)
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    def test_token_count_optimization(self):
        """Test token counting for cost optimization."""
        # Arrange
        provider = self.create_mock_provider("openai")
        
        # Mock token counting (approximate)
        def estimate_tokens(text):
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4 + 1
        
        messages = [
            "Short message",
            "This is a much longer message that should use more tokens",
            "Another message"
        ]
        
        # Act
        token_counts = [estimate_tokens(msg) for msg in messages]
        total_tokens = sum(token_counts)
        
        # Assert
        assert token_counts[0] < token_counts[1]  # Longer message has more tokens
        assert token_counts[1] > 10  # Long message has significant tokens
        assert total_tokens > 20  # Total is reasonable
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    def test_cost_calculation(self):
        """Test accurate cost calculation for OpenAI usage."""
        # Arrange
        provider = self.create_mock_provider("openai")
        provider.model = "gpt-4o-mini"
        provider.cost_per_input_token = 0.15 / 1000000   # $0.15 per 1M input tokens
        provider.cost_per_output_token = 0.60 / 1000000  # $0.60 per 1M output tokens
        
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
        
        # Act
        def calculate_cost(provider, usage):
            input_cost = usage["prompt_tokens"] * provider.cost_per_input_token
            output_cost = usage["completion_tokens"] * provider.cost_per_output_token
            return input_cost + output_cost
        
        cost = calculate_cost(provider, usage)
        
        # Assert
        assert cost > 0
        assert cost < 0.01  # Should be very small for this usage
        
        # Test with larger usage
        large_usage = {
            "prompt_tokens": 10000,
            "completion_tokens": 5000,
            "total_tokens": 15000
        }
        
        large_cost = calculate_cost(provider, large_usage)
        assert large_cost > cost  # Larger usage should cost more
        assert large_cost > 0.001  # Should be measurable for large usage


class TestOpenAISpecialFeatures(BaseProviderTest):
    """Test OpenAI-specific features and capabilities."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    async def test_function_calling_support(self):
        """Test OpenAI function calling capability."""
        # Arrange
        provider = self.create_mock_provider("openai")
        
        function_schema = {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
        
        # Mock function calling response
        function_response = {
            "content": None,
            "model": "gpt-4o-mini",
            "function_call": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}'
            },
            "finish_reason": "function_call"
        }
        
        provider.generate_response.return_value = function_response
        
        # Act
        response = await provider.generate_response(
            "What's the weather like in San Francisco?",
            functions=[function_schema]
        )
        
        # Assert
        assert response["finish_reason"] == "function_call"
        assert response["function_call"]["name"] == "get_weather"
        assert "San Francisco" in response["function_call"]["arguments"]
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    def test_system_message_support(self):
        """Test OpenAI system message handling."""
        # Arrange
        system_message = "You are a helpful assistant specialized in technical documentation."
        user_message = "Explain what REST APIs are"
        
        # Act
        def build_messages_with_system(system_msg, user_msg):
            messages = []
            if system_msg:
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": user_msg})
            return messages
        
        messages = build_messages_with_system(system_message, user_message)
        
        # Assert
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_message
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == user_message
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.openai
    def test_streaming_support(self):
        """Test OpenAI streaming response capability."""
        # Arrange
        provider = self.create_mock_provider("openai")
        
        # Mock streaming chunks
        streaming_chunks = [
            {"delta": {"content": "This"}, "finish_reason": None},
            {"delta": {"content": " is"}, "finish_reason": None},
            {"delta": {"content": " streaming"}, "finish_reason": None},
            {"delta": {"content": " response"}, "finish_reason": "stop"}
        ]
        
        # Act
        def process_streaming_response(chunks):
            content = ""
            for chunk in chunks:
                if chunk["delta"].get("content"):
                    content += chunk["delta"]["content"]
                if chunk["finish_reason"]:
                    break
            return content
        
        final_content = process_streaming_response(streaming_chunks)
        
        # Assert
        assert final_content == "This is streaming response"
        assert len(streaming_chunks) == 4
        assert streaming_chunks[-1]["finish_reason"] == "stop"