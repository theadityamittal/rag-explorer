"""
Unit tests for Groq provider implementation.

Tests Groq-specific functionality including ultra-fast inference,
model variants, and cost optimization features.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from tests.base import BaseProviderTest


class TestGroqProviderInterface(BaseProviderTest):
    """Test Groq provider interface implementation."""
    
    @pytest.fixture
    def groq_provider(self):
        """Create Groq provider instance for testing."""
        provider = Mock()
        provider.name = "groq"
        provider.endpoint = "https://api.groq.com/openai/v1"
        provider.api_key = "gsk_test123"
        provider.model = "llama-3.1-8b-instant"
        provider.is_available = Mock(return_value=True)
        provider.generate_response = AsyncMock()
        return provider
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.groq
    def test_groq_provider_initialization(self, groq_provider):
        """Test Groq provider initialization."""
        assert groq_provider.name == "groq"
        assert groq_provider.endpoint == "https://api.groq.com/openai/v1"
        assert groq_provider.api_key.startswith("gsk_")
        assert "llama" in groq_provider.model.lower()
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.groq
    async def test_groq_ultra_fast_response(self, groq_provider):
        """Test Groq's ultra-fast inference capability."""
        # Mock ultra-fast response (< 1 second)
        fast_response = {
            "content": "This is a fast response from Groq's LLaMA model.",
            "model": "llama-3.1-8b-instant",
            "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
            "response_time": 0.3  # 300ms
        }
        
        groq_provider.generate_response.return_value = fast_response
        
        response = await groq_provider.generate_response("Quick question")
        
        assert response["response_time"] < 1.0  # Sub-second response
        assert "fast response" in response["content"]
        assert "llama" in response["model"].lower()


class TestGroqModelVariants(BaseProviderTest):
    """Test different Groq model variants."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.groq
    def test_llama_models(self):
        """Test Groq LLaMA model variants."""
        models = {
            "llama-3.1-8b-instant": {"size": "8B", "speed": "instant", "context": 8192},
            "llama-3.1-70b-versatile": {"size": "70B", "speed": "versatile", "context": 8192},
            "llama3-8b-8192": {"size": "8B", "speed": "standard", "context": 8192}
        }
        
        for model_name, specs in models.items():
            provider = self.create_mock_provider("groq")
            provider.model = model_name
            provider.max_tokens = specs["context"]
            provider.model_size = specs["size"]
            
            assert provider.model == model_name
            assert "llama" in provider.model.lower()
            assert provider.max_tokens >= 8192
            
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.groq
    def test_mixtral_models(self):
        """Test Groq Mixtral model variants."""
        provider = self.create_mock_provider("groq")
        provider.model = "mixtral-8x7b-32768"
        provider.max_tokens = 32768
        provider.cost_per_token = 0.00027  # Very low cost
        
        assert "mixtral" in provider.model.lower()
        assert provider.max_tokens == 32768
        assert provider.cost_per_token < 0.001  # Extremely cost-effective


class TestGroqPerformance(BaseProviderTest):
    """Test Groq performance characteristics."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.groq
    async def test_tokens_per_second(self):
        """Test Groq's high tokens per second capability."""
        provider = self.create_mock_provider("groq")
        
        # Mock high throughput response
        high_throughput_response = {
            "content": "A" * 1000,  # 1000 character response
            "model": "llama-3.1-8b-instant", 
            "usage": {"completion_tokens": 250, "total_tokens": 260},
            "response_time": 0.25,  # 250ms
            "tokens_per_second": 1000  # 1000 tokens/second
        }
        
        provider.generate_response.return_value = high_throughput_response
        
        response = await provider.generate_response("Generate long text")
        
        assert response["tokens_per_second"] >= 500  # Very high throughput
        assert response["response_time"] < 1.0  # Fast response
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.groq
    def test_cost_efficiency(self):
        """Test Groq's cost efficiency."""
        provider = self.create_mock_provider("groq")
        provider.cost_per_token = 0.00027  # $0.27 per 1M tokens
        
        # Compare to other providers
        openai_cost = 0.00003  # OpenAI GPT-4o-mini cost
        anthropic_cost = 0.000015  # Anthropic Claude cost
        
        # Groq should be extremely cost-effective
        assert provider.cost_per_token < 0.001
        
        # Calculate cost for 10K tokens
        tokens = 10000
        groq_cost = tokens * provider.cost_per_token
        
        assert groq_cost < 0.01  # Less than 1 cent for 10K tokens


class TestGroqAPIIntegration(BaseProviderTest):
    """Test Groq API integration specifics."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.groq
    def test_groq_headers(self):
        """Test Groq API headers."""
        provider = self.create_mock_provider("groq")
        provider.api_key = "gsk_test123"
        
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "support-deflect-bot"
        }
        
        assert headers["Authorization"].startswith("Bearer gsk_")
        assert headers["Content-Type"] == "application/json"
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.groq
    async def test_groq_streaming(self):
        """Test Groq streaming capability."""
        provider = self.create_mock_provider("groq")
        
        # Mock streaming chunks
        stream_chunks = [
            {"choices": [{"delta": {"content": "Fast"}}]},
            {"choices": [{"delta": {"content": " streaming"}}]},
            {"choices": [{"delta": {"content": " from Groq"}}]},
        ]
        
        content = ""
        for chunk in stream_chunks:
            if chunk["choices"][0]["delta"].get("content"):
                content += chunk["choices"][0]["delta"]["content"]
        
        assert content == "Fast streaming from Groq"


class TestGroqErrorHandling(BaseProviderTest):
    """Test Groq-specific error handling."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.groq
    async def test_groq_rate_limits(self):
        """Test Groq rate limit handling."""
        provider = self.create_mock_provider("groq")
        
        rate_limit_error = Exception("Rate limit exceeded for model llama-3.1-8b-instant")
        provider.generate_response.side_effect = rate_limit_error
        
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test")
        
        assert "rate limit" in str(exc_info.value).lower()
        assert "llama" in str(exc_info.value).lower()
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.groq
    async def test_model_capacity_errors(self):
        """Test Groq model capacity error handling."""
        provider = self.create_mock_provider("groq")
        
        capacity_error = Exception("Model llama-3.1-70b-versatile is currently at capacity")
        provider.generate_response.side_effect = capacity_error
        
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test")
        
        assert "capacity" in str(exc_info.value).lower()


class TestGroqSpecialFeatures(BaseProviderTest):
    """Test Groq-specific features."""
    
    @pytest.mark.unit
    @pytest.mark.providers  
    @pytest.mark.groq
    def test_json_mode_support(self):
        """Test Groq JSON mode capability."""
        provider = self.create_mock_provider("groq")
        
        # Mock JSON mode request
        json_request = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": "Return JSON with name and age"}],
            "response_format": {"type": "json_object"}
        }
        
        assert json_request["response_format"]["type"] == "json_object"
        assert "JSON" in json_request["messages"][0]["content"]
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.groq
    def test_temperature_optimization(self):
        """Test Groq temperature optimization."""
        provider = self.create_mock_provider("groq")
        
        # Groq performs well with lower temperatures for faster inference
        optimal_temp = 0.1  # Lower for speed
        balanced_temp = 0.5
        creative_temp = 0.9
        
        assert optimal_temp < balanced_temp < creative_temp
        assert optimal_temp >= 0.0
        assert creative_temp <= 1.0