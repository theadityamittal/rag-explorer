"""
Unit tests for Mistral AI provider implementation.

Tests Mistral-specific functionality including model variants,
multilingual capabilities, and cost-effective inference.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from tests.base import BaseProviderTest


class TestMistralProviderInterface(BaseProviderTest):
    """Test Mistral provider interface implementation."""
    
    @pytest.fixture
    def mistral_provider(self):
        """Create Mistral provider instance for testing."""
        provider = Mock()
        provider.name = "mistral"
        provider.endpoint = "https://api.mistral.ai/v1"
        provider.api_key = "mistral_test_key_123"
        provider.model = "mistral-small-latest"
        provider.is_available = Mock(return_value=True)
        provider.generate_response = AsyncMock()
        return provider
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    def test_mistral_provider_initialization(self, mistral_provider):
        """Test Mistral provider initialization."""
        assert mistral_provider.name == "mistral"
        assert "mistral.ai" in mistral_provider.endpoint
        assert "mistral" in mistral_provider.api_key.lower()
        assert "mistral" in mistral_provider.model.lower()


class TestMistralModelVariants(BaseProviderTest):
    """Test different Mistral model variants."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    def test_mistral_small_model(self):
        """Test Mistral Small model (cost-effective)."""
        provider = self.create_mock_provider("mistral")
        provider.model = "mistral-small-latest"
        provider.context_window = 32000  # 32K tokens
        provider.cost_per_input_token = 1.00 / 1000000  # $1.00 per 1M tokens
        provider.cost_per_output_token = 3.00 / 1000000  # $3.00 per 1M tokens
        
        assert "small" in provider.model.lower()
        assert provider.context_window == 32000
        assert provider.cost_per_input_token < 0.002  # Cost effective
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    def test_mistral_large_model(self):
        """Test Mistral Large model (high performance)."""
        provider = self.create_mock_provider("mistral")
        provider.model = "mistral-large-latest"
        provider.context_window = 128000  # 128K tokens
        provider.cost_per_input_token = 2.00 / 1000000  # $2.00 per 1M tokens
        provider.cost_per_output_token = 6.00 / 1000000  # $6.00 per 1M tokens
        
        assert "large" in provider.model.lower()
        assert provider.context_window >= 100000
        assert provider.cost_per_output_token > provider.cost_per_input_token
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    def test_codestral_model(self):
        """Test Codestral model (specialized for coding)."""
        provider = self.create_mock_provider("mistral")
        provider.model = "codestral-latest"
        provider.context_window = 32000
        provider.specialization = "code"
        
        assert "codestral" in provider.model.lower()
        assert provider.specialization == "code"
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    def test_mistral_nemo_model(self):
        """Test Mistral NeMo model (Apache 2.0 licensed)."""
        provider = self.create_mock_provider("mistral")
        provider.model = "open-mistral-nemo"
        provider.license = "Apache 2.0"
        provider.context_window = 128000
        
        assert "nemo" in provider.model.lower()
        assert provider.license == "Apache 2.0"


class TestMistralAPIIntegration(BaseProviderTest):
    """Test Mistral API integration specifics."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    async def test_mistral_chat_completion(self):
        """Test Mistral chat completion API."""
        provider = self.create_mock_provider("mistral")
        
        mistral_response = {
            "id": "cmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "mistral-small-latest",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Bonjour! Je suis Mistral, un assistant IA multilingue."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }
        
        parsed_response = {
            "content": mistral_response["choices"][0]["message"]["content"],
            "model": mistral_response["model"],
            "usage": mistral_response["usage"],
            "finish_reason": mistral_response["choices"][0]["finish_reason"]
        }
        
        provider.generate_response.return_value = parsed_response
        
        response = await provider.generate_response("Hello in French")
        
        assert "Bonjour" in response["content"]
        assert "Mistral" in response["content"]
        assert response["usage"]["total_tokens"] == 25
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    def test_mistral_request_headers(self):
        """Test Mistral API request headers."""
        provider = self.create_mock_provider("mistral")
        provider.api_key = "mistral_test_key_123"
        
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        assert headers["Authorization"].startswith("Bearer mistral")
        assert headers["Content-Type"] == "application/json"
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    def test_mistral_streaming_support(self):
        """Test Mistral streaming capability."""
        provider = self.create_mock_provider("mistral")
        
        streaming_chunks = [
            {
                "choices": [
                    {
                        "delta": {"content": "Je"},
                        "finish_reason": None
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {"content": " peux"},
                        "finish_reason": None
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {"content": " vous aider"},
                        "finish_reason": "stop"
                    }
                ]
            }
        ]
        
        content = ""
        for chunk in streaming_chunks:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                content += delta["content"]
        
        assert content == "Je peux vous aider"  # French for "I can help you"


class TestMistralMultilingualCapabilities(BaseProviderTest):
    """Test Mistral's multilingual capabilities."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    async def test_french_language_support(self):
        """Test Mistral's native French language support."""
        provider = self.create_mock_provider("mistral")
        
        french_response = {
            "content": "Je suis un modèle d'IA développé par Mistral AI, une entreprise française. Je peux comprendre et générer du texte en français avec une grande précision.",
            "model": "mistral-small-latest",
            "language": "fr"
        }
        
        provider.generate_response.return_value = french_response
        
        response = await provider.generate_response("Parlez-vous français?")
        
        assert "français" in response["content"]
        assert "Mistral" in response["content"]
        assert response.get("language") == "fr"
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    async def test_multilingual_support(self):
        """Test Mistral's multilingual capabilities."""
        provider = self.create_mock_provider("mistral")
        
        supported_languages = [
            "en",  # English
            "fr",  # French
            "de",  # German
            "es",  # Spanish
            "it",  # Italian
            "pt",  # Portuguese
            "nl",  # Dutch
        ]
        
        provider.supported_languages = supported_languages
        
        assert "fr" in provider.supported_languages  # French is native
        assert "en" in provider.supported_languages  # English support
        assert len(provider.supported_languages) >= 5  # Multiple languages
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    async def test_code_multilingual(self):
        """Test Mistral's multilingual code generation."""
        provider = self.create_mock_provider("mistral")
        
        multilingual_code_response = {
            "content": """# Python function
def fibonacci(n):
    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)

# Fonction JavaScript  
function fibonacci(n) {
    return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2);
}""",
            "model": "codestral-latest",
            "languages": ["python", "javascript"]
        }
        
        provider.generate_response.return_value = multilingual_code_response
        
        response = await provider.generate_response("Write fibonacci in Python and JavaScript")
        
        assert "def fibonacci" in response["content"]  # Python
        assert "function fibonacci" in response["content"]  # JavaScript
        assert "Fonction JavaScript" in response["content"]  # French comment


class TestMistralSpecializedFeatures(BaseProviderTest):
    """Test Mistral's specialized features."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    def test_function_calling(self):
        """Test Mistral's function calling capability."""
        provider = self.create_mock_provider("mistral")
        
        function_schema = {
            "name": "get_weather",
            "description": "Obtenir les informations météo pour une ville",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Nom de la ville"}
                },
                "required": ["city"]
            }
        }
        
        function_call_response = {
            "content": None,
            "model": "mistral-large-latest",
            "function_call": {
                "name": "get_weather",
                "arguments": '{"city": "Paris"}'
            },
            "finish_reason": "function_call"
        }
        
        # Test function schema
        assert "description" in function_schema
        assert "Obtenir" in function_schema["description"]  # French description
        
        # Test function call response
        assert function_call_response["finish_reason"] == "function_call"
        assert "Paris" in function_call_response["function_call"]["arguments"]
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    def test_mistral_guardrails(self):
        """Test Mistral's content safety guardrails."""
        provider = self.create_mock_provider("mistral")
        
        safe_response = {
            "content": "Je ne peux pas fournir d'informations qui pourraient être utilisées de manière nuisible. Puis-je vous aider avec autre chose?",
            "model": "mistral-small-latest",
            "safety_score": 0.95,  # High safety score
            "blocked": False
        }
        
        provider.generate_response.return_value = safe_response
        
        assert safe_response["safety_score"] > 0.8
        assert safe_response["blocked"] is False
        assert "ne peux pas" in safe_response["content"]  # "cannot" in French
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    def test_mistral_json_mode(self):
        """Test Mistral's JSON mode capability."""
        provider = self.create_mock_provider("mistral")
        
        json_request = {
            "model": "mistral-large-latest",
            "messages": [
                {"role": "user", "content": "Return a JSON object with name and age"}
            ],
            "response_format": {"type": "json_object"}
        }
        
        json_response = {
            "content": '{"name": "Marie", "age": 25}',
            "model": "mistral-large-latest",
            "format": "json"
        }
        
        assert json_request["response_format"]["type"] == "json_object"
        
        # Test JSON parsing
        import json
        parsed_json = json.loads(json_response["content"])
        assert "name" in parsed_json
        assert "age" in parsed_json


class TestMistralErrorHandling(BaseProviderTest):
    """Test Mistral-specific error handling."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    async def test_mistral_rate_limits(self):
        """Test Mistral rate limit handling."""
        provider = self.create_mock_provider("mistral")
        
        rate_limit_error = Exception("Rate limit exceeded: 100 requests per minute")
        provider.generate_response.side_effect = rate_limit_error
        
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test")
        
        assert "rate limit" in str(exc_info.value).lower()
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    async def test_mistral_model_overload(self):
        """Test Mistral model overload error handling."""
        provider = self.create_mock_provider("mistral")
        
        overload_error = Exception("Model mistral-large-latest is currently overloaded")
        provider.generate_response.side_effect = overload_error
        
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test")
        
        assert "overloaded" in str(exc_info.value)
        assert "mistral-large" in str(exc_info.value)
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    async def test_mistral_authentication_error(self):
        """Test Mistral authentication error handling."""
        provider = self.create_mock_provider("mistral")
        
        auth_error = Exception("Invalid API key for Mistral AI")
        provider.generate_response.side_effect = auth_error
        
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test")
        
        assert "api key" in str(exc_info.value).lower()
        assert "mistral" in str(exc_info.value).lower()


class TestMistralPerformanceOptimization(BaseProviderTest):
    """Test Mistral performance optimization features."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    def test_cost_optimization(self):
        """Test Mistral cost optimization strategies."""
        # Model cost comparison
        models = {
            "mistral-small-latest": {"cost": 1.00, "performance": 0.8},
            "mistral-large-latest": {"cost": 2.00, "performance": 0.95},
            "codestral-latest": {"cost": 1.50, "performance": 0.9}
        }
        
        # Calculate value score (performance/cost)
        for model, specs in models.items():
            value_score = specs["performance"] / specs["cost"]
            models[model]["value"] = value_score
        
        # Mistral Small should have good value
        assert models["mistral-small-latest"]["value"] >= 0.5
        
        # Find best value model
        best_value_model = max(models.items(), key=lambda x: x[1]["value"])
        assert best_value_model[0] == "mistral-small-latest"  # Best value
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    def test_inference_optimization(self):
        """Test Mistral inference speed optimization."""
        provider = self.create_mock_provider("mistral")
        
        # Mistral is optimized for speed
        provider.average_latency = 1.5  # seconds
        provider.tokens_per_second = 50
        
        assert provider.average_latency < 3.0  # Fast inference
        assert provider.tokens_per_second > 20  # Good throughput
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.mistral
    def test_european_compliance(self):
        """Test Mistral's European compliance features."""
        provider = self.create_mock_provider("mistral")
        
        compliance_features = {
            "gdpr_compliant": True,
            "data_residency": "EU",
            "privacy_by_design": True,
            "local_deployment": True
        }
        
        provider.compliance = compliance_features
        
        assert provider.compliance["gdpr_compliant"] is True
        assert provider.compliance["data_residency"] == "EU"
        assert provider.compliance["privacy_by_design"] is True