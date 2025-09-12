"""
Unit tests for Google Gemini provider implementation.

Tests Google Gemini-specific functionality including multimodal capabilities,
safety settings, and integration with Google AI services.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from tests.base import BaseProviderTest


class TestGeminiProviderInterface(BaseProviderTest):
    """Test Google Gemini provider interface implementation."""
    
    @pytest.fixture
    def gemini_provider(self):
        """Create Gemini provider instance for testing."""
        provider = Mock()
        provider.name = "google_gemini"
        provider.endpoint = "https://generativelanguage.googleapis.com/v1"
        provider.api_key = "AIzaSyTest123"
        provider.model = "gemini-1.5-flash"
        provider.is_available = Mock(return_value=True)
        provider.generate_response = AsyncMock()
        provider.generate_embeddings = AsyncMock()
        return provider
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    def test_gemini_provider_initialization(self, gemini_provider):
        """Test Gemini provider initialization."""
        assert gemini_provider.name == "google_gemini"
        assert "googleapis.com" in gemini_provider.endpoint
        assert gemini_provider.api_key.startswith("AIza")
        assert "gemini" in gemini_provider.model.lower()


class TestGeminiModelVariants(BaseProviderTest):
    """Test different Gemini model variants."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    def test_gemini_flash_model(self):
        """Test Gemini Flash model (fast, cost-effective)."""
        provider = self.create_mock_provider("google_gemini")
        provider.model = "gemini-1.5-flash"
        provider.context_window = 1000000  # 1M tokens
        provider.cost_per_input_token = 0.075 / 1000000  # $0.075 per 1M tokens
        provider.cost_per_output_token = 0.30 / 1000000  # $0.30 per 1M tokens
        
        assert provider.model == "gemini-1.5-flash"
        assert provider.context_window == 1000000  # Huge context window
        assert provider.cost_per_input_token < 0.001  # Very cost effective
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    def test_gemini_pro_model(self):
        """Test Gemini Pro model (advanced reasoning)."""
        provider = self.create_mock_provider("google_gemini")
        provider.model = "gemini-1.5-pro"
        provider.context_window = 2000000  # 2M tokens
        provider.cost_per_input_token = 1.25 / 1000000  # $1.25 per 1M tokens
        provider.cost_per_output_token = 5.00 / 1000000  # $5.00 per 1M tokens
        
        assert provider.model == "gemini-1.5-pro"
        assert provider.context_window == 2000000
        assert provider.cost_per_input_token > provider.cost_per_output_token / 5
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    def test_gemini_embedding_model(self):
        """Test Gemini embedding model."""
        provider = self.create_mock_provider("google_gemini")
        provider.embedding_model = "text-embedding-004"
        provider.embedding_dimensions = 768
        provider.embedding_cost_per_token = 0.00001 / 1000  # Very cheap
        
        assert "embedding" in provider.embedding_model.lower()
        assert provider.embedding_dimensions == 768
        assert provider.embedding_cost_per_token < 0.000001  # Extremely cheap


class TestGeminiMultimodalCapabilities(BaseProviderTest):
    """Test Gemini's multimodal capabilities."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    async def test_text_and_image_processing(self):
        """Test Gemini's text + image processing capability."""
        provider = self.create_mock_provider("google_gemini")
        
        multimodal_response = {
            "content": "I can see an image of a cat sitting on a table. The image shows a tabby cat with green eyes.",
            "model": "gemini-1.5-flash",
            "usage": {"prompt_tokens": 150, "completion_tokens": 25, "total_tokens": 175}
        }
        
        provider.generate_response.return_value = multimodal_response
        
        # Mock multimodal input
        multimodal_input = {
            "text": "What do you see in this image?",
            "image_data": "base64_encoded_image_data"
        }
        
        response = await provider.generate_response(multimodal_input)
        
        assert "image" in response["content"]
        assert "cat" in response["content"]
        assert response["usage"]["prompt_tokens"] > 100  # Image adds tokens
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    async def test_document_processing(self):
        """Test Gemini's document processing capability."""
        provider = self.create_mock_provider("google_gemini")
        
        doc_response = {
            "content": "This document discusses machine learning fundamentals, covering supervised and unsupervised learning methods.",
            "model": "gemini-1.5-pro",
            "usage": {"prompt_tokens": 5000, "completion_tokens": 50, "total_tokens": 5050}
        }
        
        provider.generate_response.return_value = doc_response
        
        document_input = {
            "text": "Summarize this document",
            "document_data": "long_document_content"
        }
        
        response = await provider.generate_response(document_input)
        
        assert "document" in response["content"]
        assert "machine learning" in response["content"]
        assert response["usage"]["prompt_tokens"] > 1000  # Document processing uses many tokens


class TestGeminiSafetyFeatures(BaseProviderTest):
    """Test Gemini's safety and content filtering features."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    def test_safety_settings_configuration(self):
        """Test Gemini safety settings configuration."""
        provider = self.create_mock_provider("google_gemini")
        
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE", 
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_HIGH_ONLY",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE"
        }
        
        provider.safety_settings = safety_settings
        
        assert "HARM_CATEGORY_HARASSMENT" in provider.safety_settings
        assert "BLOCK_MEDIUM_AND_ABOVE" in provider.safety_settings.values()
        assert len(provider.safety_settings) == 4  # All categories covered
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    async def test_content_filtering_response(self):
        """Test Gemini content filtering in responses."""
        provider = self.create_mock_provider("google_gemini")
        
        filtered_response = {
            "content": None,
            "model": "gemini-1.5-flash",
            "safety_ratings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "probability": "MEDIUM",
                    "blocked": True
                }
            ],
            "finish_reason": "SAFETY"
        }
        
        provider.generate_response.return_value = filtered_response
        
        response = await provider.generate_response("potentially harmful content")
        
        assert response["content"] is None
        assert response["finish_reason"] == "SAFETY"
        assert "safety_ratings" in response
        assert response["safety_ratings"][0]["blocked"] is True


class TestGeminiAPIIntegration(BaseProviderTest):
    """Test Gemini API integration specifics."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    def test_gemini_request_format(self):
        """Test Gemini API request format."""
        provider = self.create_mock_provider("google_gemini")
        
        request_payload = {
            "contents": [
                {
                    "parts": [
                        {"text": "What is machine learning?"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000,
                "topP": 0.95
            }
        }
        
        assert "contents" in request_payload
        assert "parts" in request_payload["contents"][0]
        assert "generationConfig" in request_payload
        assert "temperature" in request_payload["generationConfig"]
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    def test_gemini_response_parsing(self):
        """Test Gemini API response parsing."""
        raw_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Machine learning is a subset of artificial intelligence..."}
                        ]
                    },
                    "finishReason": "STOP",
                    "safetyRatings": [
                        {"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"}
                    ]
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 25,
                "totalTokenCount": 35
            }
        }
        
        # Parse response
        def parse_gemini_response(raw_response):
            candidate = raw_response["candidates"][0]
            content = candidate["content"]["parts"][0]["text"]
            usage = raw_response["usageMetadata"]
            
            return {
                "content": content,
                "model": "gemini-1.5-flash",
                "usage": {
                    "prompt_tokens": usage["promptTokenCount"],
                    "completion_tokens": usage["candidatesTokenCount"],
                    "total_tokens": usage["totalTokenCount"]
                },
                "finish_reason": candidate["finishReason"]
            }
        
        parsed = parse_gemini_response(raw_response)
        
        assert "Machine learning" in parsed["content"]
        assert parsed["usage"]["total_tokens"] == 35
        assert parsed["finish_reason"] == "STOP"


class TestGeminiEmbeddings(BaseProviderTest):
    """Test Gemini embedding capabilities."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    async def test_gemini_text_embeddings(self):
        """Test Gemini text embedding generation."""
        provider = self.create_mock_provider("google_gemini")
        
        embedding_response = {
            "embedding": {
                "values": [0.1] * 768  # 768-dimensional embedding
            }
        }
        
        provider.generate_embeddings.return_value = [embedding_response["embedding"]["values"]]
        
        embeddings = await provider.generate_embeddings(["test text for embedding"])
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768
        assert all(isinstance(x, float) for x in embeddings[0])
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    async def test_gemini_batch_embeddings(self):
        """Test Gemini batch embedding processing."""
        provider = self.create_mock_provider("google_gemini")
        
        texts = ["text 1", "text 2", "text 3"]
        batch_embeddings = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        
        provider.generate_embeddings.return_value = batch_embeddings
        
        embeddings = await provider.generate_embeddings(texts)
        
        assert len(embeddings) == 3
        assert embeddings[0][0] != embeddings[1][0]  # Different embeddings
        assert embeddings[1][0] != embeddings[2][0]


class TestGeminiErrorHandling(BaseProviderTest):
    """Test Gemini-specific error handling."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    async def test_gemini_quota_errors(self):
        """Test Gemini quota exceeded error handling."""
        provider = self.create_mock_provider("google_gemini")
        
        quota_error = Exception("Quota exceeded for requests per minute")
        provider.generate_response.side_effect = quota_error
        
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test")
        
        assert "quota" in str(exc_info.value).lower()
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    async def test_gemini_api_key_errors(self):
        """Test Gemini API key validation errors."""
        provider = self.create_mock_provider("google_gemini")
        
        api_key_error = Exception("API key not valid")
        provider.generate_response.side_effect = api_key_error
        
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test")
        
        assert "api key" in str(exc_info.value).lower()
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    async def test_gemini_safety_blocking(self):
        """Test Gemini safety system blocking content."""
        provider = self.create_mock_provider("google_gemini")
        
        safety_response = {
            "content": None,
            "model": "gemini-1.5-flash",
            "finish_reason": "SAFETY",
            "safety_ratings": [
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH", "blocked": True}
            ]
        }
        
        provider.generate_response.return_value = safety_response
        
        response = await provider.generate_response("dangerous request")
        
        assert response["finish_reason"] == "SAFETY"
        assert response["content"] is None
        assert any(rating["blocked"] for rating in response["safety_ratings"])


class TestGeminiAdvancedFeatures(BaseProviderTest):
    """Test Gemini advanced features."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    def test_long_context_processing(self):
        """Test Gemini's long context window capability."""
        provider = self.create_mock_provider("google_gemini")
        provider.model = "gemini-1.5-pro"
        provider.context_window = 2000000  # 2M tokens
        
        # Simulate very long input
        long_input = "word " * 100000  # 100K words
        estimated_tokens = len(long_input.split())
        
        assert provider.context_window > estimated_tokens
        assert provider.context_window >= 1000000  # At least 1M tokens
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.gemini
    def test_code_execution_capability(self):
        """Test Gemini's code execution feature."""
        provider = self.create_mock_provider("google_gemini")
        
        code_response = {
            "content": "The Python code calculates the factorial of 5, which equals 120.",
            "model": "gemini-1.5-pro",
            "code_execution": {
                "code": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n\nresult = factorial(5)",
                "output": "120"
            }
        }
        
        assert "code_execution" in code_response
        assert code_response["code_execution"]["output"] == "120"
        assert "factorial" in code_response["code_execution"]["code"]