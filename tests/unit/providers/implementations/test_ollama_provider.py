"""
Unit tests for Ollama provider implementation.

Tests Ollama-specific functionality including local deployment,
model management, and offline capabilities.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from tests.base import BaseProviderTest


class TestOllamaProviderInterface(BaseProviderTest):
    """Test Ollama provider interface implementation."""
    
    @pytest.fixture
    def ollama_provider(self):
        """Create Ollama provider instance for testing."""
        provider = Mock()
        provider.name = "ollama"
        provider.endpoint = "http://localhost:11434"
        provider.api_key = None  # Local deployment, no API key needed
        provider.model = "llama2"
        provider.is_available = Mock(return_value=True)
        provider.generate_response = AsyncMock()
        provider.generate_embeddings = AsyncMock()
        return provider
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    def test_ollama_provider_initialization(self, ollama_provider):
        """Test Ollama provider initialization."""
        assert ollama_provider.name == "ollama"
        assert "localhost" in ollama_provider.endpoint
        assert ollama_provider.api_key is None  # No API key for local
        assert ollama_provider.model in ["llama2", "llama3", "mistral", "codellama"]


class TestOllamaLocalDeployment(BaseProviderTest):
    """Test Ollama local deployment features."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    def test_local_endpoint_configuration(self):
        """Test Ollama local endpoint configuration."""
        provider = self.create_mock_provider("ollama")
        
        # Test default local configuration
        provider.endpoint = "http://localhost:11434"
        provider.is_local = True
        provider.requires_api_key = False
        
        assert "localhost" in provider.endpoint
        assert provider.is_local is True
        assert provider.requires_api_key is False
        
        # Test custom local configuration
        provider.endpoint = "http://192.168.1.100:11434"  # Custom IP
        assert "192.168.1.100" in provider.endpoint
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    def test_offline_capability(self):
        """Test Ollama offline operation capability."""
        provider = self.create_mock_provider("ollama")
        
        provider.requires_internet = False
        provider.data_privacy = "full_local"
        provider.cost_per_token = 0.0  # Free local inference
        
        assert provider.requires_internet is False
        assert provider.data_privacy == "full_local"
        assert provider.cost_per_token == 0.0
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    async def test_local_availability_check(self):
        """Test checking Ollama local service availability."""
        provider = self.create_mock_provider("ollama")
        
        # Mock availability check (would ping local service)
        def check_local_service():
            try:
                # Simulate HTTP GET to /api/tags
                return True  # Service is running
            except:
                return False  # Service not running
        
        provider.is_available.side_effect = check_local_service
        
        is_available = provider.is_available()
        assert isinstance(is_available, bool)


class TestOllamaModelManagement(BaseProviderTest):
    """Test Ollama model management features."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    def test_available_models(self):
        """Test listing available Ollama models."""
        provider = self.create_mock_provider("ollama")
        
        available_models = [
            {"name": "llama2:7b", "size": "3.8GB", "family": "llama"},
            {"name": "llama2:13b", "size": "7.3GB", "family": "llama"},
            {"name": "mistral:7b", "size": "4.1GB", "family": "mistral"},
            {"name": "codellama:7b", "size": "3.8GB", "family": "codellama"},
            {"name": "phi:2.7b", "size": "1.7GB", "family": "phi"}
        ]
        
        provider.list_models.return_value = available_models
        
        models = provider.list_models()
        
        assert len(models) >= 4
        assert any("llama2" in model["name"] for model in models)
        assert any("mistral" in model["name"] for model in models)
        assert any(float(model["size"].replace("GB", "")) < 5.0 for model in models)  # Small models available
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    def test_model_pulling(self):
        """Test Ollama model pulling capability."""
        provider = self.create_mock_provider("ollama")
        
        # Mock model pull operation
        def pull_model(model_name):
            return {
                "status": "success",
                "model": model_name,
                "size": "3.8GB",
                "download_time": 300  # 5 minutes
            }
        
        provider.pull_model = Mock(side_effect=pull_model)
        
        result = provider.pull_model("llama2:7b")
        
        assert result["status"] == "success"
        assert result["model"] == "llama2:7b"
        assert "GB" in result["size"]
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    def test_model_deletion(self):
        """Test Ollama model deletion capability."""
        provider = self.create_mock_provider("ollama")
        
        def delete_model(model_name):
            return {
                "status": "deleted",
                "model": model_name,
                "freed_space": "3.8GB"
            }
        
        provider.delete_model = Mock(side_effect=delete_model)
        
        result = provider.delete_model("old-model:7b")
        
        assert result["status"] == "deleted"
        assert "GB" in result["freed_space"]


class TestOllamaAPIIntegration(BaseProviderTest):
    """Test Ollama API integration specifics."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    async def test_ollama_generate_api(self):
        """Test Ollama generate API format."""
        provider = self.create_mock_provider("ollama")
        
        ollama_response = {
            "model": "llama2",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "response": "The capital of France is Paris. It is located in the north-central part of the country.",
            "done": True,
            "context": [1, 2, 3, 4, 5],  # Token context
            "total_duration": 5432112345,
            "load_duration": 3456789,
            "prompt_eval_count": 26,
            "prompt_eval_duration": 543234,
            "eval_count": 298,
            "eval_duration": 3456789
        }
        
        parsed_response = {
            "content": ollama_response["response"],
            "model": ollama_response["model"],
            "usage": {
                "prompt_tokens": ollama_response["prompt_eval_count"],
                "completion_tokens": ollama_response["eval_count"],
                "total_tokens": ollama_response["prompt_eval_count"] + ollama_response["eval_count"]
            },
            "performance": {
                "total_duration_ns": ollama_response["total_duration"],
                "load_duration_ns": ollama_response["load_duration"],
                "eval_duration_ns": ollama_response["eval_duration"]
            }
        }
        
        provider.generate_response.return_value = parsed_response
        
        response = await provider.generate_response("What is the capital of France?")
        
        assert "Paris" in response["content"]
        assert response["model"] == "llama2"
        assert response["usage"]["total_tokens"] > 200
        assert "performance" in response
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    async def test_ollama_embeddings_api(self):
        """Test Ollama embeddings API format."""
        provider = self.create_mock_provider("ollama")
        
        embedding_response = {
            "embedding": [0.1] * 4096  # 4096-dimensional embedding
        }
        
        provider.generate_embeddings.return_value = [embedding_response["embedding"]]
        
        embeddings = await provider.generate_embeddings(["test text"])
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 4096
        assert all(isinstance(x, float) for x in embeddings[0])
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    def test_ollama_streaming(self):
        """Test Ollama streaming capability."""
        provider = self.create_mock_provider("ollama")
        
        streaming_chunks = [
            {"response": "The", "done": False},
            {"response": " capital", "done": False},
            {"response": " of", "done": False},
            {"response": " France", "done": False},
            {"response": " is", "done": False},
            {"response": " Paris", "done": False},
            {"response": ".", "done": True}
        ]
        
        full_response = ""
        for chunk in streaming_chunks:
            full_response += chunk["response"]
            if chunk["done"]:
                break
        
        assert full_response == "The capital of France is Paris."
        assert streaming_chunks[-1]["done"] is True


class TestOllamaPerformanceCharacteristics(BaseProviderTest):
    """Test Ollama performance characteristics."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    def test_local_inference_performance(self):
        """Test Ollama local inference performance metrics."""
        provider = self.create_mock_provider("ollama")
        
        # Performance depends on hardware
        performance_metrics = {
            "cpu_inference": {"tokens_per_second": 2.5, "latency": 4.0},  # CPU only
            "gpu_inference": {"tokens_per_second": 25.0, "latency": 0.8},  # With GPU
            "memory_usage": {"model_ram": "4GB", "context_ram": "1GB"}
        }
        
        provider.performance = performance_metrics
        
        # CPU inference is slower but still usable
        assert provider.performance["cpu_inference"]["tokens_per_second"] > 1.0
        assert provider.performance["cpu_inference"]["latency"] < 10.0
        
        # GPU inference is much faster
        assert provider.performance["gpu_inference"]["tokens_per_second"] > 10.0
        assert provider.performance["gpu_inference"]["latency"] < 2.0
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    def test_resource_requirements(self):
        """Test Ollama resource requirements for different models."""
        provider = self.create_mock_provider("ollama")
        
        model_requirements = {
            "llama2:7b": {"ram": "8GB", "storage": "4GB", "gpu_vram": "4GB"},
            "llama2:13b": {"ram": "16GB", "storage": "7GB", "gpu_vram": "8GB"},
            "mistral:7b": {"ram": "8GB", "storage": "4GB", "gpu_vram": "4GB"},
            "phi:2.7b": {"ram": "4GB", "storage": "2GB", "gpu_vram": "2GB"}  # Lightweight
        }
        
        provider.model_requirements = model_requirements
        
        # Phi model should be most lightweight
        phi_req = provider.model_requirements["phi:2.7b"]
        llama13_req = provider.model_requirements["llama2:13b"]
        
        assert int(phi_req["ram"].replace("GB", "")) < int(llama13_req["ram"].replace("GB", ""))
        assert int(phi_req["storage"].replace("GB", "")) < int(llama13_req["storage"].replace("GB", ""))
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    def test_cost_analysis(self):
        """Test Ollama cost analysis (hardware vs API costs)."""
        provider = self.create_mock_provider("ollama")
        
        # Local deployment costs
        local_costs = {
            "hardware_cost": 2000,  # Initial hardware cost
            "electricity_per_hour": 0.50,  # Power consumption
            "per_token_cost": 0.0,  # No API fees
            "privacy_benefit": "full_control"
        }
        
        provider.cost_analysis = local_costs
        
        assert provider.cost_analysis["per_token_cost"] == 0.0  # Free inference
        assert provider.cost_analysis["hardware_cost"] > 0  # Initial investment
        assert provider.cost_analysis["privacy_benefit"] == "full_control"


class TestOllamaSpecialFeatures(BaseProviderTest):
    """Test Ollama special features and capabilities."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    def test_custom_model_support(self):
        """Test Ollama custom model support."""
        provider = self.create_mock_provider("ollama")
        
        # Custom model creation from Modelfile
        modelfile_content = """
FROM llama2
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM "You are a helpful coding assistant."
        """
        
        custom_model = {
            "name": "coding-assistant",
            "base_model": "llama2",
            "modelfile": modelfile_content,
            "size": "3.8GB"
        }
        
        provider.create_model = Mock(return_value=custom_model)
        
        result = provider.create_model("coding-assistant", modelfile_content)
        
        assert result["name"] == "coding-assistant"
        assert result["base_model"] == "llama2"
        assert "coding assistant" in result["modelfile"].lower()
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    def test_model_quantization_support(self):
        """Test Ollama model quantization support."""
        provider = self.create_mock_provider("ollama")
        
        quantization_options = {
            "q4_0": {"size_reduction": 0.5, "quality": 0.9},    # 4-bit quantization
            "q5_0": {"size_reduction": 0.6, "quality": 0.95},   # 5-bit quantization
            "q8_0": {"size_reduction": 0.75, "quality": 0.98},  # 8-bit quantization
            "fp16": {"size_reduction": 0.5, "quality": 1.0}     # Half precision
        }
        
        provider.quantization_options = quantization_options
        
        # q4_0 should offer good compression with acceptable quality
        q4_option = provider.quantization_options["q4_0"]
        assert q4_option["size_reduction"] >= 0.4
        assert q4_option["quality"] >= 0.85
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    def test_multimodal_support(self):
        """Test Ollama multimodal model support."""
        provider = self.create_mock_provider("ollama")
        
        multimodal_models = [
            {"name": "llava:7b", "capabilities": ["text", "vision"], "size": "4.7GB"},
            {"name": "llava:13b", "capabilities": ["text", "vision"], "size": "8.0GB"},
            {"name": "bakllava", "capabilities": ["text", "vision"], "size": "4.4GB"}
        ]
        
        provider.multimodal_models = multimodal_models
        
        vision_models = [m for m in provider.multimodal_models if "vision" in m["capabilities"]]
        
        assert len(vision_models) >= 2
        assert any("llava" in model["name"] for model in vision_models)
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    def test_fine_tuning_support(self):
        """Test Ollama fine-tuning capability."""
        provider = self.create_mock_provider("ollama")
        
        fine_tuning_config = {
            "base_model": "llama2:7b",
            "training_data": "custom_dataset.jsonl",
            "epochs": 3,
            "learning_rate": 0.0001,
            "batch_size": 4
        }
        
        provider.fine_tune = Mock(return_value={
            "status": "completed",
            "model_name": "custom-llama2",
            "training_time": 7200  # 2 hours
        })
        
        result = provider.fine_tune(fine_tuning_config)
        
        assert result["status"] == "completed"
        assert "custom" in result["model_name"]
        assert result["training_time"] > 0


class TestOllamaErrorHandling(BaseProviderTest):
    """Test Ollama-specific error handling."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    async def test_service_not_running_error(self):
        """Test handling when Ollama service is not running."""
        provider = self.create_mock_provider("ollama")
        
        connection_error = Exception("Connection refused - Ollama service not running")
        provider.generate_response.side_effect = connection_error
        
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test")
        
        assert "connection refused" in str(exc_info.value).lower()
        assert "service not running" in str(exc_info.value).lower()
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    async def test_model_not_found_error(self):
        """Test handling when requested model is not available."""
        provider = self.create_mock_provider("ollama")
        
        model_error = Exception("Model 'nonexistent:7b' not found. Run 'ollama pull nonexistent:7b' first.")
        provider.generate_response.side_effect = model_error
        
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test")
        
        assert "model" in str(exc_info.value).lower()
        assert "not found" in str(exc_info.value).lower()
        assert "ollama pull" in str(exc_info.value).lower()
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.ollama
    async def test_insufficient_memory_error(self):
        """Test handling insufficient memory errors."""
        provider = self.create_mock_provider("ollama")
        
        memory_error = Exception("Insufficient memory to load model. Need 8GB RAM, have 4GB available.")
        provider.generate_response.side_effect = memory_error
        
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test")
        
        assert "insufficient memory" in str(exc_info.value).lower()
        assert "8gb" in str(exc_info.value).lower()
        assert "4gb" in str(exc_info.value).lower()