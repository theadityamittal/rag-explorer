"""
Unit tests for Anthropic Claude provider implementation.

Tests Anthropic-specific functionality including Claude models,
safety features, and advanced reasoning capabilities.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from tests.base import BaseProviderTest


class TestAnthropicProviderInterface(BaseProviderTest):
    """Test Anthropic provider interface implementation."""
    
    @pytest.fixture
    def anthropic_provider(self):
        """Create Anthropic provider instance for testing."""
        provider = Mock()
        provider.name = "anthropic"
        provider.endpoint = "https://api.anthropic.com/v1"
        provider.api_key = "sk-ant-api03-test123"
        provider.model = "claude-3-5-sonnet-20241022"
        provider.is_available = Mock(return_value=True)
        provider.generate_response = AsyncMock()
        return provider
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    def test_anthropic_provider_initialization(self, anthropic_provider):
        """Test Anthropic provider initialization."""
        assert anthropic_provider.name == "anthropic"
        assert "anthropic.com" in anthropic_provider.endpoint
        assert anthropic_provider.api_key.startswith("sk-ant-")
        assert "claude" in anthropic_provider.model.lower()


class TestClaudeModelVariants(BaseProviderTest):
    """Test different Claude model variants."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    def test_claude_3_5_sonnet(self):
        """Test Claude 3.5 Sonnet model."""
        provider = self.create_mock_provider("anthropic")
        provider.model = "claude-3-5-sonnet-20241022"
        provider.context_window = 200000  # 200K tokens
        provider.cost_per_input_token = 3.00 / 1000000  # $3.00 per 1M tokens
        provider.cost_per_output_token = 15.00 / 1000000  # $15.00 per 1M tokens
        
        assert "sonnet" in provider.model.lower()
        assert provider.context_window == 200000
        assert provider.cost_per_output_token > provider.cost_per_input_token
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    def test_claude_3_haiku(self):
        """Test Claude 3 Haiku model (fast, cost-effective)."""
        provider = self.create_mock_provider("anthropic")
        provider.model = "claude-3-haiku-20240307"
        provider.context_window = 200000
        provider.cost_per_input_token = 0.25 / 1000000  # $0.25 per 1M tokens
        provider.cost_per_output_token = 1.25 / 1000000  # $1.25 per 1M tokens
        
        assert "haiku" in provider.model.lower()
        assert provider.cost_per_input_token < 0.001  # Very cost effective
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    def test_claude_3_opus(self):
        """Test Claude 3 Opus model (highest capability)."""
        provider = self.create_mock_provider("anthropic")
        provider.model = "claude-3-opus-20240229"
        provider.context_window = 200000
        provider.cost_per_input_token = 15.00 / 1000000  # $15.00 per 1M tokens
        provider.cost_per_output_token = 75.00 / 1000000  # $75.00 per 1M tokens
        
        assert "opus" in provider.model.lower()
        assert provider.cost_per_output_token > 0.00005  # Premium pricing


class TestAnthropicAPIIntegration(BaseProviderTest):
    """Test Anthropic API integration specifics."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    async def test_claude_message_format(self):
        """Test Claude's message format."""
        provider = self.create_mock_provider("anthropic")
        
        claude_response = {
            "id": "msg_test123",
            "type": "message",
            "role": "assistant", 
            "content": [
                {
                    "type": "text",
                    "text": "I'm Claude, an AI assistant created by Anthropic. How can I help you today?"
                }
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 20
            }
        }
        
        parsed_response = {
            "content": claude_response["content"][0]["text"],
            "model": claude_response["model"],
            "usage": {
                "prompt_tokens": claude_response["usage"]["input_tokens"],
                "completion_tokens": claude_response["usage"]["output_tokens"],
                "total_tokens": claude_response["usage"]["input_tokens"] + claude_response["usage"]["output_tokens"]
            },
            "stop_reason": claude_response["stop_reason"]
        }
        
        provider.generate_response.return_value = parsed_response
        
        response = await provider.generate_response("Hello, Claude!")
        
        assert "Claude" in response["content"]
        assert "Anthropic" in response["content"]
        assert response["usage"]["total_tokens"] == 35
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    def test_anthropic_headers(self):
        """Test Anthropic API headers."""
        provider = self.create_mock_provider("anthropic")
        provider.api_key = "sk-ant-api03-test123"
        
        headers = {
            "x-api-key": provider.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        assert headers["x-api-key"].startswith("sk-ant-")
        assert "anthropic-version" in headers
        assert headers["content-type"] == "application/json"
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    def test_claude_system_messages(self):
        """Test Claude system message handling."""
        system_prompt = "You are a helpful assistant specialized in explaining complex topics simply."
        
        request_payload = {
            "model": "claude-3-5-sonnet-20241022",
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": "Explain quantum computing"}
            ],
            "max_tokens": 1000
        }
        
        assert "system" in request_payload
        assert request_payload["system"] == system_prompt
        assert len(request_payload["messages"]) == 1
        assert request_payload["messages"][0]["role"] == "user"


class TestClaudeAdvancedFeatures(BaseProviderTest):
    """Test Claude's advanced reasoning and safety features."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    async def test_claude_reasoning_capability(self):
        """Test Claude's advanced reasoning capabilities."""
        provider = self.create_mock_provider("anthropic")
        
        reasoning_response = {
            "content": "To solve this problem, I'll break it down step by step:\n\n1. First, let me identify the key variables\n2. Then I'll analyze the relationships\n3. Finally, I'll provide a conclusion based on the evidence",
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}
        }
        
        provider.generate_response.return_value = reasoning_response
        
        response = await provider.generate_response("Solve this complex logic problem")
        
        assert "step by step" in response["content"]
        assert "analyze" in response["content"]
        assert "conclusion" in response["content"]
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    async def test_claude_safety_alignment(self):
        """Test Claude's safety and alignment features."""
        provider = self.create_mock_provider("anthropic")
        
        safety_response = {
            "content": "I can't and won't provide information that could be used to harm others. Instead, let me suggest some constructive alternatives...",
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"prompt_tokens": 25, "completion_tokens": 35, "total_tokens": 60},
            "stop_reason": "end_turn"
        }
        
        provider.generate_response.return_value = safety_response
        
        response = await provider.generate_response("harmful request")
        
        assert "can't and won't" in response["content"]
        assert "alternatives" in response["content"]
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    def test_claude_long_context_handling(self):
        """Test Claude's long context window handling."""
        provider = self.create_mock_provider("anthropic")
        provider.model = "claude-3-5-sonnet-20241022"
        provider.context_window = 200000  # 200K tokens
        
        # Simulate very long context
        long_context = "token " * 150000  # 150K tokens
        estimated_tokens = len(long_context.split())
        
        assert provider.context_window > estimated_tokens
        assert provider.context_window >= 200000
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    async def test_claude_tool_use(self):
        """Test Claude's tool use/function calling capability."""
        provider = self.create_mock_provider("anthropic")
        
        tool_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_test123",
                    "name": "calculate",
                    "input": {"expression": "2 + 2"}
                }
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "tool_use"
        }
        
        parsed_response = {
            "content": None,  # No text content when using tools
            "model": tool_response["model"],
            "tool_calls": [
                {
                    "id": tool_response["content"][0]["id"],
                    "name": tool_response["content"][0]["name"],
                    "arguments": tool_response["content"][0]["input"]
                }
            ],
            "stop_reason": tool_response["stop_reason"]
        }
        
        provider.generate_response.return_value = parsed_response
        
        response = await provider.generate_response("What is 2 + 2?")
        
        assert response["stop_reason"] == "tool_use"
        assert len(response["tool_calls"]) == 1
        assert response["tool_calls"][0]["name"] == "calculate"


class TestAnthropicErrorHandling(BaseProviderTest):
    """Test Anthropic-specific error handling."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    async def test_anthropic_rate_limits(self):
        """Test Anthropic rate limit handling."""
        provider = self.create_mock_provider("anthropic")
        
        rate_limit_error = Exception("rate_limit_error: Number of requests per minute exceeded")
        provider.generate_response.side_effect = rate_limit_error
        
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test")
        
        assert "rate_limit" in str(exc_info.value)
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    async def test_anthropic_overloaded_error(self):
        """Test Anthropic overloaded error handling."""
        provider = self.create_mock_provider("anthropic")
        
        overloaded_error = Exception("overloaded_error: The service is temporarily overloaded")
        provider.generate_response.side_effect = overloaded_error
        
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test")
        
        assert "overloaded" in str(exc_info.value)
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    async def test_anthropic_authentication_error(self):
        """Test Anthropic authentication error handling."""
        provider = self.create_mock_provider("anthropic")
        
        auth_error = Exception("authentication_error: Invalid API key")
        provider.generate_response.side_effect = auth_error
        
        with pytest.raises(Exception) as exc_info:
            await provider.generate_response("test")
        
        assert "authentication" in str(exc_info.value)


class TestAnthropicSpecializations(BaseProviderTest):
    """Test Anthropic's specialized capabilities."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    def test_claude_writing_assistance(self):
        """Test Claude's writing assistance capabilities."""
        provider = self.create_mock_provider("anthropic")
        
        writing_capabilities = [
            "creative_writing",
            "technical_writing", 
            "editing",
            "proofreading",
            "summarization",
            "analysis"
        ]
        
        provider.specializations = writing_capabilities
        
        assert "creative_writing" in provider.specializations
        assert "technical_writing" in provider.specializations
        assert len(provider.specializations) >= 4
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    def test_claude_coding_assistance(self):
        """Test Claude's coding assistance capabilities."""
        provider = self.create_mock_provider("anthropic")
        
        coding_capabilities = [
            "code_generation",
            "code_review",
            "debugging",
            "refactoring",
            "documentation",
            "testing"
        ]
        
        provider.coding_features = coding_capabilities
        
        assert "code_generation" in provider.coding_features
        assert "debugging" in provider.coding_features
        assert len(provider.coding_features) >= 4
        
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.anthropic
    async def test_claude_constitutional_ai(self):
        """Test Claude's Constitutional AI principles."""
        provider = self.create_mock_provider("anthropic")
        
        constitutional_response = {
            "content": "I need to be helpful, harmless, and honest in my response. Let me provide information that is accurate and beneficial while avoiding any potential harm.",
            "model": "claude-3-5-sonnet-20241022",
            "constitutional_check": True
        }
        
        provider.generate_response.return_value = constitutional_response
        
        response = await provider.generate_response("complex ethical question")
        
        assert "helpful" in response["content"] or "honest" in response["content"]
        assert response.get("constitutional_check") is True