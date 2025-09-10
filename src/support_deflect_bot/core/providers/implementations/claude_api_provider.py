"""Claude API provider implementation using direct Anthropic API."""

import logging
import time
from typing import List, Optional, Dict, Any

from ..base import (
    LLMProvider,
    ProviderConfig,
    ProviderType,
    ProviderTier,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


class ClaudeAPIProvider(LLMProvider):
    """Claude API provider using direct Anthropic API - Premium cost-effective option."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Claude API provider.

        Args:
            api_key: Anthropic API key
            **kwargs: Additional configuration options
        """
        try:
            import anthropic

            self.anthropic = anthropic
        except ImportError:
            raise ProviderUnavailableError(
                "Anthropic SDK not available. Install with: pip install anthropic",
                provider="claude_api",
            )

        super().__init__(api_key=api_key, **kwargs)

        # Initialize Anthropic client
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            # Try to use default client (environment ANTHROPIC_API_KEY)
            try:
                self.client = anthropic.Anthropic()
            except Exception:
                self.client = None

        # Default model from settings
        from ....utils.settings import CLAUDE_API_MODEL

        self.default_model = CLAUDE_API_MODEL

        logger.info(f"Initialized Claude API provider with model: {self.default_model}")

    def get_config(self) -> ProviderConfig:
        """Get Claude API provider configuration."""
        return ProviderConfig(
            name="Claude API",
            provider_type=ProviderType.LLM,
            cost_per_million_tokens_input=3.00,  # Claude 3 Haiku input
            cost_per_million_tokens_output=15.00,  # Claude 3 Haiku output
            max_context_length=200000,  # Claude 3 has large context window
            rate_limit_rpm=1000,  # Anthropic rate limits (conservative estimate)
            rate_limit_tpm=300000,  # Anthropic TPM limits
            supports_streaming=True,
            requires_api_key=True,
            tier=ProviderTier.PAID,
            regions_supported=["US", "EU", "global"],  # Anthropic supported regions
            gdpr_compliant=True,  # Anthropic is GDPR compliant
            models_available=[
                "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229",
                "claude-3-5-sonnet-20240620",
            ],
        )

    def is_available(self) -> bool:
        """Check if Claude API provider is available and properly configured."""
        if not self.client:
            return False

        try:
            # Test with a minimal API call - try to list models or make a simple request
            # Since Anthropic doesn't have a models.list endpoint, we'll do a simple message test
            response = self.client.messages.create(
                model=self.default_model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return response is not None
        except Exception as e:
            logger.debug(f"Claude API availability check failed: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            if not self.client:
                return {
                    "status": "unhealthy",
                    "error": "Claude API client not initialized",
                    "provider": "claude_api",
                }

            # Test API connectivity with a minimal request
            start_time = time.time()
            response = self.client.messages.create(
                model=self.default_model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}],
            )
            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "model_tested": self.default_model,
                "provider": "claude_api",
                "timestamp": time.time(),
            }

        except Exception as e:
            error_str = str(e)
            status = "unhealthy"

            # Check for specific error types
            if "rate limit" in error_str.lower():
                status = "rate_limited"
            elif (
                "authentication" in error_str.lower() or "api key" in error_str.lower()
            ):
                status = "auth_error"

            return {
                "status": status,
                "error": error_str,
                "provider": "claude_api",
                "timestamp": time.time(),
            }

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate chat completion using Claude API.

        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Specific model to use (defaults to configured model)
            temperature: Randomness in generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic API parameters

        Returns:
            Generated response text

        Raises:
            ProviderError: If API call fails
            ProviderRateLimitError: If rate limit exceeded
        """
        if not self.client:
            raise ProviderUnavailableError(
                "Claude API client not available", provider="claude_api"
            )

        model = model or self.default_model
        max_tokens = max_tokens or 4000  # Default max tokens for Claude

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                **kwargs,
            )

            return response.content[0].text.strip()

        except self.anthropic.RateLimitError as e:
            raise ProviderRateLimitError(
                f"Claude API rate limit exceeded: {e}",
                provider="claude_api",
                original_error=e,
            )
        except self.anthropic.APIError as e:
            raise ProviderError(
                f"Claude API error: {e}", provider="claude_api", original_error=e
            )
        except Exception as e:
            raise ProviderError(
                f"Claude API chat failed: {e}", provider="claude_api", original_error=e
            )

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text using Claude's tokenization.

        Args:
            text: Text to count tokens for
            model: Model parameter (for consistency)

        Returns:
            Estimated number of tokens
        """
        try:
            # Try to use Anthropic's token counting if available
            response = self.client.count_tokens(text)
            return response.tokens
        except (AttributeError, Exception):
            # Fallback to estimation if token counting not available
            # Claude uses similar tokenization to other models (slightly more efficient)
            # Use a conservative estimate: 1 token ≈ 3.5 characters for Claude
            return max(1, len(text) // 3.5)

    def get_usage_info(self) -> Dict[str, Any]:
        """Get information about Claude API usage and pricing.

        Returns:
            Usage information dictionary
        """
        return {
            "pricing_model": "Pay-per-token",
            "haiku_input": "$0.25 per 1M tokens",
            "haiku_output": "$1.25 per 1M tokens",
            "sonnet_input": "$3.00 per 1M tokens",
            "sonnet_output": "$15.00 per 1M tokens",
            "opus_input": "$15.00 per 1M tokens",
            "opus_output": "$75.00 per 1M tokens",
            "context_length": "200,000 tokens",
            "recommended_model": "claude-3-haiku-20240307 (most cost-effective)",
            "rate_limits": "1000 RPM, 300k TPM (varies by tier)",
        }
