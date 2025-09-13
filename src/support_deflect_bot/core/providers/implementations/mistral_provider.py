"""Mistral provider implementation - EU-based, globally compliant alternative."""

import logging
import time
from typing import Any, Dict, Optional

from ..base import (
    LLMProvider,
    ProviderConfig,
    ProviderError,
    ProviderRateLimitError,
    ProviderTier,
    ProviderType,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


class MistralProvider(LLMProvider):
    """Mistral provider - EU-based, globally compliant alternative."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Mistral provider.

        Args:
            api_key: Mistral API key
            **kwargs: Additional configuration options
        """
        try:
            import mistralai
            from mistralai.client import MistralClient

            self.mistralai = mistralai
            self.MistralClient = MistralClient
        except ImportError:
            raise ProviderUnavailableError(
                "Mistral SDK not available. Install with: pip install mistralai",
                provider="mistral",
            )

        super().__init__(api_key=api_key, **kwargs)

        # Initialize Mistral client
        if self.api_key:
            self.client = MistralClient(api_key=self.api_key)
        else:
            # Try to use environment variable
            try:
                self.client = MistralClient()
            except Exception:
                self.client = None

        # Default model from settings
        from ....utils.settings import MISTRAL_MODEL

        self.default_model = MISTRAL_MODEL

        logger.info(f"Initialized Mistral provider with model: {self.default_model}")

    def get_config(self) -> ProviderConfig:
        """Get Mistral provider configuration."""
        return ProviderConfig(
            name="Mistral",
            provider_type=ProviderType.LLM,
            cost_per_million_tokens_input=0.27,  # Very competitive pricing
            cost_per_million_tokens_output=1.10,  # Reasonable output cost
            max_context_length=32000,  # 32k context for most models
            rate_limit_rpm=100,  # Conservative estimate
            rate_limit_tpm=100000,  # Conservative estimate
            supports_streaming=True,  # Mistral supports streaming
            requires_api_key=True,
            tier=ProviderTier.PAID,
            regions_supported=["global"],  # EU-based, works globally
            gdpr_compliant=True,  # French company, GDPR compliant
            models_available=[
                "mistral-small-latest",  # Cost-effective model
                "mistral-medium-latest",  # Balanced model
                "mistral-large-latest",  # Most capable model
                "open-mistral-7b",  # Open source model
                "open-mixtral-8x7b",  # Mixture of experts model
                "open-mixtral-8x22b",  # Larger mixture of experts
            ],
        )

    def is_available(self) -> bool:
        """Check if Mistral provider is available and properly configured."""
        if not self.client:
            return False

        try:
            # Test with models list
            models = self.client.list_models()
            return len(models.data) > 0
        except Exception as e:
            logger.debug(f"Mistral availability check failed: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            if not self.client:
                return {
                    "status": "unhealthy",
                    "error": "Mistral client not initialized",
                    "provider": "mistral",
                }

            # Test API connectivity
            start_time = time.time()
            models = self.client.list_models()
            response_time = time.time() - start_time

            # Check if default model is available
            available_models = [model.id for model in models.data]
            has_default_model = self.default_model in available_models

            return {
                "status": "healthy" if has_default_model else "degraded",
                "response_time_ms": round(response_time * 1000, 2),
                "models_available": len(available_models),
                "default_model_available": has_default_model,
                "provider": "mistral",
                "region": "EU",  # Mistral is EU-based
                "gdpr_compliant": True,
                "timestamp": time.time(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": "mistral",
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
        """Generate chat completion using Mistral models.

        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Specific model to use (defaults to configured model)
            temperature: Randomness in generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Mistral API parameters

        Returns:
            Generated response text

        Raises:
            ProviderError: If API call fails
            ProviderRateLimitError: If rate limit exceeded
        """
        if not self.client:
            raise ProviderUnavailableError(
                "Mistral client not available", provider="mistral"
            )

        model = model or self.default_model

        try:
            # Format messages for Mistral API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Mistral API call
            response = self.client.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                raise ProviderError("Empty response from Mistral", provider="mistral")

        except Exception as e:
            error_str = str(e).lower()

            # Check for rate limit errors
            if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
                raise ProviderRateLimitError(
                    f"Mistral rate limit exceeded: {e}",
                    provider="mistral",
                    original_error=e,
                )

            # Check for API errors
            if "api" in error_str or "unauthorized" in error_str or "401" in error_str:
                raise ProviderError(
                    f"Mistral API error: {e}", provider="mistral", original_error=e
                )

            # Check for model errors
            if "model" in error_str and "not found" in error_str:
                raise ProviderError(
                    f"Mistral model not available: {model}",
                    provider="mistral",
                    original_error=e,
                )

            # Generic error
            raise ProviderError(
                f"Mistral chat failed: {e}", provider="mistral", original_error=e
            )

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text. Mistral doesn't provide direct token counting.

        Args:
            text: Text to count tokens for
            model: Model to use for tokenization (not used for estimation)

        Returns:
            Estimated number of tokens
        """
        # Mistral uses similar tokenization to other transformer models
        # Use base class estimation (1 token ≈ 4 characters)
        return self.estimate_tokens(text)

    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific Mistral model.

        Args:
            model: Model name to get info for

        Returns:
            Model information dictionary
        """
        model = model or self.default_model

        model_info = {
            "mistral-small-latest": {
                "context_length": 32000,
                "description": "Cost-effective model suitable for simple tasks",
                "speed": "fast",
                "quality": "good",
                "use_case": "Translation, summarization, sentiment analysis",
            },
            "mistral-medium-latest": {
                "context_length": 32000,
                "description": "Balanced performance for intermediate tasks",
                "speed": "medium",
                "quality": "high",
                "use_case": "Data extraction, document analysis, code generation",
            },
            "mistral-large-latest": {
                "context_length": 32000,
                "description": "Most capable model for complex reasoning",
                "speed": "slower",
                "quality": "highest",
                "use_case": "Complex reasoning, math, code, creative writing",
            },
            "open-mistral-7b": {
                "context_length": 32000,
                "description": "Open source 7B parameter model",
                "speed": "very_fast",
                "quality": "medium",
                "use_case": "Simple completion, classification",
            },
            "open-mixtral-8x7b": {
                "context_length": 32000,
                "description": "Open source mixture of experts model",
                "speed": "fast",
                "quality": "high",
                "use_case": "General purpose, multilingual tasks",
            },
            "open-mixtral-8x22b": {
                "context_length": 64000,
                "description": "Larger open source mixture of experts",
                "speed": "medium",
                "quality": "very_high",
                "use_case": "Complex tasks, code generation, reasoning",
            },
        }

        return model_info.get(
            model,
            {
                "context_length": 32000,
                "description": f"Unknown Mistral model: {model}",
                "speed": "unknown",
                "quality": "unknown",
                "use_case": "General purpose",
            },
        )

    def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        """Generate streaming chat completion using Mistral.

        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Specific model to use
            temperature: Randomness in generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API parameters

        Yields:
            Streaming response chunks
        """
        if not self.client:
            raise ProviderUnavailableError(
                "Mistral client not available", provider="mistral"
            )

        model = model or self.default_model

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Create streaming response
            stream = self.client.chat_stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield delta.content

        except Exception as e:
            raise ProviderError(
                f"Mistral streaming failed: {e}", provider="mistral", original_error=e
            )

    def get_pricing_info(self) -> Dict[str, Dict[str, float]]:
        """Get current pricing information for Mistral models.

        Returns:
            Dictionary with pricing per model
        """
        return {
            "mistral-small-latest": {
                "input_cost_per_million": 0.27,
                "output_cost_per_million": 1.10,
            },
            "mistral-medium-latest": {
                "input_cost_per_million": 2.70,
                "output_cost_per_million": 8.10,
            },
            "mistral-large-latest": {
                "input_cost_per_million": 8.00,
                "output_cost_per_million": 24.00,
            },
            "open-mistral-7b": {
                "input_cost_per_million": 0.25,
                "output_cost_per_million": 0.25,
            },
            "open-mixtral-8x7b": {
                "input_cost_per_million": 0.70,
                "output_cost_per_million": 0.70,
            },
            "open-mixtral-8x22b": {
                "input_cost_per_million": 2.00,
                "output_cost_per_million": 6.00,
            },
        }
