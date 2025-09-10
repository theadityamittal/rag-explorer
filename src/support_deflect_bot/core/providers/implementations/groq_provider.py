"""Groq provider implementation for ultra-fast inference with Llama models."""

import logging
import time
from typing import Optional, Dict, Any

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


class GroqProvider(LLMProvider):
    """Groq provider for ultra-fast inference with Llama models."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Groq provider.

        Args:
            api_key: Groq API key
            **kwargs: Additional configuration options
        """
        try:
            import groq

            self.groq = groq
        except ImportError:
            raise ProviderUnavailableError(
                "Groq SDK not available. Install with: pip install groq",
                provider="groq",
            )

        super().__init__(api_key=api_key, **kwargs)

        # Initialize Groq client
        if self.api_key:
            self.client = groq.Groq(api_key=self.api_key)
        else:
            # Try to use default client (environment GROQ_API_KEY)
            try:
                self.client = groq.Groq()
            except Exception:
                self.client = None

        # Default model from settings
        from ....utils.settings import GROQ_MODEL

        self.default_model = GROQ_MODEL

        logger.info(f"Initialized Groq provider with model: {self.default_model}")

    def get_config(self) -> ProviderConfig:
        """Get Groq provider configuration."""
        return ProviderConfig(
            name="Groq",
            provider_type=ProviderType.LLM,
            cost_per_million_tokens_input=0.59,  # Very competitive pricing
            cost_per_million_tokens_output=0.79,  # Output slightly higher
            max_context_length=8192,  # Varies by model
            rate_limit_rpm=30,  # Conservative estimate
            rate_limit_tpm=30000,  # Conservative estimate
            supports_streaming=True,  # Groq supports streaming
            requires_api_key=True,
            tier=ProviderTier.PAID,
            regions_supported=["US", "CA"],  # US-based service
            gdpr_compliant=False,  # Not GDPR compliant (US-based)
            models_available=[
                "llama-3.1-70b-versatile",  # Primary model
                "llama-3.1-8b-instant",  # Fastest model
                "mixtral-8x7b-32768",  # Alternative model
                "gemma2-9b-it",  # Google's Gemma model
            ],
        )

    def is_available(self) -> bool:
        """Check if Groq provider is available and properly configured."""
        if not self.client:
            return False

        try:
            # Test with a minimal API call
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            logger.debug(f"Groq availability check failed: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            if not self.client:
                return {
                    "status": "unhealthy",
                    "error": "Groq client not initialized",
                    "provider": "groq",
                }

            # Test API connectivity and measure speed
            start_time = time.time()
            models = self.client.models.list()
            response_time = time.time() - start_time

            # Check if default model is available
            available_models = [model.id for model in models.data]
            has_default_model = self.default_model in available_models

            return {
                "status": "healthy" if has_default_model else "degraded",
                "response_time_ms": round(response_time * 1000, 2),
                "models_available": len(available_models),
                "default_model_available": has_default_model,
                "provider": "groq",
                "speed_rating": "ultra_fast",  # Groq's main selling point
                "timestamp": time.time(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": "groq",
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
        """Generate chat completion using Groq's ultra-fast inference.

        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Specific model to use (defaults to configured model)
            temperature: Randomness in generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Groq API parameters

        Returns:
            Generated response text

        Raises:
            ProviderError: If API call fails
            ProviderRateLimitError: If rate limit exceeded
        """
        if not self.client:
            raise ProviderUnavailableError("Groq client not available", provider="groq")

        model = model or self.default_model

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Groq API call
            completion = self.client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            error_str = str(e).lower()

            # Check for rate limit errors
            if "rate limit" in error_str or "quota" in error_str:
                raise ProviderRateLimitError(
                    f"Groq rate limit exceeded: {e}", provider="groq", original_error=e
                )

            # Check for API errors
            if "api" in error_str or "unauthorized" in error_str:
                raise ProviderError(
                    f"Groq API error: {e}", provider="groq", original_error=e
                )

            # Generic error
            raise ProviderError(
                f"Groq chat failed: {e}", provider="groq", original_error=e
            )

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text. Groq doesn't provide tiktoken equivalent, so use estimation.

        Args:
            text: Text to count tokens for
            model: Model to use for tokenization (not used for estimation)

        Returns:
            Estimated number of tokens
        """
        # Groq uses Llama-based models, which have similar tokenization to GPT
        # Use base class estimation (1 token ≈ 4 characters)
        return self.estimate_tokens(text)

    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific model.

        Args:
            model: Model name to get info for

        Returns:
            Model information dictionary
        """
        model = model or self.default_model

        model_info = {
            "llama-3.1-70b-versatile": {
                "context_length": 8192,
                "description": "Most capable Llama 3.1 model with 70B parameters",
                "speed": "fast",
                "quality": "high",
            },
            "llama-3.1-8b-instant": {
                "context_length": 8192,
                "description": "Fastest Llama 3.1 model with 8B parameters",
                "speed": "ultra_fast",
                "quality": "medium",
            },
            "mixtral-8x7b-32768": {
                "context_length": 32768,
                "description": "Mixtral model with large context window",
                "speed": "fast",
                "quality": "high",
            },
            "gemma2-9b-it": {
                "context_length": 8192,
                "description": "Google Gemma 2 model optimized for instruction following",
                "speed": "fast",
                "quality": "medium-high",
            },
        }

        return model_info.get(
            model,
            {
                "context_length": 8192,
                "description": f"Unknown model: {model}",
                "speed": "unknown",
                "quality": "unknown",
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
        """Generate streaming chat completion (Groq's strength).

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
            raise ProviderUnavailableError("Groq client not available", provider="groq")

        model = model or self.default_model

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            stream = self.client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise ProviderError(
                f"Groq streaming failed: {e}", provider="groq", original_error=e
            )
