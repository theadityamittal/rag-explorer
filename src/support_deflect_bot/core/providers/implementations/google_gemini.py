"""Google Gemini provider implementations with free and paid tiers."""

import logging
import signal
import time
from typing import Any, Dict, List, Optional

from ..base import (
    CombinedProvider,
    ProviderConfig,
    ProviderError,
    ProviderRateLimitError,
    ProviderTier,
    ProviderType,
    ProviderUnavailableError,
)
from ...resilience import (
    retry_with_backoff,
    RetryPolicy,
    CircuitBreakerConfig,
    get_circuit_breaker,
    get_provider_retry_policy,
    get_provider_circuit_breaker_config,
    ErrorClassifier,
    ErrorType
)

logger = logging.getLogger(__name__)


class TimeoutContext:
    """Simple timeout context manager for Google API calls."""

    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds

    def __enter__(self):
        # Set up timeout using signal (Unix/Linux/macOS only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(int(self.timeout_seconds))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cancel timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(f"Operation timed out after {self.timeout_seconds} seconds")


class GoogleGeminiBaseProvider(CombinedProvider):
    """Base class for Google Gemini providers with resilience patterns."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Google Gemini provider.

        Args:
            api_key: Google API key
            **kwargs: Additional configuration options
        """
        try:
            import google.generativeai as genai

            self.genai = genai
        except ImportError:
            raise ProviderUnavailableError(
                "Google GenerativeAI SDK not available. Install with: pip install google-generativeai",
                provider="google_gemini",
            )

        super().__init__(api_key=api_key, **kwargs)

        # Configure Google API
        if self.api_key:
            genai.configure(api_key=self.api_key)

        # Default models and timeout from settings
        from ....utils.settings import GOOGLE_EMBEDDING_MODEL, GOOGLE_LLM_MODEL, PROVIDER_TIMEOUT

        self.default_llm_model = GOOGLE_LLM_MODEL
        self.default_embedding_model = GOOGLE_EMBEDDING_MODEL
        self.provider_timeout = PROVIDER_TIMEOUT

        # Initialize resilience components
        self.retry_policy = get_provider_retry_policy("google")
        self.circuit_breaker = get_circuit_breaker(
            f"google_gemini_{id(self)}",
            get_provider_circuit_breaker_config("google")
        )

        # Metrics tracking
        self._request_count = 0
        self._error_count = 0
        self._rate_limit_count = 0

        # Initialize models
        try:
            self.llm_model = genai.GenerativeModel(self.default_llm_model)
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini LLM model: {e}")
            self.llm_model = None

        logger.info(
            f"Initialized Google Gemini provider with models: {self.default_llm_model}, {self.default_embedding_model}"
        )

    def is_available(self) -> bool:
        """Check if Google Gemini provider is available."""
        if not self.api_key:
            return False

        try:
            # Test with model list
            models = list(self.genai.list_models())
            return len(models) > 0
        except Exception as e:
            logger.debug(f"Google Gemini availability check failed: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check with circuit breaker state."""
        try:
            if not self.api_key:
                return {
                    "status": "unhealthy",
                    "error": "Google API key not configured",
                    "provider": "google_gemini",
                    "circuit_breaker": self.circuit_breaker.get_status(),
                    "metrics": self._get_metrics(),
                }

            # Test API connectivity
            start_time = time.time()
            models = list(self.genai.list_models())
            response_time = time.time() - start_time

            # Check if required models are available
            model_names = [model.name.split("/")[-1] for model in models]
            has_llm_model = any(self.default_llm_model in name for name in model_names)
            has_embedding_model = any(
                self.default_embedding_model in name for name in model_names
            )

            # Determine overall status
            circuit_state = self.circuit_breaker.state.value
            if circuit_state == "open":
                status = "circuit_open"
            elif not (has_llm_model and has_embedding_model):
                status = "degraded"
            else:
                status = "healthy"

            return {
                "status": status,
                "response_time_ms": round(response_time * 1000, 2),
                "models_available": len(models),
                "default_llm_available": has_llm_model,
                "default_embedding_available": has_embedding_model,
                "provider": "google_gemini",
                "timestamp": time.time(),
                "circuit_breaker": self.circuit_breaker.get_status(),
                "metrics": self._get_metrics(),
            }

        except Exception as e:
            self._error_count += 1
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": "google_gemini",
                "timestamp": time.time(),
                "circuit_breaker": self.circuit_breaker.get_status(),
                "metrics": self._get_metrics(),
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
        """Generate chat completion using Google Gemini with retry and circuit breaker protection.

        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Specific model to use (defaults to configured model)
            temperature: Randomness in generation (0.0-1.0)
            max_tokens: Maximum tokens to generate (max_output_tokens in Gemini)
            **kwargs: Additional Gemini API parameters

        Returns:
            Generated response text

        Raises:
            ProviderError: If API call fails
            ProviderRateLimitError: If rate limit exceeded
            CircuitBreakerOpenException: If circuit breaker is open
        """
        self._request_count += 1
        return self._chat_impl(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    @retry_with_backoff(get_provider_retry_policy("google"))
    def _chat_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Internal implementation of chat with retry logic."""
        if not self.llm_model:
            raise ProviderUnavailableError(
                "Google Gemini model not available", provider="google_gemini"
            )

        try:
            with self.circuit_breaker:
                # Combine system and user prompts for Gemini
                full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

                # Configure generation settings
                generation_config = self.genai.types.GenerationConfig(
                    temperature=temperature, max_output_tokens=max_tokens, **kwargs
                )

                with TimeoutContext(self.provider_timeout):
                    response = self.llm_model.generate_content(
                        full_prompt, generation_config=generation_config
                    )

                if response.parts:
                    result = response.text.strip()
                    logger.debug(f"Google Gemini chat successful, response length: {len(result)}")
                    return result
                else:
                    raise ProviderError(
                        "Empty response from Gemini", provider="google_gemini"
                    )

        except Exception as e:
            self._error_count += 1
            error_str = str(e).lower()

            # Classify error for retry decisions
            error_type = ErrorClassifier.classify_error(e)

            # Check for rate limit errors
            if "quota" in error_str or "rate limit" in error_str:
                self._rate_limit_count += 1
                logger.warning(f"Google Gemini rate limit encountered: {e}")
                raise ProviderRateLimitError(
                    f"Google Gemini rate limit exceeded: {e}",
                    provider="google_gemini",
                    original_error=e,
                )

            # Check for permanent API errors
            if (
                "api key" in error_str
                or "forbidden" in error_str
                or "unauthorized" in error_str
                or "permission" in error_str
            ):
                logger.error(f"Google Gemini authentication/authorization error: {e}")
                raise ProviderError(
                    f"Google Gemini API error: {e}",
                    provider="google_gemini",
                    original_error=e,
                )

            # Check for model/configuration errors (permanent)
            if (
                "model" in error_str and "not found" in error_str
                or "invalid" in error_str and "request" in error_str
            ):
                logger.error(f"Google Gemini configuration error: {e}")
                raise ProviderError(
                    f"Google Gemini configuration error: {e}",
                    provider="google_gemini",
                    original_error=e,
                )

            # Log error for debugging
            logger.warning(f"Google Gemini chat error (type: {error_type.value}): {e}")

            # Generic error - may be retryable
            raise ProviderError(
                f"Google Gemini chat failed: {e}",
                provider="google_gemini",
                original_error=e,
            )

    def embed_texts(
        self, texts: List[str], model: Optional[str] = None, batch_size: int = 10
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts using Google Gemini with retry and circuit breaker protection.

        Args:
            texts: List of texts to embed
            model: Specific embedding model to use
            batch_size: Number of texts to process at once

        Returns:
            List of embedding vectors

        Raises:
            ProviderError: If API call fails
            ProviderRateLimitError: If rate limit exceeded
            CircuitBreakerOpenException: If circuit breaker is open
        """
        self._request_count += 1
        return self._embed_texts_impl(texts=texts, model=model, batch_size=batch_size)

    @retry_with_backoff(get_provider_retry_policy("google"))
    def _embed_texts_impl(
        self, texts: List[str], model: Optional[str] = None, batch_size: int = 10
    ) -> List[List[float]]:
        """Internal implementation of embed_texts with retry logic."""
        if not texts:
            return []

        model = model or self.default_embedding_model
        embeddings = []

        try:
            with self.circuit_breaker:
                # Process in batches
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]

                    for text in batch:
                        if not text.strip():
                            # Use zero vector for empty text
                            embeddings.append([0.0] * self.get_embedding_dimension(model))
                            continue

                        try:
                            with TimeoutContext(self.provider_timeout):
                                result = self.genai.embed_content(
                                    model=f"models/{model}", content=text
                                )
                            embeddings.append(result["embedding"])

                        except Exception as e:
                            # Handle individual text embedding failures
                            logger.warning(f"Failed to embed text (length {len(text)}): {e}")
                            # Use zero vector as fallback for this specific text
                            embeddings.append([0.0] * self.get_embedding_dimension(model))

                logger.debug(f"Google Gemini embeddings successful for {len(texts)} texts")
                return embeddings

        except Exception as e:
            self._error_count += 1
            error_str = str(e).lower()

            # Classify error for retry decisions
            error_type = ErrorClassifier.classify_error(e)

            # Check for rate limit errors
            if "quota" in error_str or "rate limit" in error_str:
                self._rate_limit_count += 1
                logger.warning(f"Google Gemini embedding rate limit encountered: {e}")
                raise ProviderRateLimitError(
                    f"Google Gemini embedding rate limit: {e}",
                    provider="google_gemini",
                    original_error=e,
                )

            # Check for permanent API errors
            if (
                "api key" in error_str
                or "forbidden" in error_str
                or "unauthorized" in error_str
                or "permission" in error_str
            ):
                logger.error(f"Google Gemini embedding authentication error: {e}")
                raise ProviderError(
                    f"Google Gemini embedding API error: {e}",
                    provider="google_gemini",
                    original_error=e,
                )

            # Check for model errors (permanent)
            if "model" in error_str and ("not found" in error_str or "invalid" in error_str):
                logger.error(f"Google Gemini embedding model error: {e}")
                raise ProviderError(
                    f"Google Gemini embedding model error: {e}",
                    provider="google_gemini",
                    original_error=e,
                )

            # Log error for debugging
            logger.warning(f"Google Gemini embedding error (type: {error_type.value}): {e}")

            # Generic error - may be retryable
            raise ProviderError(
                f"Google Gemini embeddings failed: {e}",
                provider="google_gemini",
                original_error=e,
            )

    def embed_one(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding for single text using Google Gemini.

        Args:
            text: Text to embed
            model: Specific embedding model to use

        Returns:
            Embedding vector
        """
        if not text.strip():
            return [0.0] * 768  # Default dimension for empty text

        embeddings = self.embed_texts([text], model=model)
        return embeddings[0] if embeddings else [0.0] * 768

    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get embedding dimension for the specified model.

        Args:
            model: Embedding model name

        Returns:
            Embedding vector dimension
        """
        model = model or self.default_embedding_model

        # Google embedding model dimensions
        dimensions = {
            "gemini-embedding-001": 768,
            "text-embedding-004": 768,
            "embedding-001": 768,
        }

        return dimensions.get(model, 768)  # Default dimension

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text using Google's token counting if available.

        Args:
            text: Text to count tokens for
            model: Model to use for tokenization

        Returns:
            Number of tokens
        """
        model = model or self.default_llm_model

        try:
            # Try to use Gemini's token counting if available
            if self.llm_model:
                response = self.llm_model.count_tokens(text)
                return response.total_tokens
        except Exception as e:
            logger.debug(f"Google token counting failed: {e}")

        # Fallback to estimation
        return self.estimate_tokens(text)

    def _get_metrics(self) -> Dict[str, Any]:
        """Get provider metrics for monitoring."""
        return {
            "requests_total": self._request_count,
            "errors_total": self._error_count,
            "rate_limits_total": self._rate_limit_count,
            "error_rate": self._error_count / max(1, self._request_count),
            "rate_limit_rate": self._rate_limit_count / max(1, self._request_count),
        }

    def get_provider_status(self) -> Dict[str, Any]:
        """Get comprehensive provider status including resilience state."""
        return {
            "provider": "google_gemini",
            "available": self.is_available(),
            "health": self.health_check(),
            "circuit_breaker": self.circuit_breaker.get_status(),
            "metrics": self._get_metrics(),
            "retry_policy": {
                "max_retries": self.retry_policy.max_retries,
                "base_delay": self.retry_policy.base_delay,
                "max_delay": self.retry_policy.max_delay,
            },
            "configuration": {
                "default_llm_model": self.default_llm_model,
                "default_embedding_model": self.default_embedding_model,
                "api_key_configured": bool(self.api_key),
            }
        }


class GoogleGeminiFreeProvider(GoogleGeminiBaseProvider):
    """Google Gemini free tier provider (restricted regions)."""

    def get_config(self) -> ProviderConfig:
        """Get Google Gemini free tier configuration."""
        return ProviderConfig(
            name="Google Gemini (Free)",
            provider_type=ProviderType.BOTH,
            cost_per_million_tokens_input=0.0,  # FREE tier
            cost_per_million_tokens_output=0.0,
            max_context_length=1000000,  # 1M context window
            rate_limit_rpm=60,  # Free tier limit
            rate_limit_tpm=60000,  # Free tier limit
            supports_streaming=True,
            requires_api_key=True,
            tier=ProviderTier.FREE,
            regions_supported=["US", "CA", "AU", "JP"],  # Restricted in GDPR regions
            gdpr_compliant=False,  # Free tier not GDPR compliant
            models_available=["gemini-2.5-flash-lite", "gemini-embedding-001"],
        )


class GoogleGeminiPaidProvider(GoogleGeminiBaseProvider):
    """Google Gemini paid tier provider (globally compliant)."""

    def get_config(self) -> ProviderConfig:
        """Get Google Gemini paid tier configuration."""
        return ProviderConfig(
            name="Google Gemini (Paid)",
            provider_type=ProviderType.BOTH,
            cost_per_million_tokens_input=7.0,  # Paid tier pricing (Gemini Pro)
            cost_per_million_tokens_output=21.0,  # Higher output cost
            max_context_length=1000000,  # 1M context window
            rate_limit_rpm=300,  # Higher rate limits
            rate_limit_tpm=4000000,  # Higher token limits
            supports_streaming=True,
            requires_api_key=True,
            tier=ProviderTier.PAID,
            regions_supported=["global"],  # Works everywhere
            gdpr_compliant=True,  # Paid tier is GDPR compliant
            models_available=["gemini-2.5-flash-lite", "gemini-embedding-001"],
        )
