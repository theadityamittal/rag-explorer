"""Ollama provider implementation for local models - backward compatibility."""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from ..base import (
    CombinedProvider,
    ProviderConfig,
    ProviderError,
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

# Default model constants for reliable operation
DEFAULT_LLM_MODEL = "llama3.1"  # Reliable, well-tested
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"  # Fast, accurate embeddings


class OllamaProvider(CombinedProvider):
    """Local Ollama provider for backward compatibility and privacy-focused deployment with resilience patterns."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Ollama provider.

        Args:
            api_key: Not used for Ollama (local deployment)
            **kwargs: Additional configuration options including host
        """
        try:
            import ollama

            self.ollama = ollama
        except ImportError:
            raise ProviderUnavailableError(
                "Ollama SDK not available. Install with: pip install ollama",
                provider="ollama",
            )

        super().__init__(api_key=None, **kwargs)  # No API key needed for local

        # Get Ollama configuration from settings
        from ....utils.settings import OLLAMA_EMBED_MODEL, OLLAMA_HOST, OLLAMA_MODEL

        self.default_llm_model = OLLAMA_MODEL
        self.default_embedding_model = OLLAMA_EMBED_MODEL
        self.ollama_host = kwargs.get("host", OLLAMA_HOST)

        # Initialize resilience components with local service settings
        self.retry_policy = get_provider_retry_policy("ollama")
        self.circuit_breaker = get_circuit_breaker(
            f"ollama_{id(self)}",
            get_provider_circuit_breaker_config("ollama")
        )

        # Metrics tracking
        self._request_count = 0
        self._error_count = 0
        self._connection_failures = 0

        # Configure Ollama host if specified
        if self.ollama_host:
            os.environ["OLLAMA_HOST"] = self.ollama_host

        logger.info(
            f"Initialized Ollama provider with models: {self.default_llm_model}, {self.default_embedding_model}"
        )
        if self.ollama_host:
            logger.info(f"Using Ollama host: {self.ollama_host}")

    def get_config(self) -> ProviderConfig:
        """Get Ollama provider configuration."""
        return ProviderConfig(
            name="Ollama (Local)",
            provider_type=ProviderType.BOTH,
            cost_per_million_tokens_input=0.0,  # Local deployment, no API costs
            cost_per_million_tokens_output=0.0,
            max_context_length=4096,  # Depends on model, conservative estimate
            rate_limit_rpm=float("inf"),  # No rate limits for local
            rate_limit_tpm=float("inf"),
            supports_streaming=True,  # Ollama supports streaming
            requires_api_key=False,  # Local deployment
            tier=ProviderTier.FREE,  # No recurring costs
            regions_supported=["global"],  # Local, works anywhere
            gdpr_compliant=True,  # Local data processing
            models_available=[
                "llama3.1",
                "llama3.1:8b",
                "llama3.1:70b",
                "llama2",
                "codellama",
                "mistral",
                "nomic-embed-text",
                "all-minilm",
            ],
        )

    def is_available(self) -> bool:
        """Check if Ollama is available and running."""
        try:
            # Test connection to Ollama service
            models = self.ollama.list()
            return True
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check with circuit breaker state."""
        try:
            # Test Ollama service connection
            start_time = time.time()
            models = self.ollama.list()
            response_time = time.time() - start_time

            # Get list of available models
            available_models = (
                [model["name"] for model in models["models"]]
                if "models" in models
                else []
            )

            # Check if required models are available
            has_llm_model = any(
                self.default_llm_model in model for model in available_models
            )
            has_embedding_model = any(
                self.default_embedding_model in model for model in available_models
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
                "models_available": len(available_models),
                "available_models": available_models[:5],  # Show first 5 models
                "default_llm_available": has_llm_model,
                "default_embedding_available": has_embedding_model,
                "ollama_host": self.ollama_host or "localhost:11434",
                "provider": "ollama",
                "deployment": "local",
                "timestamp": time.time(),
                "circuit_breaker": self.circuit_breaker.get_status(),
                "metrics": self._get_metrics(),
            }

        except Exception as e:
            self._connection_failures += 1
            self._error_count += 1
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": "ollama",
                "ollama_host": self.ollama_host or "localhost:11434",
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
        """Generate chat completion using local Ollama models with retry and circuit breaker protection.

        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Specific model to use (defaults to configured model)
            temperature: Randomness in generation (0.0-1.0)
            max_tokens: Maximum tokens to generate (num_predict in Ollama)
            **kwargs: Additional Ollama parameters

        Returns:
            Generated response text

        Raises:
            ProviderError: If Ollama call fails
            ProviderUnavailableError: If Ollama not available
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

    @retry_with_backoff(get_provider_retry_policy("ollama"))
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
        if not self.is_available():
            self._connection_failures += 1
            raise ProviderUnavailableError(
                "Ollama service not available", provider="ollama"
            )

        model = model or self.default_llm_model

        try:
            with self.circuit_breaker:
                # Format messages for Ollama
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                # Ollama API options
                options = {
                    "temperature": temperature,
                }

                if max_tokens:
                    options["num_predict"] = max_tokens

                # Add any additional options
                options.update(kwargs)

                # Make the API call
                response = self.ollama.chat(model=model, messages=messages, options=options)

                if "message" in response and "content" in response["message"]:
                    result = response["message"]["content"].strip()
                    logger.debug(f"Ollama chat successful, response length: {len(result)}")
                    return result
                else:
                    raise ProviderError(
                        "Unexpected response format from Ollama", provider="ollama"
                    )

        except Exception as e:
            self._error_count += 1

            # Check for connection-related errors
            error_str = str(e).lower()
            if any(pattern in error_str for pattern in ['connection', 'network', 'timeout', 'unreachable']):
                self._connection_failures += 1
                logger.warning(f"Ollama connection error: {e}")
                raise ProviderError(
                    f"Ollama connection failed: {e}", provider="ollama", original_error=e
                )

            # Check for model-related errors (non-retryable)
            if any(pattern in error_str for pattern in ['model not found', 'invalid model', 'model not available']):
                logger.error(f"Ollama model error: {e}")
                raise ProviderError(
                    f"Ollama model error: {e}", provider="ollama", original_error=e
                )

            # Preserve ProviderUnavailableError
            if isinstance(e, ProviderUnavailableError):
                raise

            # Log error for debugging
            error_type = ErrorClassifier.classify_error(e)
            logger.warning(f"Ollama chat error (type: {error_type.value}): {e}")

            raise ProviderError(
                f"Ollama chat failed: {e}", provider="ollama", original_error=e
            )

    def embed_texts(
        self, texts: List[str], model: Optional[str] = None, batch_size: int = 10
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts using local Ollama models with retry and circuit breaker protection.

        Args:
            texts: List of texts to embed
            model: Specific embedding model to use
            batch_size: Number of texts to process at once (not strictly needed for local)

        Returns:
            List of embedding vectors

        Raises:
            ProviderError: If Ollama call fails
            ProviderUnavailableError: If Ollama service not available
            CircuitBreakerOpenException: If circuit breaker is open
        """
        self._request_count += 1
        return self._embed_texts_impl(texts=texts, model=model, batch_size=batch_size)

    @retry_with_backoff(get_provider_retry_policy("ollama"))
    def _embed_texts_impl(
        self, texts: List[str], model: Optional[str] = None, batch_size: int = 10
    ) -> List[List[float]]:
        """Internal implementation of embed_texts with retry logic."""
        if not texts:
            return []

        if not self.is_available():
            self._connection_failures += 1
            raise ProviderUnavailableError(
                "Ollama service not available", provider="ollama"
            )

        model = model or self.default_embedding_model
        embeddings = []

        try:
            with self.circuit_breaker:
                # Process texts (Ollama handles one at a time)
                dimension = self.get_embedding_dimension(model)

                for i, text in enumerate(texts):
                    if not text.strip():
                        # Use zero vector for empty text
                        embeddings.append([0.0] * dimension)
                        continue

                    try:
                        response = self.ollama.embeddings(model=model, prompt=text)

                        if "embedding" in response:
                            embeddings.append(response["embedding"])
                        else:
                            # Fallback to zero vector
                            logger.warning(
                                f"No embedding in Ollama response for text: {text[:50]}..."
                            )
                            embeddings.append([0.0] * dimension)

                    except Exception as e:
                        # Handle individual text failures gracefully
                        logger.warning(f"Failed to embed text {i+1}/{len(texts)}: {e}")
                        embeddings.append([0.0] * dimension)

                logger.debug(f"Ollama embeddings successful for {len(texts)} texts")
                return embeddings

        except Exception as e:
            self._error_count += 1

            # Check for connection-related errors
            error_str = str(e).lower()
            if any(pattern in error_str for pattern in ['connection', 'network', 'timeout', 'unreachable']):
                self._connection_failures += 1
                logger.warning(f"Ollama embedding connection error: {e}")

            # Check for model-related errors
            if any(pattern in error_str for pattern in ['model not found', 'invalid model', 'model not available']):
                logger.error(f"Ollama embedding model error: {e}")

            # Log error and provide fallback
            error_type = ErrorClassifier.classify_error(e)
            logger.warning(f"Ollama embeddings error (type: {error_type.value}): {e}")

            # Graceful fallback: return zero vectors for all texts
            dimension = self.get_embedding_dimension(model)
            logger.info(f"Returning zero vectors as fallback for {len(texts)} texts")
            return [[0.0] * dimension for _ in texts]

    def embed_one(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding for single text using local Ollama.

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

        # Common Ollama embedding model dimensions
        dimensions = {
            "nomic-embed-text": 768,
            "all-minilm": 384,
            "mxbai-embed-large": 1024,
        }

        return dimensions.get(model, 768)  # Default dimension

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text. Ollama doesn't provide direct token counting.

        Args:
            text: Text to count tokens for
            model: Model to use for tokenization (not used for estimation)

        Returns:
            Estimated number of tokens
        """
        # Ollama models use various tokenizers, use conservative estimation
        return self.estimate_tokens(text)

    def pull_model(self, model_name: str) -> bool:
        """Pull/download a model to local Ollama instance.

        Args:
            model_name: Name of model to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            self.ollama.pull(model_name)
            logger.info(f"Successfully pulled Ollama model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull Ollama model {model_name}: {e}")
            return False

    def list_models(self) -> List[str]:
        """List available models in local Ollama instance.

        Returns:
            List of model names
        """
        try:
            models = self.ollama.list()
            if "models" in models:
                return [model["name"] for model in models["models"]]
            return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific model.

        Args:
            model: Model name to get info for

        Returns:
            Model information dictionary
        """
        model = model or self.default_llm_model

        try:
            # Try to get model info from Ollama
            models = self.ollama.list()
            if "models" in models:
                for model_info in models["models"]:
                    if model in model_info["name"]:
                        return {
                            "name": model_info["name"],
                            "size": model_info.get("size", 0),
                            "modified_at": model_info.get("modified_at"),
                            "details": model_info.get("details", {}),
                            "provider": "ollama",
                            "deployment": "local",
                        }
        except Exception as e:
            logger.debug(f"Failed to get Ollama model info: {e}")

        # Fallback to basic info
        return {
            "name": model,
            "provider": "ollama",
            "deployment": "local",
            "cost": "free",
            "privacy": "high",
        }

    def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        """Generate streaming chat completion using Ollama.

        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Specific model to use
            temperature: Randomness in generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            Streaming response chunks
        """
        if not self.is_available():
            raise ProviderUnavailableError(
                "Ollama service not available", provider="ollama"
            )

        model = model or self.default_llm_model

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens
            options.update(kwargs)

            # Create streaming response
            stream = self.ollama.chat(
                model=model, messages=messages, stream=True, options=options
            )

            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

        except Exception as e:
            raise ProviderError(
                f"Ollama streaming failed: {e}", provider="ollama", original_error=e
            )

    def _get_metrics(self) -> Dict[str, Any]:
        """Get provider metrics for monitoring."""
        return {
            "requests_total": self._request_count,
            "errors_total": self._error_count,
            "connection_failures_total": self._connection_failures,
            "error_rate": self._error_count / max(1, self._request_count),
            "connection_failure_rate": self._connection_failures / max(1, self._request_count),
        }

    def get_provider_status(self) -> Dict[str, Any]:
        """Get comprehensive provider status including resilience state."""
        return {
            "provider": "ollama",
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
                "ollama_host": self.ollama_host or "localhost:11434",
                "deployment": "local",
            }
        }
