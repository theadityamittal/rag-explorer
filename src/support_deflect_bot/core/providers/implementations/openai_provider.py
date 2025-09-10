"""OpenAI provider implementation with GPT models and embeddings."""

import logging
import time
from typing import List, Optional, Dict, Any

from ..base import (
    CombinedProvider,
    ProviderConfig,
    ProviderType,
    ProviderTier,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(CombinedProvider):
    """OpenAI provider with GPT models and embeddings - Primary legally compliant provider."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            **kwargs: Additional configuration options
        """
        try:
            import openai

            self.openai = openai
        except ImportError:
            raise ProviderUnavailableError(
                "OpenAI SDK not available. Install with: pip install openai",
                provider="openai",
            )

        super().__init__(api_key=api_key, **kwargs)

        # Initialize OpenAI client
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            # Try to use default client (environment OPENAI_API_KEY)
            try:
                self.client = openai.OpenAI()
            except Exception:
                self.client = None

        # Default models from settings
        from ....utils.settings import OPENAI_LLM_MODEL, OPENAI_EMBEDDING_MODEL

        self.default_llm_model = OPENAI_LLM_MODEL
        self.default_embedding_model = OPENAI_EMBEDDING_MODEL

        logger.info(
            f"Initialized OpenAI provider with models: {self.default_llm_model}, {self.default_embedding_model}"
        )

    def get_config(self) -> ProviderConfig:
        """Get OpenAI provider configuration."""
        return ProviderConfig(
            name="OpenAI",
            provider_type=ProviderType.BOTH,
            cost_per_million_tokens_input=0.50,  # GPT-3.5-turbo input
            cost_per_million_tokens_output=1.50,  # GPT-3.5-turbo output
            max_context_length=16000,  # GPT-3.5-turbo-16k
            rate_limit_rpm=3500,  # Tier 1 rate limits
            rate_limit_tpm=90000,
            supports_streaming=True,
            requires_api_key=True,
            tier=ProviderTier.PAID,
            regions_supported=["global"],  # Works worldwide
            gdpr_compliant=True,  # GDPR compliant
            models_available=[
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o",
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
        )

    def is_available(self) -> bool:
        """Check if OpenAI provider is available and properly configured."""
        if not self.client:
            return False

        try:
            # Test with a minimal API call
            response = self.client.models.list()
            return response is not None
        except Exception as e:
            logger.debug(f"OpenAI availability check failed: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            if not self.client:
                return {
                    "status": "unhealthy",
                    "error": "OpenAI client not initialized",
                    "provider": "openai",
                }

            # Test API connectivity
            start_time = time.time()
            models = self.client.models.list()
            response_time = time.time() - start_time

            # Check if required models are available
            available_models = [model.id for model in models.data]
            has_llm_model = self.default_llm_model in available_models
            has_embedding_model = self.default_embedding_model in available_models

            return {
                "status": (
                    "healthy" if (has_llm_model and has_embedding_model) else "degraded"
                ),
                "response_time_ms": round(response_time * 1000, 2),
                "models_available": len(available_models),
                "default_llm_available": has_llm_model,
                "default_embedding_available": has_embedding_model,
                "provider": "openai",
                "timestamp": time.time(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": "openai",
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
        """Generate chat completion using OpenAI GPT models.

        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Specific model to use (defaults to configured model)
            temperature: Randomness in generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI API parameters

        Returns:
            Generated response text

        Raises:
            ProviderError: If API call fails
            ProviderRateLimitError: If rate limit exceeded
        """
        if not self.client:
            raise ProviderUnavailableError(
                "OpenAI client not available", provider="openai"
            )

        model = model or self.default_llm_model

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            return response.choices[0].message.content.strip()

        except self.openai.RateLimitError as e:
            raise ProviderRateLimitError(
                f"OpenAI rate limit exceeded: {e}", provider="openai", original_error=e
            )
        except self.openai.APIError as e:
            raise ProviderError(
                f"OpenAI API error: {e}", provider="openai", original_error=e
            )
        except Exception as e:
            raise ProviderError(
                f"OpenAI chat failed: {e}", provider="openai", original_error=e
            )

    def embed_texts(
        self, texts: List[str], model: Optional[str] = None, batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI.

        Args:
            texts: List of texts to embed
            model: Specific embedding model to use
            batch_size: Number of texts to process at once (OpenAI supports large batches)

        Returns:
            List of embedding vectors

        Raises:
            ProviderError: If API call fails
        """
        if not self.client:
            raise ProviderUnavailableError(
                "OpenAI client not available", provider="openai"
            )

        if not texts:
            return []

        model = model or self.default_embedding_model
        embeddings = []

        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                response = self.client.embeddings.create(model=model, input=batch)

                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)

            return embeddings

        except self.openai.RateLimitError as e:
            raise ProviderRateLimitError(
                f"OpenAI rate limit exceeded: {e}", provider="openai", original_error=e
            )
        except self.openai.APIError as e:
            raise ProviderError(
                f"OpenAI embeddings API error: {e}", provider="openai", original_error=e
            )
        except Exception as e:
            raise ProviderError(
                f"OpenAI embeddings failed: {e}", provider="openai", original_error=e
            )

    def embed_one(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding for single text using OpenAI.

        Args:
            text: Text to embed
            model: Specific embedding model to use

        Returns:
            Embedding vector
        """
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.get_embedding_dimension(model)

        embeddings = self.embed_texts([text], model=model)
        return embeddings[0] if embeddings else [0.0] * 1536  # Default dimension

    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get embedding dimension for the specified model.

        Args:
            model: Embedding model name

        Returns:
            Embedding vector dimension
        """
        model = model or self.default_embedding_model

        # OpenAI embedding model dimensions
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        return dimensions.get(model, 1536)  # Default to ada-002 dimension

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text using tiktoken for accurate counting.

        Args:
            text: Text to count tokens for
            model: Model to use for tokenization

        Returns:
            Number of tokens
        """
        model = model or self.default_llm_model

        try:
            import tiktoken

            # Get encoding for the model
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))

        except ImportError:
            logger.warning("tiktoken not available, using estimation")
            return self.estimate_tokens(text)
        except Exception as e:
            logger.warning(f"tiktoken failed: {e}, using estimation")
            return self.estimate_tokens(text)
