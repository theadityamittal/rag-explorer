"""Unified Embedding Service for RAG Explorer engine."""

import logging
from typing import List, Optional
from rag_explorer.core.registry import ProviderRegistry, ProviderNotConfiguredError, ProviderNotAvailableError
from rag_explorer.core.providers.base import ProviderError

logger = logging.getLogger(__name__)


class UnifiedEmbeddingService:
    """Unified embedding service that handles text embedding generation."""

    def __init__(self):
        """Initialize the embedding service."""
        self.registry = ProviderRegistry()
        logger.debug("Initialized UnifiedEmbeddingService")

    def generate_embeddings(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embeddings for a single text.

        Args:
            text: Text to generate embeddings for
            model: Optional specific model to use

        Returns:
            List of floating-point numbers representing the embedding vector

        Raises:
            ConnectionError: If embedding provider is not configured properly
            ValueError: If text is empty or invalid
            RuntimeError: If embedding generation fails
        """
        if not text:
            raise ValueError("Text cannot be empty")

        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty after cleaning")

        try:
            # Get embedding provider
            embedding_provider = self._get_embedding_provider()

            # Generate embedding
            embedding = embedding_provider.embed_one(text, model=model)

            if not embedding:
                raise RuntimeError("Embedding generation returned empty result")

            if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
                raise RuntimeError("Invalid embedding format returned from provider")

            logger.debug(f"Generated embedding of dimension {len(embedding)} for text: {text[:50]}...")
            return embedding

        except (ProviderNotConfiguredError, ProviderNotAvailableError) as e:
            raise ConnectionError(str(e))
        except ValueError as e:
            # Re-raise ValueError (input validation)
            raise
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

    def generate_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 10
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to generate embeddings for
            model: Optional specific model to use
            batch_size: Number of texts to process at once

        Returns:
            List of embedding vectors, one for each input text

        Raises:
            ConnectionError: If embedding provider is not configured properly
            ValueError: If texts list is empty or contains invalid entries
            RuntimeError: If embedding generation fails
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        if not isinstance(texts, list):
            raise ValueError("Texts must be a list")

        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        # Validate and clean texts
        cleaned_texts = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string")

            cleaned = text.strip()
            if not cleaned:
                raise ValueError(f"Text at index {i} cannot be empty")

            cleaned_texts.append(cleaned)

        try:
            # Get embedding provider
            embedding_provider = self._get_embedding_provider()

            # Generate embeddings in batches
            embeddings = embedding_provider.embed_texts(
                cleaned_texts,
                model=model,
                batch_size=batch_size
            )

            if not embeddings:
                raise RuntimeError("Batch embedding generation returned empty result")

            if len(embeddings) != len(cleaned_texts):
                raise RuntimeError(f"Embedding count mismatch: expected {len(cleaned_texts)}, got {len(embeddings)}")

            # Validate each embedding
            for i, embedding in enumerate(embeddings):
                if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
                    raise RuntimeError(f"Invalid embedding format at index {i}")

            logger.info(f"Generated {len(embeddings)} embeddings in batches of {batch_size}")
            return embeddings

        except (ProviderNotConfiguredError, ProviderNotAvailableError) as e:
            raise ConnectionError(str(e))
        except ValueError as e:
            # Re-raise ValueError (input validation)
            raise
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}")

    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get the dimension of embeddings produced by the current provider.

        Args:
            model: Optional specific model to check

        Returns:
            Embedding vector dimension

        Raises:
            ConnectionError: If embedding provider is not configured properly
            RuntimeError: If dimension query fails
        """
        try:
            embedding_provider = self._get_embedding_provider()
            dimension = embedding_provider.get_embedding_dimension(model=model)

            if dimension <= 0:
                raise RuntimeError("Invalid embedding dimension returned from provider")

            logger.debug(f"Embedding dimension: {dimension}")
            return dimension

        except (ProviderNotConfiguredError, ProviderNotAvailableError) as e:
            raise ConnectionError(str(e))
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {e}")
            raise RuntimeError(f"Failed to get embedding dimension: {e}")

    def test_embedding(self, test_text: str = "This is a test.") -> dict:
        """Test embedding generation to verify provider is working.

        Args:
            test_text: Text to use for testing

        Returns:
            Dictionary with test results

        Raises:
            ConnectionError: If embedding provider is not configured properly
        """
        try:
            # Test single embedding
            start_time = __import__('time').time()
            embedding = self.generate_embeddings(test_text)
            single_time = __import__('time').time() - start_time

            # Test batch embedding
            start_time = __import__('time').time()
            batch_embeddings = self.generate_embeddings_batch([test_text, test_text])
            batch_time = __import__('time').time() - start_time

            # Get dimension
            dimension = self.get_embedding_dimension()

            return {
                "status": "success",
                "single_embedding": {
                    "dimension": len(embedding),
                    "time_seconds": round(single_time, 3),
                    "sample_values": embedding[:5] if len(embedding) >= 5 else embedding
                },
                "batch_embedding": {
                    "count": len(batch_embeddings),
                    "time_seconds": round(batch_time, 3),
                    "dimension_consistent": all(len(emb) == dimension for emb in batch_embeddings)
                },
                "provider_dimension": dimension,
                "test_text": test_text
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "test_text": test_text
            }

    def _get_embedding_provider(self):
        """Get embedding provider with clear error messages."""
        try:
            return self.registry.get_embedding_provider()
        except ProviderNotConfiguredError as e:
            # Extract provider name from error and provide clear guidance
            provider_name = str(e).split()[0].lower() if str(e) else "unknown"
            if "openai" in provider_name:
                raise ConnectionError("OpenAI provider not configured. Please set OPENAI_API_KEY environment variable or change PRIMARY_EMBEDDING_PROVIDER setting.")
            elif "google" in provider_name:
                raise ConnectionError("Google provider not configured. Please set GEMINI_API_KEY environment variable or change PRIMARY_EMBEDDING_PROVIDER setting.")
            elif "ollama" in provider_name:
                raise ConnectionError("Ollama provider not configured. Please ensure Ollama is running or change PRIMARY_EMBEDDING_PROVIDER setting.")
            else:
                raise ConnectionError(f"Embedding provider not configured: {e}")
        except ProviderNotAvailableError as e:
            raise ConnectionError(f"Embedding provider is not available: {e}")

    def get_provider_info(self) -> dict:
        """Get information about the current embedding provider.

        Returns:
            Dictionary with provider information
        """
        try:
            provider = self._get_embedding_provider()
            config = provider.get_config()

            return {
                "name": config.name,
                "type": config.provider_type.value,
                "requires_api_key": config.requires_api_key,
                "is_available": provider.is_available()
            }

        except Exception as e:
            return {
                "error": str(e),
                "is_available": False
            }