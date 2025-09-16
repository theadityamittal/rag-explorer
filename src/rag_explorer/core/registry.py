"""Simple Provider Registry for RAG Explorer."""

import logging
from typing import Union

from .providers.base import LLMProvider, EmbeddingProvider, CombinedProvider
from .providers.implementations.openai_provider import OpenAIProvider
from .providers.implementations.anthropic_provider import ClaudeAPIProvider
from .providers.implementations.google_gemini import GoogleGeminiProvider
from .providers.implementations.ollama_provider import OllamaProvider

from ..utils.settings import (
    PRIMARY_LLM_PROVIDER,
    PRIMARY_EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    OLLAMA_HOST,
    reload_env
)

logger = logging.getLogger(__name__)


class ProviderNotConfiguredError(Exception):
    """Raised when provider is not properly configured."""
    pass


class ProviderNotAvailableError(Exception):
    """Raised when provider is configured but not available."""
    pass


class ProviderRegistry:
    """Simple provider registry using settings.py configuration.

    Reads current provider settings dynamically and instantiates providers
    on demand. No caching or fallback logic - throws exceptions on failure.
    """

    def __init__(self):
        """Initialize the registry."""
        logger.info("Initialized ProviderRegistry")

    def get_llm_provider(self) -> LLMProvider:
        """Get current LLM provider from settings.

        Returns:
            LLMProvider instance based on PRIMARY_LLM_PROVIDER setting

        Raises:
            ProviderNotConfiguredError: If provider name is invalid
            ProviderNotAvailableError: If provider is configured but unavailable
        """
        provider_name = PRIMARY_LLM_PROVIDER.lower()

        try:
            if provider_name == "openai":
                if not OPENAI_API_KEY:
                    raise ProviderNotConfiguredError("OpenAI API key not configured")
                provider = OpenAIProvider(api_key=OPENAI_API_KEY)

            elif provider_name == "anthropic":
                if not ANTHROPIC_API_KEY:
                    raise ProviderNotConfiguredError("Anthropic API key not configured")
                provider = ClaudeAPIProvider(api_key=ANTHROPIC_API_KEY)

            elif provider_name == "google":
                if not GEMINI_API_KEY:
                    raise ProviderNotConfiguredError("Google API key not configured")
                provider = GoogleGeminiProvider(api_key=GEMINI_API_KEY)

            elif provider_name == "ollama":
                provider = OllamaProvider(host=OLLAMA_HOST)

            else:
                raise ProviderNotConfiguredError(f"Unknown LLM provider: {provider_name}")

            # Test provider availability
            if not provider.is_available():
                raise ProviderNotAvailableError(f"LLM provider {provider_name} is not available")

            logger.debug(f"Created LLM provider: {provider_name}")
            return provider

        except (ProviderNotConfiguredError, ProviderNotAvailableError):
            raise
        except Exception as e:
            raise ProviderNotAvailableError(f"Failed to create LLM provider {provider_name}: {e}")

    def get_embedding_provider(self) -> EmbeddingProvider:
        """Get current embedding provider from settings.

        Returns:
            EmbeddingProvider instance based on PRIMARY_EMBEDDING_PROVIDER setting

        Raises:
            ProviderNotConfiguredError: If provider name is invalid or doesn't support embeddings
            ProviderNotAvailableError: If provider is configured but unavailable
        """
        provider_name = PRIMARY_EMBEDDING_PROVIDER.lower()

        try:
            if provider_name == "openai":
                if not OPENAI_API_KEY:
                    raise ProviderNotConfiguredError("OpenAI API key not configured")
                provider = OpenAIProvider(api_key=OPENAI_API_KEY)

            elif provider_name == "google":
                if not GEMINI_API_KEY:
                    raise ProviderNotConfiguredError("Google API key not configured")
                provider = GoogleGeminiProvider(api_key=GEMINI_API_KEY)

            elif provider_name == "ollama":
                provider = OllamaProvider(host=OLLAMA_HOST)

            elif provider_name == "anthropic":
                raise ProviderNotConfiguredError("Anthropic does not provide embedding services")

            else:
                raise ProviderNotConfiguredError(f"Unknown embedding provider: {provider_name}")

            # Test provider availability
            if not provider.is_available():
                raise ProviderNotAvailableError(f"Embedding provider {provider_name} is not available")

            logger.debug(f"Created embedding provider: {provider_name}")
            return provider

        except (ProviderNotConfiguredError, ProviderNotAvailableError):
            raise
        except Exception as e:
            raise ProviderNotAvailableError(f"Failed to create embedding provider {provider_name}: {e}")

    def get_combined_provider(self) -> CombinedProvider:
        """Get a combined provider that supports both LLM and embeddings.

        Returns:
            CombinedProvider instance

        Raises:
            ProviderNotConfiguredError: If no provider supports both capabilities
            ProviderNotAvailableError: If provider is configured but unavailable
        """
        # Only some providers support both LLM and embeddings
        llm_name = PRIMARY_LLM_PROVIDER.lower()
        embedding_name = PRIMARY_EMBEDDING_PROVIDER.lower()

        if llm_name == embedding_name and llm_name in ["openai", "google", "ollama"]:
            # Same provider for both - return it
            if llm_name == "openai":
                return self.get_llm_provider()
            elif llm_name == "google":
                return self.get_llm_provider()
            elif llm_name == "ollama":
                return self.get_llm_provider()

        raise ProviderNotConfiguredError(
            f"No single provider supports both LLM ({llm_name}) and embeddings ({embedding_name})"
        )

    def reload_settings(self):
        """Force reload of environment variables from .env file.

        Call this after making changes to .env file to pick up new settings.
        """
        reload_env()
        logger.info("Reloaded environment settings")

    def validate_configuration(self) -> dict:
        """Validate current provider configuration.

        Returns:
            Dictionary with validation results for each provider type
        """
        results = {
            "llm": {"provider": PRIMARY_LLM_PROVIDER, "status": "unknown", "error": None},
            "embedding": {"provider": PRIMARY_EMBEDDING_PROVIDER, "status": "unknown", "error": None}
        }

        # Test LLM provider
        try:
            provider = self.get_llm_provider()
            results["llm"]["status"] = "available"
        except Exception as e:
            results["llm"]["status"] = "error"
            results["llm"]["error"] = str(e)

        # Test embedding provider
        try:
            provider = self.get_embedding_provider()
            results["embedding"]["status"] = "available"
        except Exception as e:
            results["embedding"]["status"] = "error"
            results["embedding"]["error"] = str(e)

        return results