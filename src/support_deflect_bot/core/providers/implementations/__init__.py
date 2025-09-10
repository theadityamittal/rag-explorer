"""Concrete provider implementations for LLM and embedding services."""

import logging

logger = logging.getLogger(__name__)

# Import all provider implementations
from .openai_provider import OpenAIProvider
from .groq_provider import GroqProvider
from .mistral_provider import MistralProvider
from .google_gemini import GoogleGeminiFreeProvider, GoogleGeminiPaidProvider
from .claude_api_provider import ClaudeAPIProvider
from .claude_code_provider import ClaudeCodeProvider
from .ollama_provider import OllamaProvider

# Import registration function
from ..config import register_provider

__all__ = [
    "OpenAIProvider",
    "GroqProvider",
    "MistralProvider",
    "GoogleGeminiFreeProvider",
    "GoogleGeminiPaidProvider",
    "ClaudeAPIProvider",
    "ClaudeCodeProvider",
    "OllamaProvider",
    "register_all_providers",
]


def register_all_providers():
    """Register all available providers with the default registry."""
    providers = [
        ("openai", OpenAIProvider),
        ("groq", GroqProvider),
        ("mistral", MistralProvider),
        ("google_gemini_free", GoogleGeminiFreeProvider),
        ("google_gemini_paid", GoogleGeminiPaidProvider),
        ("claude_api", ClaudeAPIProvider),
        ("claude_code", ClaudeCodeProvider),
        ("ollama", OllamaProvider),
    ]

    registered_count = 0
    for name, provider_class in providers:
        try:
            register_provider(name, provider_class)
            registered_count += 1
            logger.debug(f"Registered provider: {name}")
        except Exception as e:
            logger.warning(f"Failed to register provider {name}: {e}")

    logger.info(f"Registered {registered_count} providers with default registry")
    return registered_count


# Auto-register all providers when module is imported
try:
    register_all_providers()
except Exception as e:
    logger.error(f"Failed to auto-register providers: {e}")
    # Don't fail import if registration fails
