"""
⚠️  DEPRECATED: This settings file is deprecated in favor of simple_settings.py

This file is kept for backward compatibility only.
For new projects and simplified usage, please use:
- src/rag_explorer/utils/simple_settings.py
"""

import warnings
from typing import List

# Log deprecation warning
warnings.warn(
    "rag_explorer.utils.settings is deprecated. Use rag_explorer.utils.simple_settings instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export selected items from simple_settings
from .simple_settings import (
    # API Keys
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    
    # Provider settings
    PRIMARY_LLM_PROVIDER,
    PRIMARY_EMBEDDING_PROVIDER,
    
    # Model settings
    OPENAI_LLM_MODEL,
    OPENAI_EMBEDDING_MODEL,
    ANTHROPIC_LLM_MODEL,
    GOOGLE_LLM_MODEL,
    GOOGLE_EMBEDDING_MODEL,
    OLLAMA_LLM_MODEL as OLLAMA_MODEL,
    OLLAMA_EMBEDDING_MODEL as OLLAMA_EMBED_MODEL,
    OLLAMA_HOST,
    
    # RAG settings
    CHUNK_SIZE as MAX_CHARS_PER_CHUNK,
    MIN_CONFIDENCE as ANSWER_MIN_CONF,
    MAX_CHUNKS,
    
    # Paths
    DOCS_FOLDER,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
    
    # Database settings
    DB_POOL_SIZE,
    DB_CONNECTION_TIMEOUT,
    DB_POOL_CLEANUP_INTERVAL,
    DATABASE_QUERY_TIMEOUT,
)

# Import constants for USER_AGENT
from ..constants import DEFAULT_USER_AGENT
USER_AGENT = DEFAULT_USER_AGENT

# Legacy compatibility settings with defaults
import os
from dotenv import load_dotenv

load_dotenv()

# Provider models for backward compatibility
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

# Timeout settings
PROVIDER_TIMEOUT = float(os.getenv("PROVIDER_TIMEOUT", "30.0"))

# Resilience settings for backward compatibility
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
OPENAI_CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("OPENAI_CIRCUIT_BREAKER_THRESHOLD", "5"))
GOOGLE_MAX_RETRIES = int(os.getenv("GOOGLE_MAX_RETRIES", "3"))
GOOGLE_CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("GOOGLE_CIRCUIT_BREAKER_THRESHOLD", "5"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
OLLAMA_CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("OLLAMA_CIRCUIT_BREAKER_THRESHOLD", "3"))

# Web crawling settings for backward compatibility
def _parse_csv(env_var: str, default: str = "") -> List[str]:
    """Parse comma-separated values from environment variables."""
    val = os.getenv(env_var, default)
    return [s.strip() for s in val.split(",") if s.strip()]

ALLOW_HOSTS = set(_parse_csv("ALLOW_HOSTS", "docs.python.org,packaging.python.org,pip.pypa.io,virtualenv.pypa.io,help.sigmacomputing.com"))

# Legacy functions for backward compatibility
def get_configured_providers() -> List[str]:
    """Get list of providers that have API keys configured."""
    providers = []
    if OPENAI_API_KEY:
        providers.append("openai")
    if ANTHROPIC_API_KEY:
        providers.append("anthropic")
    if GOOGLE_API_KEY:
        providers.append("google_gemini")
    if OLLAMA_HOST:
        providers.append("ollama")
    return providers
