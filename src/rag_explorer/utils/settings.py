"""Simple settings loading from environment variables."""

import os
from dotenv import load_dotenv
from typing import List

# Load environment variables from .env file
load_dotenv()

VALID_LLM_PROVIDERS: List[str] = ["ollama", "openai", "anthropic", "google"]
VALID_EMBEDDING_PROVIDERS: List[str] = ["ollama", "openai", "google"]

# Provider validation and normalization
def normalize_provider_name(provider: str) -> str:
    """Normalize provider name to standard format."""
    try:
        if not provider:
            raise ValueError("Provider name cannot be empty.")

        provider = provider.lower()
        if provider in VALID_LLM_PROVIDERS or provider in VALID_EMBEDDING_PROVIDERS:
            return provider
        else:
            raise ValueError(f"Invalid provider name: {provider}")
    except ValueError:
        print(f"Provider '{provider}' is not valid.")
        print(f"Valid LLM providers are: {', '.join(VALID_LLM_PROVIDERS)}")
        print(f"Valid embedding providers are: {', '.join(VALID_EMBEDDING_PROVIDERS)}")
        print("Defaulting to 'ollama'.")
        return "ollama"

# Provider settings
PRIMARY_LLM_PROVIDER: str = normalize_provider_name(os.getenv("PRIMARY_LLM_PROVIDER", "ollama"))
PRIMARY_EMBEDDING_PROVIDER: str = normalize_provider_name(os.getenv("PRIMARY_EMBEDDING_PROVIDER", "ollama"))

# API Keys
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Ollama settings
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")
OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# OpenAI model settings
OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Anthropic model settings
ANTHROPIC_LLM_MODEL: str = os.getenv("ANTHROPIC_LLM_MODEL", "claude-3-7-sonnet-20250219")

# Google model settings
GEMINI_LLM_MODEL: str = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# RAG settings
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.25"))
MAX_CHUNKS: int = int(os.getenv("MAX_CHUNKS", "5"))

# Crawl settings
CRAWL_SOURCES: List[str] = os.getenv("CRAWL_SOURCES", "").split(",") if os.getenv("CRAWL_SOURCES") else []
CRAWL_DEPTH: int = int(os.getenv("CRAWL_DEPTH", "2"))
CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "50"))

# Paths
DOCS_FOLDER: str = os.getenv("DOCS_FOLDER", "./docs")
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "knowledge_base")

# Additional settings for database and timeouts
DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
DB_CONNECTION_TIMEOUT: int = int(os.getenv("DB_CONNECTION_TIMEOUT", "30"))
DB_POOL_CLEANUP_INTERVAL: int = int(os.getenv("DB_POOL_CLEANUP_INTERVAL", "300"))
DATABASE_QUERY_TIMEOUT: int = int(os.getenv("DATABASE_QUERY_TIMEOUT", "120"))
PING_TIMEOUT: int = int(os.getenv("PING_TIMEOUT", "5"))
PING_RETRY_COUNT: int = int(os.getenv("PING_RETRY_COUNT", "3"))
SEARCH_MAX_RESULTS: int = int(os.getenv("SEARCH_MAX_RESULTS", "10"))

# Configuration settings
CONFIG_SHOW_SENSITIVE: bool = os.getenv("CONFIG_SHOW_SENSITIVE", "false").lower() == "true"
METRICS_OUTPUT_FORMAT: str = os.getenv("METRICS_OUTPUT_FORMAT", "table")

def reload_env():
    """Reload environment variables from .env file.

    This function reloads the .env file to pick up any changes
    made to environment variables without restarting the application.
    """
    load_dotenv(override=True)