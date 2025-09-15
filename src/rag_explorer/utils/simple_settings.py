"""Simple settings loading from environment variables."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Provider settings
PRIMARY_LLM_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER", "ollama")
PRIMARY_EMBEDDING_PROVIDER = os.getenv("PRIMARY_EMBEDDING_PROVIDER", "ollama")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ollama settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# OpenAI model settings
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Anthropic model settings
ANTHROPIC_LLM_MODEL = os.getenv("ANTHROPIC_LLM_MODEL", "claude-3-5-sonnet-20241022")

# Google model settings
GOOGLE_LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-1.5-flash")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")

# RAG settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.25"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "5"))

# Crawl settings
CRAWL_SOURCES = os.getenv("CRAWL_SOURCES", "").split(",") if os.getenv("CRAWL_SOURCES") else []
CRAWL_DEPTH = int(os.getenv("CRAWL_DEPTH", "2"))
CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "50"))
CRAWL_SAME_DOMAIN = os.getenv("CRAWL_SAME_DOMAIN", "true").lower() == "true"

# Search settings
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))

# Configure settings
CONFIG_INTERACTIVE_MODE = os.getenv("CONFIG_INTERACTIVE_MODE", "true").lower() == "true"
CONFIG_SHOW_SENSITIVE = os.getenv("CONFIG_SHOW_SENSITIVE", "false").lower() == "true"

# Metrics settings
METRICS_OUTPUT_FORMAT = os.getenv("METRICS_OUTPUT_FORMAT", "table")

# Ping settings
PING_TIMEOUT = int(os.getenv("PING_TIMEOUT", "30"))
PING_RETRY_COUNT = int(os.getenv("PING_RETRY_COUNT", "3"))

# Paths
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "./docs")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "knowledge_base")

# Database connection settings (minimal defaults for simple mode)
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
DB_CONNECTION_TIMEOUT = int(os.getenv("DB_CONNECTION_TIMEOUT", "30"))
DB_POOL_CLEANUP_INTERVAL = int(os.getenv("DB_POOL_CLEANUP_INTERVAL", "300"))
DATABASE_QUERY_TIMEOUT = int(os.getenv("DATABASE_QUERY_TIMEOUT", "30"))

# Provider validation and normalization
def normalize_provider_name(provider: str) -> str:
    """Normalize provider name to standard format."""
    if not provider:
        return "ollama"  # Default fallback

    provider = provider.lower().strip()

    # Handle aliases
    provider_aliases = {
        "google_gemini": "google",
        "gemini": "google",
        "claude": "anthropic",
        "gpt": "openai",
        "chatgpt": "openai"
    }

    provider = provider_aliases.get(provider, provider)

    # Validate against supported providers
    valid_providers = {"ollama", "openai", "anthropic", "google"}
    if provider not in valid_providers:
        raise ValueError(f"Unsupported provider: {provider}. Valid options: {', '.join(sorted(valid_providers))}")

    return provider

def get_valid_llm_providers():
    """Get list of valid LLM providers."""
    return ["ollama", "openai", "anthropic", "google"]

def get_valid_embedding_providers():
    """Get list of valid embedding providers."""
    return ["ollama", "openai", "google"]  # Note: Anthropic doesn't provide embeddings

# Apply normalization to loaded settings
PRIMARY_LLM_PROVIDER = normalize_provider_name(PRIMARY_LLM_PROVIDER)
PRIMARY_EMBEDDING_PROVIDER = normalize_provider_name(PRIMARY_EMBEDDING_PROVIDER)
