import os

from dotenv import load_dotenv

load_dotenv()

# General
APP_NAME = os.getenv("APP_NAME", "Support Deflection Bot")
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")

# Ollama (local LLM + embeddings)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
# If you run app in Docker but Ollama on the host, set:
#   OLLAMA_HOST=http://host.docker.internal:11434  (macOS/Windows)
#   OLLAMA_HOST=http://172.17.0.1:11434 or use --add-host for Linux
OLLAMA_HOST = os.getenv("OLLAMA_HOST")  # optional; ollama pkg reads it

# API Provider Models (for multi-provider setup)
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")  # Cost-effective default
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CLAUDE_API_MODEL = os.getenv("CLAUDE_API_MODEL", "claude-3-haiku-20240307")  # Most cost-effective
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")

# Claude Code (subprocess fallback)
CLAUDE_CODE_PATH = os.getenv("CLAUDE_CODE_PATH", "claude")  # Path to Claude Code executable

# API Keys (load from environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Cost Control Settings
MONTHLY_BUDGET_USD = float(os.getenv("MONTHLY_BUDGET_USD", "10.0"))
COST_ALERT_THRESHOLD = float(os.getenv("COST_ALERT_THRESHOLD", "0.8"))  # 80% of budget
ENABLE_COST_TRACKING = os.getenv("ENABLE_COST_TRACKING", "true").lower() == "true"

# Provider Selection Strategy
DEFAULT_STRATEGY = os.getenv("DEFAULT_PROVIDER_STRATEGY", "cost_optimized")
REGIONAL_COMPLIANCE = os.getenv("REGIONAL_COMPLIANCE", "true").lower() == "true"

# Vector store (Chroma)
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "knowledge_base")

# RAG behavior
ANSWER_MIN_CONF = float(os.getenv("ANSWER_MIN_CONF", "0.20"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "5"))
MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "800"))


def _csv(name: str, default: str = ""):
    val = os.getenv(name, default)
    return [s.strip() for s in val.split(",") if s.strip()]


# Crawl config
USER_AGENT = os.getenv(
    "CRAWL_USER_AGENT",
    "SupportDeflectBot/0.1 (+https://example.local; contact: you@example.com)",
)
ALLOW_HOSTS = set(
    _csv(
        "ALLOW_HOSTS",
        "docs.python.org,packaging.python.org,pip.pypa.io,virtualenv.pypa.io,help.sigmacomputing.com",
    )
)
TRUSTED_DOMAINS = set(
    _csv(
        "TRUSTED_DOMAINS",
        "help.sigmacomputing.com",
    )
)
DEFAULT_SEEDS = _csv(
    "DEFAULT_SEEDS",
    "https://docs.python.org/3/faq/index.html,"
    "https://docs.python.org/3/library/venv.html",
)
CRAWL_DEPTH = int(os.getenv("CRAWL_DEPTH", "1"))
CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "40"))
CRAWL_SAME_DOMAIN = os.getenv("CRAWL_SAME_DOMAIN", "true").lower() == "true"

# File paths
CRAWL_CACHE_PATH = os.getenv("CRAWL_CACHE_PATH", "./data/crawl_cache.json")
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "./docs")
