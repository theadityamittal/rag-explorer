"""Configuration settings for Support Deflect Bot with multi-provider support."""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# GENERAL APPLICATION SETTINGS
# ============================================================================

APP_NAME = os.getenv("APP_NAME", "Support Deflection Bot")
APP_VERSION = os.getenv("APP_VERSION", "0.2.0")

# ============================================================================
# PROVIDER CONFIGURATION (NEW MULTI-PROVIDER SYSTEM)
# ============================================================================

# Provider Selection Strategy
PROVIDER_STRATEGY = os.getenv("PROVIDER_STRATEGY", "cost_optimized")
# Options: cost_optimized, speed_focused, quality_first, balanced, custom

# Primary Providers (legally compliant defaults)
PRIMARY_LLM_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER", "google_gemini_paid")
PRIMARY_EMBEDDING_PROVIDER = os.getenv(
    "PRIMARY_EMBEDDING_PROVIDER", "google_gemini_paid"
)


# Fallback Provider Chains
def _parse_csv(env_var: str, default: str = "") -> List[str]:
    """Parse comma-separated values from environment variables."""
    val = os.getenv(env_var, default)
    return [s.strip() for s in val.split(",") if s.strip()]


FALLBACK_LLM_PROVIDERS = _parse_csv("FALLBACK_LLM_PROVIDERS", "openai,groq,ollama")
FALLBACK_EMBEDDING_PROVIDERS = _parse_csv(
    "FALLBACK_EMBEDDING_PROVIDERS", "openai,ollama"
)

# ============================================================================
# REGIONAL COMPLIANCE AND LEGAL SETTINGS
# ============================================================================

# Regional Detection and Compliance
USER_REGION = os.getenv(
    "USER_REGION", "auto"
)  # auto-detect or manual override (e.g., 'US', 'EU')
ENFORCE_REGIONAL_COMPLIANCE = (
    os.getenv("ENFORCE_REGIONAL_COMPLIANCE", "true").lower() == "true"
)

# GDPR Compliance Mode
GDPR_COMPLIANCE_MODE = os.getenv(
    "GDPR_COMPLIANCE_MODE", "auto"
)  # auto, strict, disabled
REQUIRE_AI_DISCLOSURE = os.getenv("REQUIRE_AI_DISCLOSURE", "true").lower() == "true"
ENABLE_CONSENT_MANAGEMENT = (
    os.getenv("ENABLE_CONSENT_MANAGEMENT", "true").lower() == "true"
)

# ============================================================================
# COST CONTROL AND BUDGET MANAGEMENT
# ============================================================================

# Budget Controls
MONTHLY_BUDGET_USD = float(os.getenv("MONTHLY_BUDGET_USD", "10.0"))
DAILY_REQUEST_LIMIT = int(os.getenv("DAILY_REQUEST_LIMIT", "200"))
COST_ALERT_THRESHOLD = float(os.getenv("COST_ALERT_THRESHOLD", "0.8"))  # 80% of budget

# Cost Tracking
ENABLE_COST_TRACKING = os.getenv("ENABLE_COST_TRACKING", "true").lower() == "true"
COST_TRACKING_FILE = os.getenv("COST_TRACKING_FILE", "./usage_costs.json")

# ============================================================================
# API KEYS (ALL OPTIONAL - CONFIGURE ONLY WHAT YOU NEED)
# ============================================================================

# Primary Providers (Legally Compliant)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Budget-Friendly Alternatives
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Google Services (Different Compliance for Free vs Paid)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Local/Self-Hosted Options (Optional)
OLLAMA_HOST = os.getenv("OLLAMA_HOST")  # For backward compatibility
CLAUDE_CODE_PATH = os.getenv("CLAUDE_CODE_PATH", "claude")  # Path to Claude Code CLI

# ============================================================================
# PROVIDER-SPECIFIC MODEL SELECTIONS
# ============================================================================

# OpenAI Models (Default/Primary)
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-3.5-turbo")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Alternative Provider Models
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.1-70b-versatile")
MISTRAL_LLM_MODEL = os.getenv("MISTRAL_LLM_MODEL", "mistral-small-latest")
ANTHROPIC_LLM_MODEL = os.getenv("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307")

# Google Models
GOOGLE_LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")

# ============================================================================
# LEGACY OLLAMA SETTINGS (FOR BACKWARD COMPATIBILITY)
# ============================================================================

# Ollama (local LLM + embeddings) - Optional fallback
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")  # Optional; defaults to localhost:11434

# ============================================================================
# VECTOR STORE CONFIGURATION
# ============================================================================

# Vector store (Chroma)
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "knowledge_base")

# ============================================================================
# RAG BEHAVIOR SETTINGS
# ============================================================================

# RAG behavior
ANSWER_MIN_CONF = float(os.getenv("ANSWER_MIN_CONF", "0.25"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "5"))
MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "800"))

# ============================================================================
# DOCUMENTATION SOURCES (FLEXIBLE)
# ============================================================================

# Documentation sources (comma-separated, supports local and remote)
DOCS_SOURCES = _parse_csv("DOCS_SOURCES", "./docs")
# Example: "./docs,/home/user/my-docs,https://my-site.com/docs"

# Remote documentation settings
ENABLE_REMOTE_DOCS = os.getenv("ENABLE_REMOTE_DOCS", "true").lower() == "true"
DOCS_AUTO_REFRESH_HOURS = int(os.getenv("DOCS_AUTO_REFRESH_HOURS", "24"))

# ============================================================================
# WEB CRAWLING CONFIGURATION
# ============================================================================

# Crawl config
USER_AGENT = os.getenv(
    "CRAWL_USER_AGENT",
    "SupportDeflectBot/0.2 (+https://github.com/theadityamittal/support-deflect-bot; contact: theadityamittal@gmail.com)",
)

# Allowed and trusted domains for crawling
ALLOW_HOSTS = set(
    _parse_csv(
        "ALLOW_HOSTS",
        "docs.python.org,packaging.python.org,pip.pypa.io,virtualenv.pypa.io,help.sigmacomputing.com",
    )
)

TRUSTED_DOMAINS = set(_parse_csv("TRUSTED_DOMAINS", "help.sigmacomputing.com"))

# Default crawl seeds
DEFAULT_SEEDS = _parse_csv(
    "DEFAULT_SEEDS",
    "https://docs.python.org/3/faq/index.html,https://docs.python.org/3/library/venv.html",
)

# Crawling limits
CRAWL_DEPTH = int(os.getenv("CRAWL_DEPTH", "1"))
CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "40"))
CRAWL_SAME_DOMAIN = os.getenv("CRAWL_SAME_DOMAIN", "true").lower() == "true"

# ============================================================================
# FILE PATHS AND CACHING
# ============================================================================

# File paths
CRAWL_CACHE_PATH = os.getenv("CRAWL_CACHE_PATH", "./data/crawl_cache.json")
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "./docs")  # Backward compatibility

# ============================================================================
# PROVIDER HEALTH CHECK AND MONITORING
# ============================================================================

# Health check intervals
PROVIDER_HEALTH_CHECK_INTERVAL = int(
    os.getenv("PROVIDER_HEALTH_CHECK_INTERVAL", "300")
)  # 5 minutes
ENABLE_PROVIDER_MONITORING = (
    os.getenv("ENABLE_PROVIDER_MONITORING", "true").lower() == "true"
)

# Failure handling
MAX_PROVIDER_FAILURES = int(os.getenv("MAX_PROVIDER_FAILURES", "3"))
PROVIDER_RETRY_DELAY = int(os.getenv("PROVIDER_RETRY_DELAY", "60"))  # seconds

# ============================================================================
# LOGGING AND DEBUG SETTINGS
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEBUG_PROVIDER_SELECTION = (
    os.getenv("DEBUG_PROVIDER_SELECTION", "false").lower() == "true"
)
LOG_PROVIDER_COSTS = os.getenv("LOG_PROVIDER_COSTS", "true").lower() == "true"

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================


def validate_configuration() -> List[str]:
    """Validate current configuration and return list of warnings/errors."""
    warnings = []

    # Check if at least one LLM provider is configured
    llm_keys = [
        OPENAI_API_KEY,
        ANTHROPIC_API_KEY,
        GROQ_API_KEY,
        MISTRAL_API_KEY,
        GOOGLE_API_KEY,
    ]
    if not any(llm_keys) and not OLLAMA_HOST:
        warnings.append(
            "No LLM provider API keys configured. Add at least one API key."
        )

    # Check budget settings
    if MONTHLY_BUDGET_USD <= 0:
        warnings.append("Monthly budget must be greater than 0")

    if DAILY_REQUEST_LIMIT <= 0:
        warnings.append("Daily request limit must be greater than 0")

    # Check strategy
    valid_strategies = [
        "cost_optimized",
        "speed_focused",
        "quality_first",
        "balanced",
        "custom",
    ]
    if PROVIDER_STRATEGY not in valid_strategies:
        warnings.append(f"Invalid provider strategy: {PROVIDER_STRATEGY}")

    return warnings


def get_configured_providers() -> List[str]:
    """Get list of providers that have API keys configured."""
    providers = []

    if OPENAI_API_KEY:
        providers.append("openai")
    if ANTHROPIC_API_KEY:
        providers.append("anthropic")
    if GROQ_API_KEY:
        providers.append("groq")
    if MISTRAL_API_KEY:
        providers.append("mistral")
    if GOOGLE_API_KEY:
        providers.append("google_gemini")
    if OLLAMA_HOST:
        providers.append("ollama")

    return providers


def estimate_monthly_cost() -> float:
    """Estimate monthly cost based on current settings."""
    # Rough estimation based on daily limits and average token usage
    avg_tokens_per_request = 1500  # Conservative estimate
    requests_per_month = DAILY_REQUEST_LIMIT * 30
    total_tokens_per_month = requests_per_month * avg_tokens_per_request

    # Use OpenAI pricing as baseline (most providers are similar or cheaper)
    cost_per_million_tokens = 0.5  # GPT-3.5 turbo input cost
    estimated_cost = (total_tokens_per_month / 1_000_000) * cost_per_million_tokens

    return min(estimated_cost, MONTHLY_BUDGET_USD)


# ============================================================================
# ENVIRONMENT-SPECIFIC OVERRIDES
# ============================================================================

# Development/Testing overrides
if os.getenv("ENVIRONMENT") == "development":
    DEBUG_PROVIDER_SELECTION = True
    LOG_LEVEL = "DEBUG"
    ENABLE_COST_TRACKING = True

# Production safety overrides
if os.getenv("ENVIRONMENT") == "production":
    ENFORCE_REGIONAL_COMPLIANCE = True
    ENABLE_CONSENT_MANAGEMENT = True
    LOG_PROVIDER_COSTS = True
