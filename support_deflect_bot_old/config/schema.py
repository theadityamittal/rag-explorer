"""Configuration schema definitions using Pydantic."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator, validator


class ApiKeysConfig(BaseModel):
    """API keys for various providers."""

    google_api_key: Optional[str] = Field(None, description="Google Gemini API key")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    groq_api_key: Optional[str] = Field(None, description="Groq API key")
    mistral_api_key: Optional[str] = Field(None, description="Mistral API key")

    @validator("*", pre=True)
    def empty_str_to_none(cls, v):
        return None if v == "" else v


class DocsConfig(BaseModel):
    """Documentation repository configuration."""

    local_path: str = Field("./docs", description="Path to local documentation folder")
    auto_refresh: bool = Field(True, description="Automatically refresh documentation")
    sources: List[str] = Field(
        default_factory=lambda: ["./docs"], description="List of documentation sources"
    )

    @validator("local_path")
    def validate_local_path(cls, v):
        if not v or not v.strip():
            raise ValueError("Local path cannot be empty")
        return v.strip()


class RagConfig(BaseModel):
    """RAG (Retrieval-Augmented Generation) configuration."""

    confidence_threshold: float = Field(
        0.25, ge=0.0, le=1.0, description="Minimum confidence threshold for answers"
    )
    max_chunks: int = Field(
        5, ge=1, le=20, description="Maximum number of chunks to retrieve"
    )
    max_chars_per_chunk: int = Field(
        800, ge=100, le=5000, description="Maximum characters per chunk"
    )

    @validator("confidence_threshold")
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return v


class CrawlConfig(BaseModel):
    """Web crawling configuration."""

    allow_hosts: List[str] = Field(
        default_factory=lambda: [
            "docs.python.org",
            "packaging.python.org",
            "pip.pypa.io",
        ],
        description="Allowed hosts for crawling",
    )
    trusted_domains: List[str] = Field(
        default_factory=lambda: ["docs.python.org"],
        description="Trusted domains (subset of allowed hosts for sensitive operations)",
    )
    default_seeds: List[str] = Field(
        default_factory=lambda: [
            "https://docs.python.org/3/faq/index.html",
            "https://docs.python.org/3/library/venv.html",
        ],
        description="Default seed URLs for crawling",
    )
    depth: int = Field(1, ge=1, le=5, description="Maximum crawl depth")
    max_pages: int = Field(
        40, ge=1, le=500, description="Maximum number of pages to crawl"
    )
    same_domain: bool = Field(True, description="Restrict crawling to same domain")
    user_agent: str = Field(
        "SupportDeflectBot/0.2 (+https://github.com/theadityamittal/support-deflect-bot)",
        description="User agent string for web requests",
    )

    @validator("allow_hosts", "trusted_domains", "default_seeds")
    def validate_url_lists(cls, v):
        return [item.strip() for item in v if item.strip()]

    @model_validator(mode="after")
    def validate_trusted_domains_subset(self):
        allow_hosts = set(self.allow_hosts or [])
        trusted_domains = set(self.trusted_domains or [])

        if not trusted_domains.issubset(allow_hosts):
            extra_domains = trusted_domains - allow_hosts
            raise ValueError(
                f"Trusted domains must be subset of allowed hosts. Extra domains: {extra_domains}"
            )

        return self


class ModelOverridesConfig(BaseModel):
    """Model overrides for specific providers."""

    gemini_llm_model: Optional[str] = Field(
        None, description="Override Gemini LLM model"
    )
    gemini_embedding_model: Optional[str] = Field(
        None, description="Override Gemini embedding model"
    )
    openai_llm_model: Optional[str] = Field(
        None, description="Override OpenAI LLM model"
    )
    openai_embedding_model: Optional[str] = Field(
        None, description="Override OpenAI embedding model"
    )

    @validator("*", pre=True)
    def empty_str_to_none(cls, v):
        return None if v == "" else v


class AppConfig(BaseModel):
    """Complete application configuration."""

    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    docs: DocsConfig = Field(default_factory=DocsConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    crawl: CrawlConfig = Field(default_factory=CrawlConfig)
    model_overrides: ModelOverridesConfig = Field(default_factory=ModelOverridesConfig)

    # General settings
    primary_llm_provider: str = Field(
        "google_gemini_paid", description="Primary LLM provider"
    )
    primary_embedding_provider: str = Field(
        "google_gemini_paid", description="Primary embedding provider"
    )
    monthly_budget_usd: float = Field(
        10.0, ge=0.0, description="Monthly budget limit in USD"
    )

    class Config:
        """Pydantic config."""

        extra = "ignore"  # Ignore extra fields for forward compatibility
        validate_assignment = True  # Validate on assignment
        use_enum_values = True

    def get_flat_dict(self) -> Dict[str, Any]:
        """Get flattened configuration dictionary for environment variables."""
        flat = {}

        # API Keys
        if self.api_keys.google_api_key:
            flat["GOOGLE_API_KEY"] = self.api_keys.google_api_key
        if self.api_keys.openai_api_key:
            flat["OPENAI_API_KEY"] = self.api_keys.openai_api_key
        if self.api_keys.anthropic_api_key:
            flat["ANTHROPIC_API_KEY"] = self.api_keys.anthropic_api_key
        if self.api_keys.groq_api_key:
            flat["GROQ_API_KEY"] = self.api_keys.groq_api_key
        if self.api_keys.mistral_api_key:
            flat["MISTRAL_API_KEY"] = self.api_keys.mistral_api_key

        # Docs
        flat["DOCS_FOLDER"] = self.docs.local_path

        # RAG
        flat["ANSWER_MIN_CONF"] = str(self.rag.confidence_threshold)
        flat["MAX_CHUNKS"] = str(self.rag.max_chunks)
        flat["MAX_CHARS_PER_CHUNK"] = str(self.rag.max_chars_per_chunk)

        # Crawl
        flat["ALLOW_HOSTS"] = ",".join(self.crawl.allow_hosts)
        flat["TRUSTED_DOMAINS"] = ",".join(self.crawl.trusted_domains)
        flat["DEFAULT_SEEDS"] = ",".join(self.crawl.default_seeds)
        flat["CRAWL_DEPTH"] = str(self.crawl.depth)
        flat["CRAWL_MAX_PAGES"] = str(self.crawl.max_pages)
        flat["CRAWL_SAME_DOMAIN"] = str(self.crawl.same_domain).lower()
        flat["CRAWL_USER_AGENT"] = self.crawl.user_agent

        # Model overrides
        if self.model_overrides.gemini_llm_model:
            flat["GOOGLE_LLM_MODEL"] = self.model_overrides.gemini_llm_model
        if self.model_overrides.gemini_embedding_model:
            flat["GOOGLE_EMBEDDING_MODEL"] = self.model_overrides.gemini_embedding_model
        if self.model_overrides.openai_llm_model:
            flat["OPENAI_LLM_MODEL"] = self.model_overrides.openai_llm_model
        if self.model_overrides.openai_embedding_model:
            flat["OPENAI_EMBEDDING_MODEL"] = self.model_overrides.openai_embedding_model

        # General
        flat["PRIMARY_LLM_PROVIDER"] = self.primary_llm_provider
        flat["PRIMARY_EMBEDDING_PROVIDER"] = self.primary_embedding_provider
        flat["MONTHLY_BUDGET_USD"] = str(self.monthly_budget_usd)

        return flat
