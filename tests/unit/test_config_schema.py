"""Unit tests for configuration schema and validation."""

import pytest
from pydantic import ValidationError

from support_deflect_bot.config.schema import (
    ApiKeysConfig,
    DocsConfig,
    RagConfig,
    CrawlConfig,
    ModelOverridesConfig,
    AppConfig
)


class TestApiKeysConfig:
    """Test API keys configuration schema."""
    
    def test_valid_api_keys(self):
        """Test valid API key configuration."""
        config = ApiKeysConfig(
            google_api_key="AIzaSyTest123",
            openai_api_key="sk-test123",
            anthropic_api_key="test-anthropic-key"
        )
        
        assert config.google_api_key == "AIzaSyTest123"
        assert config.openai_api_key == "sk-test123"
        assert config.anthropic_api_key == "test-anthropic-key"
        assert config.groq_api_key is None
        assert config.mistral_api_key is None
    
    def test_empty_string_to_none(self):
        """Test that empty strings are converted to None."""
        config = ApiKeysConfig(
            google_api_key="",
            openai_api_key="valid_key",
            anthropic_api_key=""
        )
        
        assert config.google_api_key is None
        assert config.openai_api_key == "valid_key"
        assert config.anthropic_api_key is None
    
    def test_all_none_allowed(self):
        """Test that all None values are allowed."""
        config = ApiKeysConfig()
        
        assert config.google_api_key is None
        assert config.openai_api_key is None
        assert config.anthropic_api_key is None
        assert config.groq_api_key is None
        assert config.mistral_api_key is None


class TestDocsConfig:
    """Test documentation configuration schema."""
    
    def test_valid_docs_config(self):
        """Test valid documentation configuration."""
        config = DocsConfig(
            local_path="/path/to/docs",
            auto_refresh=False,
            sources=["/path/to/docs", "/path/to/more/docs"]
        )
        
        assert config.local_path == "/path/to/docs"
        assert config.auto_refresh is False
        assert config.sources == ["/path/to/docs", "/path/to/more/docs"]
    
    def test_default_values(self):
        """Test default values for documentation config."""
        config = DocsConfig()
        
        assert config.local_path == "./docs"
        assert config.auto_refresh is True
        assert config.sources == ["./docs"]
    
    def test_path_validation(self):
        """Test path validation."""
        # Empty path should fail
        with pytest.raises(ValidationError) as exc_info:
            DocsConfig(local_path="")
        assert "Local path cannot be empty" in str(exc_info.value)
        
        # Whitespace-only path should fail
        with pytest.raises(ValidationError) as exc_info:
            DocsConfig(local_path="   ")
        assert "Local path cannot be empty" in str(exc_info.value)
    
    def test_path_stripping(self):
        """Test that paths are stripped of whitespace."""
        config = DocsConfig(local_path="  /path/to/docs  ")
        assert config.local_path == "/path/to/docs"


class TestRagConfig:
    """Test RAG configuration schema."""
    
    def test_valid_rag_config(self):
        """Test valid RAG configuration."""
        config = RagConfig(
            confidence_threshold=0.75,
            max_chunks=10,
            max_chars_per_chunk=1200
        )
        
        assert config.confidence_threshold == 0.75
        assert config.max_chunks == 10
        assert config.max_chars_per_chunk == 1200
    
    def test_default_values(self):
        """Test default RAG values."""
        config = RagConfig()
        
        assert config.confidence_threshold == 0.25
        assert config.max_chunks == 5
        assert config.max_chars_per_chunk == 800
    
    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation."""
        # Valid values
        config = RagConfig(confidence_threshold=0.0)
        assert config.confidence_threshold == 0.0
        
        config = RagConfig(confidence_threshold=1.0)
        assert config.confidence_threshold == 1.0
        
        config = RagConfig(confidence_threshold=0.5)
        assert config.confidence_threshold == 0.5
        
        # Invalid values
        with pytest.raises(ValidationError):
            RagConfig(confidence_threshold=-0.1)
        
        with pytest.raises(ValidationError):
            RagConfig(confidence_threshold=1.1)
        
        with pytest.raises(ValidationError):
            RagConfig(confidence_threshold=2.0)
    
    def test_max_chunks_validation(self):
        """Test max chunks validation."""
        # Valid values
        config = RagConfig(max_chunks=1)
        assert config.max_chunks == 1
        
        config = RagConfig(max_chunks=20)
        assert config.max_chunks == 20
        
        # Invalid values
        with pytest.raises(ValidationError):
            RagConfig(max_chunks=0)
        
        with pytest.raises(ValidationError):
            RagConfig(max_chunks=21)
    
    def test_max_chars_validation(self):
        """Test max chars per chunk validation."""
        # Valid values
        config = RagConfig(max_chars_per_chunk=100)
        assert config.max_chars_per_chunk == 100
        
        config = RagConfig(max_chars_per_chunk=5000)
        assert config.max_chars_per_chunk == 5000
        
        # Invalid values
        with pytest.raises(ValidationError):
            RagConfig(max_chars_per_chunk=99)
        
        with pytest.raises(ValidationError):
            RagConfig(max_chars_per_chunk=5001)


class TestCrawlConfig:
    """Test crawl configuration schema."""
    
    def test_valid_crawl_config(self):
        """Test valid crawl configuration."""
        config = CrawlConfig(
            allow_hosts=["example.com", "test.com"],
            trusted_domains=["example.com"],
            default_seeds=["http://example.com", "http://test.com"],
            depth=3,
            max_pages=100,
            same_domain=False,
            user_agent="CustomBot/1.0"
        )
        
        assert config.allow_hosts == ["example.com", "test.com"]
        assert config.trusted_domains == ["example.com"]
        assert config.default_seeds == ["http://example.com", "http://test.com"]
        assert config.depth == 3
        assert config.max_pages == 100
        assert config.same_domain is False
        assert config.user_agent == "CustomBot/1.0"
    
    def test_default_values(self):
        """Test default crawl values."""
        config = CrawlConfig()
        
        assert "docs.python.org" in config.allow_hosts
        assert "docs.python.org" in config.trusted_domains
        assert len(config.default_seeds) >= 1
        assert config.depth == 1
        assert config.max_pages == 40
        assert config.same_domain is True
        assert "SupportDeflectBot" in config.user_agent
    
    def test_depth_validation(self):
        """Test crawl depth validation."""
        # Valid values
        config = CrawlConfig(depth=1)
        assert config.depth == 1
        
        config = CrawlConfig(depth=5)
        assert config.depth == 5
        
        # Invalid values
        with pytest.raises(ValidationError):
            CrawlConfig(depth=0)
        
        with pytest.raises(ValidationError):
            CrawlConfig(depth=6)
    
    def test_max_pages_validation(self):
        """Test max pages validation."""
        # Valid values
        config = CrawlConfig(max_pages=1)
        assert config.max_pages == 1
        
        config = CrawlConfig(max_pages=500)
        assert config.max_pages == 500
        
        # Invalid values
        with pytest.raises(ValidationError):
            CrawlConfig(max_pages=0)
        
        with pytest.raises(ValidationError):
            CrawlConfig(max_pages=501)
    
    def test_url_list_cleaning(self):
        """Test that URL lists are cleaned of empty strings."""
        config = CrawlConfig(
            allow_hosts=["example.com", "", "  ", "test.com", ""],
            trusted_domains=["example.com", "", "  "],
            default_seeds=["http://example.com", "", "http://test.com"]
        )
        
        assert config.allow_hosts == ["example.com", "test.com"]
        assert config.trusted_domains == ["example.com"]
        assert config.default_seeds == ["http://example.com", "http://test.com"]
    
    def test_trusted_domains_subset_validation(self):
        """Test that trusted domains must be subset of allowed hosts."""
        # Valid case - trusted domains are subset of allowed hosts
        config = CrawlConfig(
            allow_hosts=["example.com", "test.com", "other.com"],
            trusted_domains=["example.com", "test.com"]
        )
        assert config.trusted_domains == ["example.com", "test.com"]
        
        # Invalid case - trusted domains contain hosts not in allowed hosts
        with pytest.raises(ValidationError) as exc_info:
            CrawlConfig(
                allow_hosts=["example.com"],
                trusted_domains=["example.com", "unauthorized.com"]
            )
        assert "subset of allowed hosts" in str(exc_info.value)


class TestModelOverridesConfig:
    """Test model overrides configuration schema."""
    
    def test_valid_model_overrides(self):
        """Test valid model overrides."""
        config = ModelOverridesConfig(
            gemini_llm_model="gemini-2.5-flash-lite",
            gemini_embedding_model="gemini-embedding-001",
            openai_llm_model="gpt-4o",
            openai_embedding_model="text-embedding-3-large"
        )
        
        assert config.gemini_llm_model == "gemini-2.5-flash-lite"
        assert config.gemini_embedding_model == "gemini-embedding-001"
        assert config.openai_llm_model == "gpt-4o"
        assert config.openai_embedding_model == "text-embedding-3-large"
    
    def test_default_none_values(self):
        """Test that all overrides default to None."""
        config = ModelOverridesConfig()
        
        assert config.gemini_llm_model is None
        assert config.gemini_embedding_model is None
        assert config.openai_llm_model is None
        assert config.openai_embedding_model is None
    
    def test_empty_string_to_none(self):
        """Test that empty strings are converted to None."""
        config = ModelOverridesConfig(
            gemini_llm_model="",
            openai_llm_model="gpt-4o",
            openai_embedding_model=""
        )
        
        assert config.gemini_llm_model is None
        assert config.openai_llm_model == "gpt-4o"
        assert config.openai_embedding_model is None


class TestAppConfig:
    """Test complete application configuration."""
    
    def test_valid_app_config(self):
        """Test valid complete application configuration."""
        config = AppConfig(
            api_keys=ApiKeysConfig(google_api_key="test_key"),
            docs=DocsConfig(local_path="/test/docs"),
            rag=RagConfig(confidence_threshold=0.5),
            crawl=CrawlConfig(depth=2),
            model_overrides=ModelOverridesConfig(gemini_llm_model="custom-model"),
            primary_llm_provider="openai",
            monthly_budget_usd=20.0
        )
        
        assert config.api_keys.google_api_key == "test_key"
        assert config.docs.local_path == "/test/docs"
        assert config.rag.confidence_threshold == 0.5
        assert config.crawl.depth == 2
        assert config.model_overrides.gemini_llm_model == "custom-model"
        assert config.primary_llm_provider == "openai"
        assert config.monthly_budget_usd == 20.0
    
    def test_default_app_config(self):
        """Test default application configuration."""
        config = AppConfig()
        
        assert config.primary_llm_provider == "google_gemini_paid"
        assert config.primary_embedding_provider == "google_gemini_paid"
        assert config.monthly_budget_usd == 10.0
        
        # Check that nested configs are created with defaults
        assert config.api_keys.google_api_key is None
        assert config.docs.local_path == "./docs"
        assert config.rag.confidence_threshold == 0.25
        assert config.crawl.depth == 1
        assert config.model_overrides.gemini_llm_model is None
    
    def test_get_flat_dict(self):
        """Test getting flattened configuration dictionary."""
        config = AppConfig(
            api_keys=ApiKeysConfig(
                google_api_key="test_google",
                openai_api_key="test_openai"
            ),
            docs=DocsConfig(local_path="/test/docs"),
            rag=RagConfig(confidence_threshold=0.6, max_chunks=8),
            crawl=CrawlConfig(
                allow_hosts=["site1.com", "site2.com"],
                depth=3,
                same_domain=False
            ),
            model_overrides=ModelOverridesConfig(gemini_llm_model="custom"),
            monthly_budget_usd=15.5
        )
        
        flat = config.get_flat_dict()
        
        # Check API keys
        assert flat["GOOGLE_API_KEY"] == "test_google"
        assert flat["OPENAI_API_KEY"] == "test_openai"
        
        # Check docs
        assert flat["DOCS_FOLDER"] == "/test/docs"
        
        # Check RAG
        assert flat["ANSWER_MIN_CONF"] == "0.6"
        assert flat["MAX_CHUNKS"] == "8"
        
        # Check crawl
        assert flat["ALLOW_HOSTS"] == "site1.com,site2.com"
        assert flat["CRAWL_DEPTH"] == "3"
        assert flat["CRAWL_SAME_DOMAIN"] == "false"
        
        # Check model overrides
        assert flat["GOOGLE_LLM_MODEL"] == "custom"
        
        # Check general
        assert flat["MONTHLY_BUDGET_USD"] == "15.5"
        assert flat["PRIMARY_LLM_PROVIDER"] == "google_gemini_paid"
    
    def test_get_flat_dict_excludes_none(self):
        """Test that None values are excluded from flat dictionary."""
        config = AppConfig()  # Default config with mostly None API keys
        
        flat = config.get_flat_dict()
        
        # Should not contain keys for None values
        assert "GOOGLE_API_KEY" not in flat
        assert "OPENAI_API_KEY" not in flat
        assert "GOOGLE_LLM_MODEL" not in flat
        
        # Should contain non-None values
        assert "DOCS_FOLDER" in flat
        assert "ANSWER_MIN_CONF" in flat
    
    def test_budget_validation(self):
        """Test monthly budget validation."""
        # Valid values
        config = AppConfig(monthly_budget_usd=0.0)
        assert config.monthly_budget_usd == 0.0
        
        config = AppConfig(monthly_budget_usd=100.50)
        assert config.monthly_budget_usd == 100.50
        
        # Invalid values
        with pytest.raises(ValidationError):
            AppConfig(monthly_budget_usd=-1.0)
    
    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored (forward compatibility)."""
        # This should not raise an error
        config = AppConfig(
            unknown_field="value",
            api_keys={"unknown_key": "value", "google_api_key": "test"}
        )
        
        # Should have created config with valid fields
        assert config.primary_llm_provider == "google_gemini_paid"
        assert config.api_keys.google_api_key == "test"
        
        # Unknown fields should be ignored
        assert not hasattr(config, 'unknown_field')
    
    def test_validation_on_assignment(self):
        """Test that validation occurs on field assignment."""
        config = AppConfig()
        
        # Valid assignment should work
        config.monthly_budget_usd = 25.0
        assert config.monthly_budget_usd == 25.0
        
        # Invalid assignment should raise error
        with pytest.raises(ValidationError):
            config.monthly_budget_usd = -5.0