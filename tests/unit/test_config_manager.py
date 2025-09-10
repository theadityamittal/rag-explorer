"""Unit tests for configuration manager."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from support_deflect_bot.config.manager import ConfigurationManager
from support_deflect_bot.config.schema import AppConfig, ApiKeysConfig, DocsConfig, RagConfig, CrawlConfig


class TestConfigurationManager:
    """Test cases for ConfigurationManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary directory for config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.json"
        self.manager = ConfigurationManager(str(self.config_file))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_config_defaults(self):
        """Test loading default configuration when no file exists."""
        config = self.manager.load_config()
        
        assert isinstance(config, AppConfig)
        assert config.primary_llm_provider == "google_gemini_paid"
        assert config.primary_embedding_provider == "google_gemini_paid"
        assert config.rag.confidence_threshold == 0.25
        assert config.rag.max_chunks == 5
        assert config.crawl.depth == 1
        assert config.crawl.max_pages == 40
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration from file."""
        # Create test config
        config = AppConfig(
            api_keys=ApiKeysConfig(google_api_key="test_key_123"),
            docs=DocsConfig(local_path="/test/docs"),
            rag=RagConfig(confidence_threshold=0.5, max_chunks=10),
            monthly_budget_usd=25.0
        )
        
        # Save config
        self.manager.save_config(config)
        
        # Verify file exists
        assert self.config_file.exists()
        
        # Load config back
        loaded_config = self.manager.load_config()
        
        assert loaded_config.api_keys.google_api_key == "test_key_123"
        assert loaded_config.docs.local_path == "/test/docs"
        assert loaded_config.rag.confidence_threshold == 0.5
        assert loaded_config.rag.max_chunks == 10
        assert loaded_config.monthly_budget_usd == 25.0
    
    def test_load_config_with_file_and_env(self):
        """Test config loading with both file and environment variables."""
        # Create file config
        file_config = {
            "api_keys": {"google_api_key": "file_key"},
            "rag": {"confidence_threshold": 0.3}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(file_config, f)
        
        # Set environment variables (should override file)
        with patch.dict(os.environ, {
            'GOOGLE_API_KEY': 'env_key',
            'ANSWER_MIN_CONF': '0.7',
            'DOCS_FOLDER': '/env/docs'
        }):
            config = self.manager.load_config()
        
        # Environment should override file
        assert config.api_keys.google_api_key == "env_key"
        assert config.rag.confidence_threshold == 0.7
        assert config.docs.local_path == "/env/docs"
    
    def test_update_config(self):
        """Test updating configuration values."""
        # Load initial config
        config = self.manager.load_config()
        
        # Update config
        updated = self.manager.update_config(
            **{
                "api_keys.openai_api_key": "new_openai_key",
                "rag.confidence_threshold": 0.8,
                "docs.local_path": "/updated/docs"
            }
        )
        
        assert updated.api_keys.openai_api_key == "new_openai_key"
        assert updated.rag.confidence_threshold == 0.8
        assert updated.docs.local_path == "/updated/docs"
    
    def test_set_api_key(self):
        """Test setting API keys for specific providers."""
        self.manager.set_api_key("google", "test_google_key")
        self.manager.set_api_key("openai", "test_openai_key")
        
        config = self.manager.get_config()
        assert config.api_keys.google_api_key == "test_google_key"
        assert config.api_keys.openai_api_key == "test_openai_key"
    
    def test_set_docs_path(self):
        """Test setting documentation path."""
        self.manager.set_docs_path("/custom/docs/path")
        
        config = self.manager.get_config()
        assert config.docs.local_path == "/custom/docs/path"
    
    def test_set_rag_config(self):
        """Test setting RAG configuration."""
        self.manager.set_rag_config(
            confidence_threshold=0.6,
            max_chunks=8,
            max_chars_per_chunk=1200
        )
        
        config = self.manager.get_config()
        assert config.rag.confidence_threshold == 0.6
        assert config.rag.max_chunks == 8
        assert config.rag.max_chars_per_chunk == 1200
    
    def test_set_crawl_config(self):
        """Test setting crawl configuration."""
        self.manager.set_crawl_config(
            allow_hosts=["example.com", "test.com"],
            depth=3,
            max_pages=100
        )
        
        config = self.manager.get_config()
        assert config.crawl.allow_hosts == ["example.com", "test.com"]
        assert config.crawl.depth == 3
        assert config.crawl.max_pages == 100
    
    def test_get_env_vars(self):
        """Test getting environment variables dictionary."""
        config = AppConfig(
            api_keys=ApiKeysConfig(
                google_api_key="test_google",
                openai_api_key="test_openai"
            ),
            docs=DocsConfig(local_path="/test/docs"),
            rag=RagConfig(confidence_threshold=0.4, max_chunks=7)
        )
        
        self.manager._config = config
        env_vars = self.manager.get_env_vars()
        
        assert env_vars["GOOGLE_API_KEY"] == "test_google"
        assert env_vars["OPENAI_API_KEY"] == "test_openai"
        assert env_vars["DOCS_FOLDER"] == "/test/docs"
        assert env_vars["ANSWER_MIN_CONF"] == "0.4"
        assert env_vars["MAX_CHUNKS"] == "7"
    
    def test_export_env_file(self):
        """Test exporting configuration as .env file."""
        config = AppConfig(
            api_keys=ApiKeysConfig(google_api_key="test_key"),
            docs=DocsConfig(local_path="/test/docs"),
            crawl=CrawlConfig(allow_hosts=["site1.com", "site2.com"])
        )
        
        self.manager._config = config
        env_file = Path(self.temp_dir) / "test.env"
        
        self.manager.export_env_file(str(env_file))
        
        # Verify file was created
        assert env_file.exists()
        
        # Check content
        content = env_file.read_text()
        assert "GOOGLE_API_KEY=test_key" in content
        assert "DOCS_FOLDER=/test/docs" in content
        assert "ALLOW_HOSTS=\"site1.com,site2.com\"" in content  # Should be quoted
    
    def test_validate_config(self):
        """Test configuration validation."""
        config = AppConfig(
            api_keys=ApiKeysConfig(google_api_key="test_key"),
            docs=DocsConfig(local_path="/nonexistent/path"),
            rag=RagConfig(confidence_threshold=0.05)  # Very low threshold
        )
        
        self.manager._config = config
        validation = self.manager.validate_config()
        
        assert validation["valid"] is True  # Should be valid despite warnings
        assert len(validation["warnings"]) >= 2  # Should have warnings about low threshold and missing docs
        assert "Very low confidence threshold" in str(validation["warnings"])
        assert "does not exist" in str(validation["warnings"])
    
    def test_invalid_json_file_handling(self):
        """Test handling of invalid JSON configuration file."""
        # Create invalid JSON file
        with open(self.config_file, 'w') as f:
            f.write("invalid json content {")
        
        # Should not raise error, should return defaults
        config = self.manager.load_config()
        assert isinstance(config, AppConfig)
        assert config.primary_llm_provider == "google_gemini_paid"  # Default value
    
    def test_config_file_creation(self):
        """Test that config directory is created if it doesn't exist."""
        # Use nested path that doesn't exist
        nested_path = Path(self.temp_dir) / "nested" / "config" / "config.json"
        manager = ConfigurationManager(str(nested_path))
        
        # Parent directory should be created
        assert nested_path.parent.exists()
        
        # Should be able to save config
        config = AppConfig()
        manager.save_config(config)
        assert nested_path.exists()


class TestEnvironmentVariableLoading:
    """Test environment variable loading functionality."""
    
    def test_all_env_vars_loaded(self):
        """Test that all supported environment variables are loaded."""
        env_vars = {
            'GOOGLE_API_KEY': 'google_test',
            'OPENAI_API_KEY': 'openai_test',
            'ANTHROPIC_API_KEY': 'anthropic_test',
            'GROQ_API_KEY': 'groq_test',
            'MISTRAL_API_KEY': 'mistral_test',
            'DOCS_FOLDER': '/test/docs',
            'ANSWER_MIN_CONF': '0.3',
            'MAX_CHUNKS': '8',
            'MAX_CHARS_PER_CHUNK': '1000',
            'ALLOW_HOSTS': 'host1.com,host2.com',
            'TRUSTED_DOMAINS': 'trusted1.com',
            'DEFAULT_SEEDS': 'http://seed1.com,http://seed2.com',
            'CRAWL_DEPTH': '2',
            'CRAWL_MAX_PAGES': '50',
            'CRAWL_SAME_DOMAIN': 'false',
            'GOOGLE_LLM_MODEL': 'custom-gemini',
            'GOOGLE_EMBEDDING_MODEL': 'custom-embedding',
            'PRIMARY_LLM_PROVIDER': 'openai',
            'MONTHLY_BUDGET_USD': '15.5'
        }
        
        with patch.dict(os.environ, env_vars):
            manager = ConfigurationManager()
            config = manager.load_config()
        
        # Check API keys
        assert config.api_keys.google_api_key == 'google_test'
        assert config.api_keys.openai_api_key == 'openai_test'
        assert config.api_keys.anthropic_api_key == 'anthropic_test'
        assert config.api_keys.groq_api_key == 'groq_test'
        assert config.api_keys.mistral_api_key == 'mistral_test'
        
        # Check docs
        assert config.docs.local_path == '/test/docs'
        
        # Check RAG
        assert config.rag.confidence_threshold == 0.3
        assert config.rag.max_chunks == 8
        assert config.rag.max_chars_per_chunk == 1000
        
        # Check crawl
        assert config.crawl.allow_hosts == ['host1.com', 'host2.com']
        assert config.crawl.trusted_domains == ['trusted1.com']
        assert config.crawl.default_seeds == ['http://seed1.com', 'http://seed2.com']
        assert config.crawl.depth == 2
        assert config.crawl.max_pages == 50
        assert config.crawl.same_domain is False
        
        # Check model overrides
        assert config.model_overrides.gemini_llm_model == 'custom-gemini'
        assert config.model_overrides.gemini_embedding_model == 'custom-embedding'
        
        # Check general
        assert config.primary_llm_provider == 'openai'
        assert config.monthly_budget_usd == 15.5
    
    def test_env_var_type_conversion_errors(self):
        """Test handling of invalid environment variable values."""
        with patch.dict(os.environ, {
            'ANSWER_MIN_CONF': 'not_a_number',
            'MAX_CHUNKS': 'invalid',
            'CRAWL_SAME_DOMAIN': 'maybe'
        }):
            manager = ConfigurationManager()
            # Should not raise error, should use defaults
            config = manager.load_config()
            
            # Should use defaults when conversion fails
            assert config.rag.confidence_threshold == 0.25  # Default
            assert config.rag.max_chunks == 5  # Default
            assert config.crawl.same_domain is True  # Default