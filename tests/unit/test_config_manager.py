"""Unit tests for the ConfigurationManager."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock

from support_deflect_bot.config.manager import ConfigurationManager, get_config_manager
from support_deflect_bot.config.schema import AppConfig, ApiKeysConfig


class TestConfigurationManager:
    """Test suite for ConfigurationManager class."""

    @pytest.fixture
    def temp_config_path(self, tmp_path):
        """Create a temporary config path for testing."""
        config_path = tmp_path / "config.json"
        return str(config_path)

    @pytest.fixture
    def config_manager(self, temp_config_path):
        """Create a ConfigurationManager instance with temporary config path."""
        return ConfigurationManager(config_file=temp_config_path)

    @pytest.fixture
    def sample_config_data(self):
        """Sample configuration data for testing."""
        return {
            "api_keys": {
                "google_api_key": "test_google_key",
                "openai_api_key": "test_openai_key"
            },
            "docs": {
                "local_path": "/test/docs"
            },
            "rag": {
                "confidence_threshold": 0.7,
                "max_chunks": 5
            },
            "crawl": {
                "allow_hosts": ["example.com"],
                "trusted_domains": ["example.com"]
            }
        }

    def test_initialization_default_path(self):
        """Test ConfigurationManager initialization with default path."""
        manager = ConfigurationManager()
        expected_path = Path.home() / ".support-deflect-bot" / "config.json"
        assert manager.config_file == expected_path

    def test_initialization_custom_path(self, temp_config_path):
        """Test ConfigurationManager initialization with custom path."""
        manager = ConfigurationManager(config_file=temp_config_path)
        assert str(manager.config_file) == temp_config_path

    def test_load_config_no_file(self, config_manager):
        """Test loading configuration when no file exists."""
        config = config_manager.load_config()

        assert isinstance(config, AppConfig)
        assert config_manager._config is not None

    def test_load_config_from_file(self, config_manager, sample_config_data):
        """Test loading configuration from file."""
        # Write test config to file
        with open(config_manager.config_file, 'w') as f:
            json.dump(sample_config_data, f)

        config = config_manager.load_config()

        assert config.api_keys.google_api_key == "test_google_key"
        assert config.api_keys.openai_api_key == "test_openai_key"
        assert config.docs.local_path == "/test/docs"
        assert config.rag.confidence_threshold == 0.7

    def test_load_config_invalid_json(self, config_manager):
        """Test loading configuration with invalid JSON file."""
        # Write invalid JSON to file
        with open(config_manager.config_file, 'w') as f:
            f.write("{ invalid json }")

        config = config_manager.load_config()

        # Should fall back to default config
        assert isinstance(config, AppConfig)

    @patch.dict(os.environ, {
        'GOOGLE_API_KEY': 'env_google_key',
        'OPENAI_API_KEY': 'env_openai_key',
        'DOCS_FOLDER': '/env/docs',
        'ANSWER_MIN_CONF': '0.8'
    })
    def test_load_config_from_environment(self, config_manager):
        """Test loading configuration from environment variables."""
        config = config_manager.load_config()

        assert config.api_keys.google_api_key == "env_google_key"
        assert config.api_keys.openai_api_key == "env_openai_key"
        assert config.docs.local_path == "/env/docs"
        assert config.rag.confidence_threshold == 0.8

    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'env_key'})
    def test_load_config_environment_overrides_file(self, config_manager, sample_config_data):
        """Test that environment variables override file configuration."""
        # Write config to file
        with open(config_manager.config_file, 'w') as f:
            json.dump(sample_config_data, f)

        config = config_manager.load_config()

        # Environment should override file
        assert config.api_keys.google_api_key == "env_key"
        # File values should still be present for non-overridden keys
        assert config.api_keys.openai_api_key == "test_openai_key"

    def test_save_config(self, config_manager):
        """Test saving configuration to file."""
        # Load default config
        config = config_manager.load_config()

        # Modify config
        config.api_keys.google_api_key = "saved_key"

        # Save config
        config_manager.save_config(config)

        # Verify file was written
        assert config_manager.config_file.exists()

        # Load and verify
        with open(config_manager.config_file, 'r') as f:
            saved_data = json.load(f)

        assert saved_data["api_keys"]["google_api_key"] == "saved_key"

    def test_save_config_no_config(self, config_manager):
        """Test saving configuration when no config is loaded."""
        with pytest.raises(ValueError, match="No configuration to save"):
            config_manager.save_config()

    def test_get_config(self, config_manager):
        """Test getting current configuration."""
        # First call should load config
        config1 = config_manager.get_config()
        assert isinstance(config1, AppConfig)

        # Second call should return cached config
        config2 = config_manager.get_config()
        assert config1 is config2

    def test_update_config_simple(self, config_manager):
        """Test updating configuration with simple key-value pairs."""
        updated_config = config_manager.update_config(primary_llm_provider="openai")

        assert updated_config.primary_llm_provider == "openai"

    def test_update_config_nested(self, config_manager):
        """Test updating configuration with nested keys."""
        updated_config = config_manager.update_config(**{
            "api_keys.google_api_key": "new_key",
            "rag.confidence_threshold": 0.9
        })

        assert updated_config.api_keys.google_api_key == "new_key"
        assert updated_config.rag.confidence_threshold == 0.9

    def test_set_api_key(self, config_manager):
        """Test setting API key for a provider."""
        config_manager.set_api_key("google", "new_google_key")

        config = config_manager.get_config()
        assert config.api_keys.google_api_key == "new_google_key"

    def test_set_docs_path(self, config_manager):
        """Test setting documentation path."""
        config_manager.set_docs_path("/new/docs/path")

        config = config_manager.get_config()
        assert config.docs.local_path == "/new/docs/path"

    def test_set_rag_config(self, config_manager):
        """Test setting RAG configuration."""
        config_manager.set_rag_config(
            confidence_threshold=0.8,
            max_chunks=10
        )

        config = config_manager.get_config()
        assert config.rag.confidence_threshold == 0.8
        assert config.rag.max_chunks == 10

    def test_set_crawl_config(self, config_manager):
        """Test setting crawl configuration."""
        config_manager.set_crawl_config(
            allow_hosts=["test.com"],
            depth=3
        )

        config = config_manager.get_config()
        assert config.crawl.allow_hosts == ["test.com"]
        assert config.crawl.depth == 3

    def test_get_env_vars(self, config_manager):
        """Test getting configuration as environment variables."""
        config_manager.set_api_key("google", "test_key")

        env_vars = config_manager.get_env_vars()

        assert isinstance(env_vars, dict)
        assert "GOOGLE_API_KEY" in env_vars
        assert env_vars["GOOGLE_API_KEY"] == "test_key"

    def test_export_env_file(self, config_manager, tmp_path):
        """Test exporting configuration as .env file."""
        config_manager.set_api_key("google", "test_key")
        env_file_path = tmp_path / ".env"

        config_manager.export_env_file(str(env_file_path))

        assert env_file_path.exists()
        content = env_file_path.read_text()
        assert "GOOGLE_API_KEY=test_key" in content

    def test_export_env_file_with_spaces(self, config_manager, tmp_path):
        """Test exporting configuration with values containing spaces."""
        config_manager.set_docs_path("/path with spaces")
        env_file_path = tmp_path / ".env"

        config_manager.export_env_file(str(env_file_path))

        content = env_file_path.read_text()
        assert 'DOCS_FOLDER="/path with spaces"' in content

    def test_export_env_file_io_error(self, config_manager):
        """Test export_env_file with IO error."""
        with pytest.raises(Exception):
            config_manager.export_env_file("/invalid/path/file.env")

    def test_validate_config_no_issues(self, config_manager, tmp_path):
        """Test configuration validation with no issues."""
        # Set up valid config
        config_manager.set_api_key("google", "valid_key")
        config_manager.set_docs_path(str(tmp_path))  # Use tmp_path
        config_manager.set_rag_config(confidence_threshold=0.7)
        config_manager.set_crawl_config(
            allow_hosts=["example.com"],
            trusted_domains=["example.com"]
        )

        validation = config_manager.validate_config()

        assert validation["valid"] is True
        assert len(validation["issues"]) == 0

    def test_validate_config_with_warnings(self, config_manager):
        """Test configuration validation with warnings."""
        # Create config with potential issues
        config_manager.set_rag_config(confidence_threshold=0.05)  # Very low

        validation = config_manager.validate_config()

        assert len(validation["warnings"]) > 0
        assert any("low confidence threshold" in warning for warning in validation["warnings"])

    def test_validate_config_no_api_keys(self, config_manager):
        """Test configuration validation with no API keys."""
        validation = config_manager.validate_config()

        assert any("No API keys configured" in warning for warning in validation["warnings"])

    def test_validate_config_nonexistent_docs_path(self, config_manager):
        """Test configuration validation with nonexistent docs path."""
        config_manager.set_docs_path("/nonexistent/path")

        validation = config_manager.validate_config()

        assert any("does not exist" in warning for warning in validation["warnings"])

    def test_validate_config_extreme_confidence_thresholds(self, config_manager):
        """Test configuration validation with extreme confidence thresholds."""
        # Test very high threshold
        config_manager.set_rag_config(confidence_threshold=0.95)
        validation = config_manager.validate_config()
        assert any("too many refusals" in warning for warning in validation["warnings"])

    @patch.dict(os.environ, {
        'ALLOW_HOSTS': 'host1.com, host2.com',
        'TRUSTED_DOMAINS': 'domain1.com, domain2.com',
        'CRAWL_DEPTH': '3',
        'CRAWL_SAME_DOMAIN': 'true',
        'MONTHLY_BUDGET_USD': '100.50'
    })
    def test_load_from_environment_comprehensive(self, config_manager):
        """Test comprehensive environment variable loading."""
        config = config_manager.load_config()

        assert config.crawl.allow_hosts == ["host1.com", "host2.com"]
        assert config.crawl.trusted_domains == ["domain1.com", "domain2.com"]
        assert config.crawl.depth == 3
        assert config.crawl.same_domain is True
        assert config.monthly_budget_usd == 100.50

    @patch.dict(os.environ, {
        'GOOGLE_LLM_MODEL': 'gemini-pro-custom',
        'OPENAI_EMBEDDING_MODEL': 'text-embedding-custom'
    })
    def test_load_model_overrides_from_environment(self, config_manager):
        """Test loading model overrides from environment."""
        config = config_manager.load_config()

        assert config.model_overrides.gemini_llm_model == "gemini-pro-custom"
        assert config.model_overrides.openai_embedding_model == "text-embedding-custom"

    def test_config_file_directory_creation(self, tmp_path):
        """Test that config directory is created if it doesn't exist."""
        config_path = tmp_path / "subdir" / "config.json"
        manager = ConfigurationManager(config_file=str(config_path))

        # Directory should be created during initialization
        assert config_path.parent.exists()

    def test_load_config_validation_failure(self, config_manager):
        """Test loading configuration with validation failure."""
        # Create invalid config data that will fail validation
        invalid_data = {
            "rag": {
                "confidence_threshold": "invalid_string"  # Should be float
            }
        }

        with open(config_manager.config_file, 'w') as f:
            json.dump(invalid_data, f)

        # Should fall back to default config
        config = config_manager.load_config()
        assert isinstance(config, AppConfig)

    def test_save_config_io_error(self, config_manager):
        """Test save_config with IO error."""
        config = config_manager.load_config()

        with patch("builtins.open", side_effect=IOError("disk full")):
            with pytest.raises(Exception):
                config_manager.save_config(config)

    def test_global_config_manager(self):
        """Test global configuration manager singleton."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()

        assert manager1 is manager2
        assert isinstance(manager1, ConfigurationManager)

    def test_load_from_environment_empty_values(self, config_manager):
        """Test that empty environment values are handled correctly."""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': ''}):
            env_config = config_manager._load_from_environment()

            # Empty values should not be included
            assert "api_keys" not in env_config or not env_config.get("api_keys", {}).get("google_api_key")

    def test_update_config_creates_nested_structure(self, config_manager):
        """Test that update_config creates nested structure for new keys."""
        config_manager.update_config(**{"new_section.new_key": "new_value"})

        config = config_manager.get_config()
        config_dict = config.dict()
        assert config_dict.get("new_section", {}).get("new_key") == "new_value"

    @patch.dict(os.environ, {
        'ANSWER_MIN_CONF': 'invalid_float',
        'MAX_CHUNKS': 'invalid_int'
    })
    def test_load_from_environment_invalid_types(self, config_manager):
        """Test handling of invalid type conversions from environment."""
        # Should not raise exception, invalid values should be ignored
        config = config_manager.load_config()
        assert isinstance(config, AppConfig)