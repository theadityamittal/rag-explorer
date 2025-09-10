"""Configuration manager for loading, saving, and managing app settings."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .schema import (
    ApiKeysConfig,
    AppConfig,
    CrawlConfig,
    DocsConfig,
    ModelOverridesConfig,
    RagConfig,
)

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Manages application configuration with multiple sources and persistence."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager.

        Args:
            config_file: Path to configuration file. Defaults to ~/.support-deflect-bot/config.json
        """
        if config_file:
            self.config_file = Path(config_file)
        else:
            self.config_file = Path.home() / ".support-deflect-bot" / "config.json"

        # Ensure config directory exists
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        self._config: Optional[AppConfig] = None

    def load_config(self) -> AppConfig:
        """Load configuration from all sources (file, environment, defaults).

        Priority order:
        1. Environment variables (highest priority)
        2. Configuration file
        3. Defaults (lowest priority)

        Returns:
            Loaded and validated configuration
        """
        # Start with defaults
        config_data = {}

        # Load from file if it exists
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    file_config = json.load(f)
                config_data.update(file_config)
                logger.debug(f"Loaded configuration from {self.config_file}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")

        # Override with environment variables
        env_config = self._load_from_environment()
        config_data.update(env_config)

        # Validate and create config object
        try:
            self._config = AppConfig(**config_data)
            logger.debug("Configuration loaded and validated successfully")
            return self._config
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            # Return default config if validation fails
            self._config = AppConfig()
            return self._config

    def save_config(self, config: Optional[AppConfig] = None) -> None:
        """Save configuration to file.

        Args:
            config: Configuration to save. Uses current config if None.
        """
        if config is None:
            config = self._config

        if config is None:
            raise ValueError("No configuration to save")

        try:
            # Convert to dictionary, excluding environment-only settings
            config_dict = config.dict(exclude_none=True)

            # Save to file
            with open(self.config_file, "w") as f:
                json.dump(config_dict, f, indent=2)

            logger.info(f"Configuration saved to {self.config_file}")

        except IOError as e:
            logger.error(f"Failed to save configuration to {self.config_file}: {e}")
            raise

    def get_config(self) -> AppConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config

    def update_config(self, **kwargs) -> AppConfig:
        """Update configuration with new values.

        Args:
            **kwargs: Configuration values to update

        Returns:
            Updated configuration
        """
        current = self.get_config()

        # Update specific sections
        config_dict = current.dict()

        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys like 'api_keys.google_api_key'
                parts = key.split(".")
                nested_dict = config_dict
                for part in parts[:-1]:
                    if part not in nested_dict:
                        nested_dict[part] = {}
                    nested_dict = nested_dict[part]
                nested_dict[parts[-1]] = value
            else:
                config_dict[key] = value

        # Recreate and validate config
        self._config = AppConfig(**config_dict)
        return self._config

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for a specific provider.

        Args:
            provider: Provider name (google, openai, anthropic, groq, mistral)
            api_key: API key value
        """
        provider_key = f"api_keys.{provider}_api_key"
        self.update_config(**{provider_key: api_key})

    def set_docs_path(self, path: str) -> None:
        """Set local documentation path.

        Args:
            path: Path to documentation folder
        """
        self.update_config(**{"docs.local_path": path})

    def set_rag_config(self, **kwargs) -> None:
        """Set RAG configuration parameters.

        Args:
            **kwargs: RAG parameters (confidence_threshold, max_chunks, max_chars_per_chunk)
        """
        rag_updates = {f"rag.{key}": value for key, value in kwargs.items()}
        self.update_config(**rag_updates)

    def set_crawl_config(self, **kwargs) -> None:
        """Set crawl configuration parameters.

        Args:
            **kwargs: Crawl parameters (allow_hosts, trusted_domains, etc.)
        """
        crawl_updates = {f"crawl.{key}": value for key, value in kwargs.items()}
        self.update_config(**crawl_updates)

    def get_env_vars(self) -> Dict[str, str]:
        """Get configuration as environment variables dictionary.

        Returns:
            Dictionary of environment variable names and values
        """
        config = self.get_config()
        return config.get_flat_dict()

    def export_env_file(self, file_path: str) -> None:
        """Export configuration as .env file.

        Args:
            file_path: Path to save .env file
        """
        env_vars = self.get_env_vars()

        try:
            with open(file_path, "w") as f:
                f.write("# Support Deflect Bot Configuration\n")
                f.write(
                    "# Generated automatically - edit with 'deflect-bot configure'\n\n"
                )

                for key, value in sorted(env_vars.items()):
                    # Quote values that might contain spaces
                    if " " in str(value) or "," in str(value):
                        f.write(f'{key}="{value}"\n')
                    else:
                        f.write(f"{key}={value}\n")

            logger.info(f"Configuration exported to {file_path}")

        except IOError as e:
            logger.error(f"Failed to export configuration to {file_path}: {e}")
            raise

    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration and return status.

        Returns:
            Dictionary with validation results
        """
        config = self.get_config()
        issues = []
        warnings = []

        # Check API keys
        api_keys = [
            config.api_keys.google_api_key,
            config.api_keys.openai_api_key,
            config.api_keys.anthropic_api_key,
            config.api_keys.groq_api_key,
            config.api_keys.mistral_api_key,
        ]

        if not any(api_keys):
            warnings.append(
                "No API keys configured. Add at least one API key for enhanced functionality."
            )

        # Check docs path
        if not os.path.exists(config.docs.local_path):
            warnings.append(
                f"Documentation path does not exist: {config.docs.local_path}"
            )

        # Check RAG config
        if config.rag.confidence_threshold < 0.1:
            warnings.append(
                "Very low confidence threshold may result in unreliable answers"
            )
        elif config.rag.confidence_threshold > 0.9:
            warnings.append(
                "Very high confidence threshold may result in too many refusals"
            )

        # Check crawl config
        if not config.crawl.allow_hosts:
            warnings.append("No allowed hosts configured for crawling")

        if not config.crawl.trusted_domains:
            warnings.append("No trusted domains configured")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "config_file": str(self.config_file),
            "config_exists": self.config_file.exists(),
        }

    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary from environment
        """
        env_config = {
            "api_keys": {},
            "docs": {},
            "rag": {},
            "crawl": {},
            "model_overrides": {},
        }

        # API Keys
        if os.getenv("GOOGLE_API_KEY"):
            env_config["api_keys"]["google_api_key"] = os.getenv("GOOGLE_API_KEY")
        if os.getenv("OPENAI_API_KEY"):
            env_config["api_keys"]["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("ANTHROPIC_API_KEY"):
            env_config["api_keys"]["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")
        if os.getenv("GROQ_API_KEY"):
            env_config["api_keys"]["groq_api_key"] = os.getenv("GROQ_API_KEY")
        if os.getenv("MISTRAL_API_KEY"):
            env_config["api_keys"]["mistral_api_key"] = os.getenv("MISTRAL_API_KEY")

        # Docs
        if os.getenv("DOCS_FOLDER"):
            env_config["docs"]["local_path"] = os.getenv("DOCS_FOLDER")

        # RAG
        if os.getenv("ANSWER_MIN_CONF"):
            env_config["rag"]["confidence_threshold"] = float(
                os.getenv("ANSWER_MIN_CONF")
            )
        if os.getenv("MAX_CHUNKS"):
            env_config["rag"]["max_chunks"] = int(os.getenv("MAX_CHUNKS"))
        if os.getenv("MAX_CHARS_PER_CHUNK"):
            env_config["rag"]["max_chars_per_chunk"] = int(
                os.getenv("MAX_CHARS_PER_CHUNK")
            )

        # Crawl
        if os.getenv("ALLOW_HOSTS"):
            env_config["crawl"]["allow_hosts"] = [
                h.strip() for h in os.getenv("ALLOW_HOSTS").split(",")
            ]
        if os.getenv("TRUSTED_DOMAINS"):
            env_config["crawl"]["trusted_domains"] = [
                d.strip() for d in os.getenv("TRUSTED_DOMAINS").split(",")
            ]
        if os.getenv("DEFAULT_SEEDS"):
            env_config["crawl"]["default_seeds"] = [
                s.strip() for s in os.getenv("DEFAULT_SEEDS").split(",")
            ]
        if os.getenv("CRAWL_DEPTH"):
            env_config["crawl"]["depth"] = int(os.getenv("CRAWL_DEPTH"))
        if os.getenv("CRAWL_MAX_PAGES"):
            env_config["crawl"]["max_pages"] = int(os.getenv("CRAWL_MAX_PAGES"))
        if os.getenv("CRAWL_SAME_DOMAIN"):
            env_config["crawl"]["same_domain"] = (
                os.getenv("CRAWL_SAME_DOMAIN").lower() == "true"
            )
        if os.getenv("CRAWL_USER_AGENT"):
            env_config["crawl"]["user_agent"] = os.getenv("CRAWL_USER_AGENT")

        # Model overrides
        if os.getenv("GOOGLE_LLM_MODEL"):
            env_config["model_overrides"]["gemini_llm_model"] = os.getenv(
                "GOOGLE_LLM_MODEL"
            )
        if os.getenv("GOOGLE_EMBEDDING_MODEL"):
            env_config["model_overrides"]["gemini_embedding_model"] = os.getenv(
                "GOOGLE_EMBEDDING_MODEL"
            )
        if os.getenv("OPENAI_LLM_MODEL"):
            env_config["model_overrides"]["openai_llm_model"] = os.getenv(
                "OPENAI_LLM_MODEL"
            )
        if os.getenv("OPENAI_EMBEDDING_MODEL"):
            env_config["model_overrides"]["openai_embedding_model"] = os.getenv(
                "OPENAI_EMBEDDING_MODEL"
            )

        # General
        if os.getenv("PRIMARY_LLM_PROVIDER"):
            env_config["primary_llm_provider"] = os.getenv("PRIMARY_LLM_PROVIDER")
        if os.getenv("PRIMARY_EMBEDDING_PROVIDER"):
            env_config["primary_embedding_provider"] = os.getenv(
                "PRIMARY_EMBEDDING_PROVIDER"
            )
        if os.getenv("MONTHLY_BUDGET_USD"):
            env_config["monthly_budget_usd"] = float(os.getenv("MONTHLY_BUDGET_USD"))

        # Remove empty sections
        return {k: v for k, v in env_config.items() if v}


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager
