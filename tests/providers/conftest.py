"""
Pytest configuration and fixtures for provider tests.
"""

import pytest
import os
import logging
from unittest.mock import patch, Mock
from typing import Dict, Any

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def clean_environment():
    """Fixture that provides a clean environment without API keys."""
    original_env = dict(os.environ)

    # Clear all provider API keys
    api_keys = [
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
    ]

    for key in api_keys:
        if key in os.environ:
            del os.environ[key]

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_api_keys():
    """Fixture that provides mock API keys for testing."""
    mock_keys = {
        "OPENAI_API_KEY": "sk-test-openai-key-12345678901234567890",
        "GROQ_API_KEY": "gsk_test_groq_key_12345678901234567890",
        "MISTRAL_API_KEY": "test-mistral-key-12345678901234567890",
        "ANTHROPIC_API_KEY": "sk-ant-test-claude-key-12345678901234567890",
        "GOOGLE_API_KEY": "AIza_test_google_key_12345678901234567890",
    }

    with patch.dict(os.environ, mock_keys):
        yield mock_keys


@pytest.fixture
def mock_provider_config():
    """Fixture that provides a standard mock provider configuration."""

    def _create_config(
        name: str = "Test Provider",
        cost_input: float = 1.0,
        cost_output: float = 2.0,
        gdpr_compliant: bool = True,
        free_tier: bool = False,
    ):
        from src.support_deflect_bot.core.providers.base import (
            ProviderConfig,
            ProviderType,
            ProviderTier,
        )

        return ProviderConfig(
            name=name,
            provider_type=ProviderType.LLM,
            cost_per_million_tokens_input=cost_input,
            cost_per_million_tokens_output=cost_output,
            max_context_length=8000,
            rate_limit_rpm=60,
            rate_limit_tpm=60000,
            supports_streaming=True,
            requires_api_key=not free_tier,
            tier=ProviderTier.FREE if free_tier else ProviderTier.PAID,
            regions_supported=["global"],
            gdpr_compliant=gdpr_compliant,
            models_available=["test-model-1", "test-model-2"],
        )

    return _create_config


@pytest.fixture
def mock_provider():
    """Fixture that provides a mock provider instance."""

    def _create_provider(
        name: str = "test_provider",
        available: bool = True,
        config_override: Dict[str, Any] = None,
    ):
        from src.support_deflect_bot.core.providers.base import LLMProvider

        provider = Mock(spec=LLMProvider)
        provider.is_available.return_value = available

        # Default config
        config = {
            "name": f"Test {name.title()}",
            "cost_input": 1.0,
            "cost_output": 2.0,
            "gdpr_compliant": True,
            "free_tier": False,
        }

        if config_override:
            config.update(config_override)

        provider.get_config.return_value = mock_provider_config()(**config)
        provider.health_check.return_value = {
            "status": "healthy" if available else "unhealthy",
            "provider": name,
            "response_time_ms": 100.0 if available else None,
            "timestamp": 1234567890.0,
        }

        return provider

    return _create_provider


@pytest.fixture
def sample_providers(mock_provider):
    """Fixture that provides a set of sample providers for testing."""
    return {
        "cheap": mock_provider(
            name="cheap", config_override={"cost_input": 0.15, "cost_output": 0.60}
        ),
        "expensive": mock_provider(
            name="expensive", config_override={"cost_input": 10.0, "cost_output": 30.0}
        ),
        "free": mock_provider(
            name="free",
            config_override={"cost_input": 0.0, "cost_output": 0.0, "free_tier": True},
        ),
        "gdpr_compliant": mock_provider(
            name="gdpr_compliant", config_override={"gdpr_compliant": True}
        ),
        "non_gdpr": mock_provider(
            name="non_gdpr", config_override={"gdpr_compliant": False}
        ),
        "unavailable": mock_provider(name="unavailable", available=False),
    }


@pytest.fixture
def registry_with_providers(sample_providers):
    """Fixture that provides a registry with sample providers."""
    from src.support_deflect_bot.core.providers.config import ProviderRegistry

    registry = ProviderRegistry()

    # Register sample providers
    for name, provider_instance in sample_providers.items():
        # Mock the provider class
        provider_class = Mock()
        provider_class.return_value = provider_instance

        registry._provider_classes[name] = provider_class

    return registry


@pytest.fixture(scope="session")
def integration_test_requirements():
    """Check if integration test requirements are met."""
    requirements = {
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "has_groq_key": bool(os.getenv("GROQ_API_KEY")),
        "has_claude_key": bool(os.getenv("ANTHROPIC_API_KEY")),
        "has_google_key": bool(os.getenv("GOOGLE_API_KEY")),
        "has_mistral_key": bool(os.getenv("MISTRAL_API_KEY")),
        "has_any_key": any(
            [
                os.getenv("OPENAI_API_KEY"),
                os.getenv("GROQ_API_KEY"),
                os.getenv("ANTHROPIC_API_KEY"),
                os.getenv("GOOGLE_API_KEY"),
                os.getenv("MISTRAL_API_KEY"),
            ]
        ),
    }

    logger.info(f"Integration test requirements: {requirements}")
    return requirements


@pytest.fixture
def cost_tracking_setup():
    """Fixture that sets up cost tracking configuration."""
    cost_config = {
        "MONTHLY_BUDGET_USD": "10.0",
        "COST_ALERT_THRESHOLD": "0.8",
        "ENABLE_COST_TRACKING": "true",
    }

    with patch.dict(os.environ, cost_config):
        yield cost_config


@pytest.fixture
def regional_compliance_setup():
    """Fixture that sets up regional compliance testing."""

    def _setup_region(region: str, gdpr_required: bool = None):
        config = {"REGIONAL_COMPLIANCE": "true"}

        # Mock region detection
        with (
            patch.dict(os.environ, config),
            patch(
                "src.support_deflect_bot.utils.region_detector.RegionDetector.detect_region",
                return_value=region,
            ),
            patch(
                "src.support_deflect_bot.utils.region_detector.is_gdpr_region",
                return_value=(
                    gdpr_required
                    if gdpr_required is not None
                    else region in ["DE", "FR", "IT", "ES", "NL"]
                ),
            ),
        ):

            yield {"region": region, "gdpr_required": gdpr_required, "config": config}

    return _setup_region
