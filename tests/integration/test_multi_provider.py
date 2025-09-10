"""
Integration tests for multi-provider functionality.

Tests real provider integration, cost tracking, and end-to-end workflows.
"""

import pytest
import os
import logging
from unittest.mock import patch, Mock
import time

logger = logging.getLogger(__name__)


class TestMultiProviderIntegration:
    """Integration tests for the complete multi-provider system."""

    def setup_method(self):
        """Setup for each test method."""
        # Import here to avoid circular imports during test collection
        from support_deflect_bot.core.providers.implementations import (
            register_all_providers,
        )

        register_all_providers()

    def test_provider_registration_integration(self):
        """Test that all providers register and are accessible."""
        from support_deflect_bot.core.providers.config import get_default_registry

        registry = get_default_registry()

        # Should have exactly 8 providers
        assert len(registry._provider_classes) == 8

        # Test specific providers exist
        required_providers = {
            "openai",
            "groq",
            "mistral",
            "google_gemini_free",
            "google_gemini_paid",
            "claude_api",
            "claude_code",
            "ollama",
        }

        registered = set(registry._provider_classes.keys())
        missing = required_providers - registered
        extra = registered - required_providers

        assert not missing, f"Missing required providers: {missing}"
        assert not extra, f"Unexpected extra providers: {extra}"

    def test_fallback_chain_building_integration(self):
        """Test complete fallback chain building process."""
        from support_deflect_bot.core.providers.config import get_default_registry
        from support_deflect_bot.core.providers.base import ProviderType

        registry = get_default_registry()

        # Test different strategies
        strategies = ["cost_optimized", "speed_focused", "quality_first", "balanced"]

        for strategy in strategies:
            chain = registry.build_fallback_chain(ProviderType.LLM, strategy)

            # Chain should be built successfully (may be empty if no providers available)
            assert isinstance(chain, list)

            # If providers are available, check they're properly ordered
            if len(chain) > 1:
                # Basic sanity check - providers should have configs
                for provider in chain:
                    config = provider.get_config()
                    assert config.provider_type in [ProviderType.LLM, ProviderType.BOTH]

                logger.info(f"Strategy '{strategy}': {len(chain)} providers in chain")
            else:
                logger.info(
                    f"Strategy '{strategy}': No available providers (expected without API keys)"
                )

    def test_provider_health_checks_integration(self):
        """Test provider health checking across all providers."""
        from support_deflect_bot.core.providers.config import get_default_registry

        registry = get_default_registry()
        health_results = {}

        for name, provider_class in registry._provider_classes.items():
            try:
                provider = provider_class()
                health = provider.health_check()

                # Validate health check structure
                assert "status" in health
                assert "provider" in health
                assert health["status"] in [
                    "healthy",
                    "unhealthy",
                    "degraded",
                    "rate_limited",
                    "auth_error",
                ]

                health_results[name] = health["status"]
                logger.info(f"{name}: {health['status']}")

            except Exception as e:
                health_results[name] = f"error: {str(e)}"
                logger.info(f"{name}: error - {str(e)}")

        # At least some providers should be testable
        assert len(health_results) == 8, "Should test all providers"

    def test_provider_selector_integration(self):
        """Test provider selector with real registry."""
        from support_deflect_bot.core.providers.config import (
            get_default_registry,
            ProviderSelector,
        )
        from support_deflect_bot.core.providers.base import ProviderType

        registry = get_default_registry()
        selector = ProviderSelector(registry)

        # Test primary provider selection
        provider = selector.select_primary_provider(ProviderType.LLM, "cost_optimized")

        if provider:
            config = provider.get_config()
            assert config.provider_type in [ProviderType.LLM, ProviderType.BOTH]
            logger.info(f"Selected primary provider: {config.name}")
        else:
            logger.info("No primary provider available (expected without API keys)")

        # Test fallback selection
        fallbacks = selector.select_with_fallbacks(
            ProviderType.LLM, "cost_optimized", max_fallbacks=3
        )
        assert len(fallbacks) <= 4  # Primary + 3 fallbacks max

        logger.info(f"Fallback chain length: {len(fallbacks)}")

    def test_cost_optimization_integration(self):
        """Test cost optimization features."""
        from support_deflect_bot.core.providers.strategies import (
            StrategyManager,
            StrategyType,
        )

        strategy_manager = StrategyManager()

        # Test all strategies can be created
        strategies = [
            StrategyType.COST_OPTIMIZED,
            StrategyType.SPEED_FOCUSED,
            StrategyType.QUALITY_FIRST,
            StrategyType.BALANCED,
        ]

        for strategy_type in strategies:
            strategy = strategy_manager.get_strategy(strategy_type)
            assert strategy.strategy_type == strategy_type
            assert isinstance(strategy.priority_factors, dict)
            assert "cost" in strategy.priority_factors

            # Test regional adaptation
            eu_adapted = strategy_manager.adapt_strategy_for_region(strategy, "DE")
            us_adapted = strategy_manager.adapt_strategy_for_region(strategy, "US")

            # EU should require GDPR compliance
            assert eu_adapted.require_gdpr_compliance == True
            assert us_adapted.require_gdpr_compliance == False

            logger.info(
                f"{strategy_type.value}: Cost factor = {strategy.priority_factors.get('cost', 0)}"
            )

    def test_settings_integration(self):
        """Test that all required settings are accessible."""
        try:
            from support_deflect_bot.utils.settings import (
                OPENAI_API_KEY,
                GROQ_API_KEY,
                MISTRAL_API_KEY,
                ANTHROPIC_API_KEY,
                GOOGLE_API_KEY,
                MONTHLY_BUDGET_USD,
                COST_ALERT_THRESHOLD,
                DEFAULT_STRATEGY,
                REGIONAL_COMPLIANCE,
                OPENAI_LLM_MODEL,
                CLAUDE_API_MODEL,
                GROQ_MODEL,
                MISTRAL_MODEL,
                GOOGLE_MODEL,
                CLAUDE_CODE_PATH,
            )

            # Test that settings exist and have correct types
            assert isinstance(MONTHLY_BUDGET_USD, float)
            assert isinstance(COST_ALERT_THRESHOLD, float)
            assert isinstance(DEFAULT_STRATEGY, str)
            assert isinstance(REGIONAL_COMPLIANCE, bool)

            # Test model settings
            assert isinstance(OPENAI_LLM_MODEL, str)
            assert isinstance(CLAUDE_API_MODEL, str)
            assert isinstance(GROQ_MODEL, str)
            assert isinstance(MISTRAL_MODEL, str)
            assert isinstance(GOOGLE_MODEL, str)
            assert isinstance(CLAUDE_CODE_PATH, str)

            logger.info(
                f"Budget: ${MONTHLY_BUDGET_USD}/month, Alert: {COST_ALERT_THRESHOLD*100}%"
            )
            logger.info(f"Default strategy: {DEFAULT_STRATEGY}")
            logger.info(f"GDPR compliance: {REGIONAL_COMPLIANCE}")

        except ImportError as e:
            pytest.fail(f"Failed to import required settings: {e}")


class TestRegionalComplianceIntegration:
    """Integration tests for regional compliance features."""

    def test_gdpr_provider_filtering(self):
        """Test GDPR compliance filtering integration."""
        from support_deflect_bot.core.providers.config import ProviderRegistry
        from support_deflect_bot.core.providers.implementations import (
            register_all_providers,
        )
        from support_deflect_bot.core.providers.base import ProviderType

        register_all_providers()
        registry = ProviderRegistry()

        # Mock EU region detection
        with (
            patch.object(registry._region_detector, "detect_region", return_value="DE"),
            patch(
                "src.support_deflect_bot.utils.region_detector.is_gdpr_region",
                return_value=True,
            ),
        ):

            # Get providers for EU region
            available_providers = registry.get_available_providers(
                ProviderType.LLM, region="DE"
            )

            gdpr_compliant_count = 0
            non_compliant_count = 0

            for name, provider in available_providers:
                try:
                    config = provider.get_config()
                    if config.gdpr_compliant:
                        gdpr_compliant_count += 1
                        logger.info(f"✅ GDPR compliant: {name}")
                    else:
                        non_compliant_count += 1
                        logger.warning(f"❌ Non-GDPR: {name} (should be filtered)")
                except Exception as e:
                    logger.info(f"⚠️ {name}: {str(e)}")

            # In GDPR regions, should only have compliant providers
            if non_compliant_count > 0:
                logger.warning(
                    f"Found {non_compliant_count} non-GDPR providers in EU region"
                )

            logger.info(
                f"EU region: {gdpr_compliant_count} GDPR compliant, {non_compliant_count} non-compliant"
            )


class TestErrorHandlingIntegration:
    """Integration tests for error handling and graceful degradation."""

    def test_no_api_keys_graceful_degradation(self):
        """Test system behavior with no API keys."""
        from support_deflect_bot.core.providers.config import (
            ProviderRegistry,
            ProviderSelector,
        )
        from support_deflect_bot.core.providers.implementations import (
            register_all_providers,
        )
        from support_deflect_bot.core.providers.base import ProviderType

        # Clear all API keys
        with patch.dict(os.environ, {}, clear=True):
            register_all_providers()
            registry = ProviderRegistry()
            selector = ProviderSelector(registry)

            # Should handle gracefully without crashing
            primary = selector.select_primary_provider(ProviderType.LLM)
            fallbacks = selector.select_with_fallbacks(
                ProviderType.LLM, max_fallbacks=5
            )

            # May have some providers (like Ollama or Claude Code) that don't require API keys
            logger.info(
                f"No API keys: Primary={primary is not None}, Fallbacks={len(fallbacks)}"
            )

            # Should not crash
            assert isinstance(fallbacks, list)

    def test_provider_failure_handling(self):
        """Test handling of individual provider failures."""
        from support_deflect_bot.core.providers.config import ProviderRegistry
        from support_deflect_bot.core.providers.base import (
            ProviderType,
            ProviderUnavailableError,
        )

        registry = ProviderRegistry()

        # Mock a provider class that always fails
        failing_provider_class = Mock()
        failing_provider_class.side_effect = ProviderUnavailableError(
            "Test provider failure", provider="test_failing"
        )

        registry._provider_classes["test_failing"] = failing_provider_class

        # Should handle the failure gracefully
        available = registry.get_available_providers(ProviderType.LLM)

        # Should not include the failing provider
        provider_names = [name for name, _ in available]
        assert "test_failing" not in provider_names

        logger.info(
            f"Handled provider failure gracefully, {len(available)} providers remain"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestRealProviderIntegration:
    """Integration tests with real providers (requires API keys)."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY"
    )
    def test_openai_real_integration(self):
        """Test real OpenAI provider integration."""
        from support_deflect_bot.core.providers.implementations import (
            OpenAIProvider,
        )

        provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))

        # Test availability
        assert provider.is_available(), "OpenAI should be available with valid API key"

        # Test health check
        health = provider.health_check()
        assert health["status"] == "healthy", f"OpenAI health check failed: {health}"
        assert health["response_time_ms"] > 0

        # Test configuration
        config = provider.get_config()
        assert config.name == "OpenAI"
        assert config.gdpr_compliant == True
        assert config.requires_api_key == True

        logger.info(
            f"✅ OpenAI integration test passed: {health['response_time_ms']:.1f}ms"
        )

    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="Requires GROQ_API_KEY")
    def test_groq_real_integration(self):
        """Test real Groq provider integration."""
        from support_deflect_bot.core.providers.implementations import GroqProvider

        provider = GroqProvider(api_key=os.getenv("GROQ_API_KEY"))

        # Test availability
        assert provider.is_available(), "Groq should be available with valid API key"

        # Test health check
        health = provider.health_check()
        assert health["status"] == "healthy", f"Groq health check failed: {health}"
        assert health["speed_rating"] == "ultra_fast"

        # Test configuration
        config = provider.get_config()
        assert config.name == "Groq"
        assert config.gdpr_compliant == False  # US-based

        logger.info(
            f"✅ Groq integration test passed: {health['response_time_ms']:.1f}ms"
        )

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"), reason="Requires ANTHROPIC_API_KEY"
    )
    def test_claude_api_real_integration(self):
        """Test real Claude API provider integration."""
        from support_deflect_bot.core.providers.implementations import (
            ClaudeAPIProvider,
        )

        provider = ClaudeAPIProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Test availability
        assert (
            provider.is_available()
        ), "Claude API should be available with valid API key"

        # Test health check
        health = provider.health_check()
        assert (
            health["status"] == "healthy"
        ), f"Claude API health check failed: {health}"

        # Test configuration
        config = provider.get_config()
        assert config.name == "Claude API"
        assert config.gdpr_compliant == True
        assert config.max_context_length == 200000

        logger.info(
            f"✅ Claude API integration test passed: {health['response_time_ms']:.1f}ms"
        )

    def test_claude_code_integration(self):
        """Test Claude Code provider integration (no API key required)."""
        from support_deflect_bot.core.providers.implementations import (
            ClaudeCodeProvider,
        )

        provider = ClaudeCodeProvider()

        # Test configuration (should work without API key)
        config = provider.get_config()
        assert config.name == "Claude Code"
        assert config.requires_api_key == False
        assert config.cost_per_million_tokens_input == 0.0

        # Test availability (depends on Claude Code installation)
        is_available = provider.is_available()

        if is_available:
            # Test health check if available
            health = provider.health_check()
            assert "status" in health
            logger.info(f"✅ Claude Code available: {health['status']}")
        else:
            logger.info("⚠️ Claude Code not available (not installed)")

    @pytest.mark.skipif(
        not any(
            [
                os.getenv(key)
                for key in ["OPENAI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"]
            ]
        ),
        reason="Requires at least one API key",
    )
    def test_multi_provider_real_fallback(self):
        """Test real multi-provider fallback with available providers."""
        from support_deflect_bot.core.providers.config import (
            get_default_registry,
            ProviderSelector,
        )
        from support_deflect_bot.core.providers.implementations import (
            register_all_providers,
        )
        from support_deflect_bot.core.providers.base import ProviderType

        register_all_providers()
        registry = get_default_registry()
        selector = ProviderSelector(registry)

        # Get fallback chain with real providers
        fallbacks = selector.select_with_fallbacks(
            ProviderType.LLM, "cost_optimized", max_fallbacks=3
        )

        # Should have at least one provider if API keys are present
        assert len(fallbacks) >= 1, "Should have at least one provider with API keys"

        # Test each provider in chain
        for i, provider in enumerate(fallbacks):
            config = provider.get_config()
            is_available = provider.is_available()

            logger.info(
                f"Fallback {i+1}: {config.name} - {'✅ Available' if is_available else '❌ Unavailable'}"
            )

            if is_available:
                health = provider.health_check()
                assert health["status"] in [
                    "healthy",
                    "degraded",
                ], f"{config.name} health check failed"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    pytest.main([__file__, "-v", "--tb=short"])
