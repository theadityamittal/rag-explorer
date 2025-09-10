"""Provider configuration and management system."""

import importlib.util
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from ...utils.region_detector import ComplianceChecker, RegionDetector, is_gdpr_region
from .base import (
    BaseProvider,
    EmbeddingProvider,
    LLMProvider,
    ProviderConfig,
    ProviderTier,
    ProviderType,
)
from .strategies import ProviderStrategy, StrategyManager, StrategyType

logger = logging.getLogger(__name__)


@dataclass
class ProviderInstance:
    """Wrapper for provider instances with metadata."""

    provider: BaseProvider
    config: ProviderConfig
    last_health_check: Optional[Dict[str, Any]] = None
    is_healthy: bool = True
    usage_count: int = 0
    total_cost: float = 0.0


class ProviderRegistry:
    """Central registry for managing provider configurations and discovery."""

    def __init__(self):
        self._provider_classes: Dict[str, Type[BaseProvider]] = {}
        self._provider_instances: Dict[str, ProviderInstance] = {}
        self._region_detector = RegionDetector()
        self._compliance_checker = ComplianceChecker(self._region_detector)
        self._strategy_manager = StrategyManager()
        self._user_region = self._region_detector.detect_region()

        logger.info(f"Initialized provider registry for region: {self._user_region}")

    def register_provider_class(
        self, name: str, provider_class: Type[BaseProvider]
    ) -> None:
        """Register a provider class for discovery.

        Args:
            name: Unique name for the provider
            provider_class: Provider class to register
        """
        self._provider_classes[name] = provider_class
        logger.debug(f"Registered provider class: {name}")

    def get_or_create_provider(
        self, name: str, api_key: Optional[str] = None, **kwargs
    ) -> Optional[BaseProvider]:
        """Get existing provider instance or create new one.

        Args:
            name: Provider name
            api_key: API key for the provider
            **kwargs: Additional provider-specific arguments

        Returns:
            Provider instance or None if unavailable
        """
        # Check if already instantiated
        if name in self._provider_instances:
            instance = self._provider_instances[name]
            if instance.is_healthy:
                return instance.provider

        # Create new instance
        if name not in self._provider_classes:
            logger.warning(f"Provider class not registered: {name}")
            return None

        try:
            provider_class = self._provider_classes[name]
            provider = provider_class(api_key=api_key, **kwargs)

            # Check if provider is available
            if not provider.is_available():
                logger.warning(f"Provider {name} is not available")
                return None

            # Check regional compliance
            if not self._compliance_checker.check_provider_compliance(
                name, self._user_region
            ):
                logger.warning(
                    f"Provider {name} not compliant for region {self._user_region}"
                )
                return None

            # Store instance
            instance = ProviderInstance(
                provider=provider, config=provider.get_config(), is_healthy=True
            )
            self._provider_instances[name] = instance

            logger.info(f"Created provider instance: {name}")
            return provider

        except Exception as e:
            logger.error(f"Failed to create provider {name}: {e}")
            return None

    def get_available_providers(
        self, provider_type: Optional[ProviderType] = None, region: Optional[str] = None
    ) -> List[Tuple[str, BaseProvider]]:
        """Get list of available providers filtered by type and region.

        Args:
            provider_type: Filter by provider type (LLM, EMBEDDING, BOTH)
            region: Filter by region compliance (defaults to user region)

        Returns:
            List of (name, provider) tuples for available providers
        """
        region = region or self._user_region
        available = []

        # Try to get API keys from environment for each provider
        api_keys = self._get_available_api_keys()

        for name, provider_class in self._provider_classes.items():
            try:
                # Skip if not compliant for region
                if not self._compliance_checker.check_provider_compliance(name, region):
                    continue

                # Try to create provider instance
                api_key = api_keys.get(name)
                provider = self.get_or_create_provider(name, api_key)

                if provider:
                    # Filter by provider type
                    config = provider.get_config()
                    if (
                        provider_type is None
                        or config.provider_type == provider_type
                        or config.provider_type == ProviderType.BOTH
                    ):
                        available.append((name, provider))

            except Exception as e:
                logger.debug(f"Provider {name} not available: {e}")
                continue

        logger.info(
            f"Found {len(available)} available providers for type {provider_type} in region {region}"
        )
        return available

    def build_fallback_chain(
        self,
        provider_type: ProviderType,
        strategy: str = "cost_optimized",
        exclude: Optional[List[str]] = None,
    ) -> List[BaseProvider]:
        """Build provider fallback chain based on strategy.

        Args:
            provider_type: Type of providers to include in chain
            strategy: Selection strategy name
            exclude: Provider names to exclude from chain

        Returns:
            Ordered list of providers for fallback chain
        """
        exclude = exclude or []

        # Get strategy configuration
        try:
            strategy_config = self._strategy_manager.get_strategy_by_name(strategy)
            # Adapt for user's region
            strategy_config = self._strategy_manager.adapt_strategy_for_region(
                strategy_config, self._user_region
            )
        except ValueError as e:
            logger.warning(f"Unknown strategy {strategy}, using cost_optimized: {e}")
            strategy_config = self._strategy_manager.get_strategy(
                StrategyType.COST_OPTIMIZED
            )

        # Get available providers
        available_providers = self.get_available_providers(
            provider_type, self._user_region
        )

        # Filter out excluded providers
        filtered_providers = [
            (name, provider)
            for name, provider in available_providers
            if name not in exclude
        ]

        # Score and sort providers
        scored_providers = []
        for name, provider in filtered_providers:
            config = provider.get_config()

            # Skip providers that exceed strategy limits
            if (
                config.cost_per_million_tokens_input
                > strategy_config.max_cost_per_million_input
                or config.cost_per_million_tokens_output
                > strategy_config.max_cost_per_million_output
            ):
                continue

            # Skip if minimum context length not met
            if config.max_context_length < strategy_config.min_context_length:
                continue

            # Skip free tier if strategy doesn't prefer it
            if (
                not strategy_config.prefer_free_tier
                and config.tier == ProviderTier.FREE
            ):
                continue

            # Skip if streaming required but not supported
            if strategy_config.require_streaming and not config.supports_streaming:
                continue

            # Calculate provider score
            metrics = self._calculate_provider_metrics(config)
            score = self._strategy_manager.calculate_provider_score(
                strategy_config, metrics
            )

            scored_providers.append((score, name, provider))

        # Sort by score (highest first)
        scored_providers.sort(key=lambda x: x[0], reverse=True)

        # Return just the providers in order
        chain = [provider for _, _, provider in scored_providers]

        logger.info(
            f"Built fallback chain with {len(chain)} providers using strategy {strategy}"
        )
        for i, (score, name, _) in enumerate(scored_providers):
            logger.debug(f"  {i+1}. {name} (score: {score:.3f})")

        return chain

    def get_provider_by_name(self, name: str) -> Optional[BaseProvider]:
        """Get specific provider instance by name.

        Args:
            name: Provider name

        Returns:
            Provider instance or None if not available
        """
        if name in self._provider_instances:
            instance = self._provider_instances[name]
            if instance.is_healthy:
                return instance.provider

        # Try to create it
        api_keys = self._get_available_api_keys()
        api_key = api_keys.get(name)
        return self.get_or_create_provider(name, api_key)

    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all providers.

        Returns:
            Dict mapping provider names to health check results
        """
        results = {}

        for name, instance in self._provider_instances.items():
            try:
                health_result = instance.provider.health_check()
                instance.last_health_check = health_result
                instance.is_healthy = health_result.get("status") == "healthy"
                results[name] = health_result
            except Exception as e:
                instance.is_healthy = False
                results[name] = {"status": "error", "error": str(e)}

        return results

    def get_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all providers.

        Returns:
            Dict mapping provider names to usage stats
        """
        stats = {}

        for name, instance in self._provider_instances.items():
            stats[name] = {
                "usage_count": instance.usage_count,
                "total_cost": instance.total_cost,
                "is_healthy": instance.is_healthy,
                "config": {
                    "cost_per_million_input": instance.config.cost_per_million_tokens_input,
                    "cost_per_million_output": instance.config.cost_per_million_tokens_output,
                    "tier": instance.config.tier.value,
                    "regions_supported": instance.config.regions_supported,
                },
            }

        return stats

    def _get_available_api_keys(self) -> Dict[str, str]:
        """Get available API keys from environment variables."""
        api_keys = {}

        # Common API key environment variable patterns
        key_mappings = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "google_gemini": "GOOGLE_API_KEY",
            "google_gemini_paid": "GOOGLE_API_KEY",
            "google_gemini_free": "GOOGLE_API_KEY",
        }

        for provider_name, env_var in key_mappings.items():
            api_key = os.getenv(env_var)
            if api_key:
                api_keys[provider_name] = api_key

        return api_keys

    def _calculate_provider_metrics(self, config: ProviderConfig) -> Dict[str, float]:
        """Calculate normalized metrics for provider scoring.

        Args:
            config: Provider configuration

        Returns:
            Dict with normalized metrics (0-1, higher is better)
        """
        # Cost score (lower cost = higher score)
        max_reasonable_cost = 50.0  # $50 per million tokens as reference
        cost_input_score = max(
            0, 1 - (config.cost_per_million_tokens_input / max_reasonable_cost)
        )
        cost_output_score = max(
            0, 1 - (config.cost_per_million_tokens_output / max_reasonable_cost)
        )
        cost_score = (cost_input_score + cost_output_score) / 2

        # Speed score (based on rate limits as proxy)
        max_reasonable_rpm = 1000  # 1000 RPM as reference
        speed_score = min(1.0, config.rate_limit_rpm / max_reasonable_rpm)

        # Quality score (based on tier and context length)
        tier_scores = {
            ProviderTier.FREE: 0.3,
            ProviderTier.PAID: 0.7,
            ProviderTier.PREMIUM: 1.0,
        }
        tier_score = tier_scores.get(config.tier, 0.5)

        # Context length as quality indicator
        max_reasonable_context = 100000  # 100k tokens as reference
        context_score = min(1.0, config.max_context_length / max_reasonable_context)

        quality_score = (tier_score + context_score) / 2

        # Reliability score (based on global availability and GDPR compliance)
        reliability_score = 0.5  # Base reliability
        if "global" in config.regions_supported:
            reliability_score += 0.3
        if config.gdpr_compliant:
            reliability_score += 0.2
        reliability_score = min(1.0, reliability_score)

        return {
            "cost": cost_score,
            "speed": speed_score,
            "quality": quality_score,
            "reliability": reliability_score,
        }


class ProviderSelector:
    """Handles provider selection logic based on various criteria."""

    def __init__(self, registry: Optional[ProviderRegistry] = None):
        self.registry = registry or ProviderRegistry()
        self.region_detector = RegionDetector()

    def select_primary_provider(
        self,
        provider_type: ProviderType,
        strategy: str = "cost_optimized",
        region: Optional[str] = None,
    ) -> Optional[BaseProvider]:
        """Select the best primary provider based on strategy.

        Args:
            provider_type: Type of provider needed
            strategy: Selection strategy name
            region: Target region (defaults to user region)

        Returns:
            Best available provider or None
        """
        chain = self.registry.build_fallback_chain(provider_type, strategy)
        return chain[0] if chain else None

    def select_with_fallbacks(
        self,
        provider_type: ProviderType,
        strategy: str = "cost_optimized",
        max_fallbacks: int = 3,
    ) -> List[BaseProvider]:
        """Select primary provider with fallback chain.

        Args:
            provider_type: Type of provider needed
            strategy: Selection strategy name
            max_fallbacks: Maximum number of fallback providers

        Returns:
            List of providers in preference order
        """
        chain = self.registry.build_fallback_chain(provider_type, strategy)
        return chain[: max_fallbacks + 1]  # Primary + fallbacks


# Convenience functions for direct usage
def get_default_registry() -> ProviderRegistry:
    """Get default provider registry instance."""
    if not hasattr(get_default_registry, "_instance"):
        get_default_registry._instance = ProviderRegistry()
    return get_default_registry._instance


def register_provider(name: str, provider_class: Type[BaseProvider]) -> None:
    """Register a provider class with the default registry."""
    registry = get_default_registry()
    registry.register_provider_class(name, provider_class)


def get_available_providers(
    provider_type: Optional[ProviderType] = None,
) -> List[Tuple[str, BaseProvider]]:
    """Get available providers from default registry."""
    registry = get_default_registry()
    return registry.get_available_providers(provider_type)


def build_fallback_chain(
    provider_type: ProviderType, strategy: str = "cost_optimized"
) -> List[BaseProvider]:
    """Build fallback chain using default registry."""
    registry = get_default_registry()
    return registry.build_fallback_chain(provider_type, strategy)
