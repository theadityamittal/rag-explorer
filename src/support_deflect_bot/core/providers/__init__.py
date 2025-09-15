"""Provider implementations for LLM and embedding services."""

from .base import (
    BaseProvider,
    CombinedProvider,
    EmbeddingProvider,
    LLMProvider,
    ProviderConfig,
    ProviderCostExceededError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTier,
    ProviderType,
    ProviderUnavailableError,
)
from .config import (
    ProviderInstance,
    ProviderRegistry,
    ProviderSelector,
    build_fallback_chain,
    get_available_providers,
    get_default_registry,
    register_provider,
)

# Import all concrete provider implementations (auto-registers them)
from .implementations import *
from .strategies import (
    BALANCED_STRATEGY,
    COST_OPTIMIZED_STRATEGY,
    QUALITY_FIRST_STRATEGY,
    SPEED_FOCUSED_STRATEGY,
    STRATEGIES,
    ProviderStrategy,
    StrategyManager,
    StrategyType,
)

__all__ = [
    # Base provider classes
    "BaseProvider",
    "LLMProvider",
    "EmbeddingProvider",
    "CombinedProvider",
    "ProviderConfig",
    "ProviderType",
    "ProviderTier",
    # Error classes
    "ProviderError",
    "ProviderUnavailableError",
    "ProviderRateLimitError",
    "ProviderCostExceededError",
    # Strategy system
    "ProviderStrategy",
    "StrategyType",
    "StrategyManager",
    "STRATEGIES",
    "COST_OPTIMIZED_STRATEGY",
    "SPEED_FOCUSED_STRATEGY",
    "QUALITY_FIRST_STRATEGY",
    "BALANCED_STRATEGY",
    # Configuration and management
    "ProviderRegistry",
    "ProviderSelector",
    "ProviderInstance",
    "get_default_registry",
    "register_provider",
    "get_available_providers",
    "build_fallback_chain",
]
