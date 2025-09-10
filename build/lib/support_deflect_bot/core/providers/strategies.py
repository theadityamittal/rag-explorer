"""Provider selection strategies for different use cases."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class StrategyType(Enum):
    """Available provider selection strategies."""

    COST_OPTIMIZED = "cost_optimized"
    SPEED_FOCUSED = "speed_focused"
    QUALITY_FIRST = "quality_first"
    BALANCED = "balanced"
    CUSTOM = "custom"


@dataclass
class ProviderStrategy:
    """Defines how providers should be prioritized and selected."""

    name: str
    strategy_type: StrategyType
    description: str
    priority_factors: Dict[str, float]  # cost, speed, quality, reliability
    max_cost_per_million_input: float
    max_cost_per_million_output: float
    prefer_free_tier: bool
    require_streaming: bool
    require_gdpr_compliance: bool
    min_context_length: int
    max_daily_cost_usd: float


# Pre-defined strategies for different use cases
COST_OPTIMIZED_STRATEGY = ProviderStrategy(
    name="Cost Optimized",
    strategy_type=StrategyType.COST_OPTIMIZED,
    description="Prioritize lowest cost providers while maintaining reliability",
    priority_factors={"cost": 0.6, "reliability": 0.3, "speed": 0.05, "quality": 0.05},
    max_cost_per_million_input=2.0,  # $2 per million input tokens max
    max_cost_per_million_output=6.0,  # $6 per million output tokens max
    prefer_free_tier=True,
    require_streaming=False,
    require_gdpr_compliance=False,  # Will be overridden by region detection
    min_context_length=4000,
    max_daily_cost_usd=1.0,  # $1 per day maximum
)

SPEED_FOCUSED_STRATEGY = ProviderStrategy(
    name="Speed Focused",
    strategy_type=StrategyType.SPEED_FOCUSED,
    description="Prioritize fastest inference speed for real-time applications",
    priority_factors={"speed": 0.5, "reliability": 0.3, "cost": 0.15, "quality": 0.05},
    max_cost_per_million_input=10.0,
    max_cost_per_million_output=30.0,
    prefer_free_tier=False,
    require_streaming=True,
    require_gdpr_compliance=False,
    min_context_length=8000,
    max_daily_cost_usd=5.0,  # $5 per day for speed
)

QUALITY_FIRST_STRATEGY = ProviderStrategy(
    name="Quality First",
    strategy_type=StrategyType.QUALITY_FIRST,
    description="Prioritize highest quality models regardless of cost",
    priority_factors={"quality": 0.5, "reliability": 0.3, "speed": 0.1, "cost": 0.1},
    max_cost_per_million_input=100.0,  # High budget for quality
    max_cost_per_million_output=300.0,
    prefer_free_tier=False,
    require_streaming=False,
    require_gdpr_compliance=False,
    min_context_length=16000,
    max_daily_cost_usd=20.0,  # $20 per day for quality
)

BALANCED_STRATEGY = ProviderStrategy(
    name="Balanced",
    strategy_type=StrategyType.BALANCED,
    description="Balance cost, speed, quality, and reliability equally",
    priority_factors={
        "cost": 0.25,
        "speed": 0.25,
        "quality": 0.25,
        "reliability": 0.25,
    },
    max_cost_per_million_input=5.0,
    max_cost_per_million_output=15.0,
    prefer_free_tier=False,
    require_streaming=False,
    require_gdpr_compliance=False,
    min_context_length=8000,
    max_daily_cost_usd=3.0,  # $3 per day balanced
)

# Registry of all available strategies
STRATEGIES = {
    StrategyType.COST_OPTIMIZED: COST_OPTIMIZED_STRATEGY,
    StrategyType.SPEED_FOCUSED: SPEED_FOCUSED_STRATEGY,
    StrategyType.QUALITY_FIRST: QUALITY_FIRST_STRATEGY,
    StrategyType.BALANCED: BALANCED_STRATEGY,
}


class StrategyManager:
    """Manages provider selection strategies."""

    def __init__(self):
        self._strategies = STRATEGIES.copy()
        self._custom_strategies = {}

    def get_strategy(self, strategy_type: StrategyType) -> ProviderStrategy:
        """Get strategy by type.

        Args:
            strategy_type: Type of strategy to retrieve

        Returns:
            ProviderStrategy configuration

        Raises:
            ValueError: If strategy type not found
        """
        if strategy_type in self._strategies:
            return self._strategies[strategy_type]
        elif strategy_type in self._custom_strategies:
            return self._custom_strategies[strategy_type]
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def get_strategy_by_name(self, name: str) -> ProviderStrategy:
        """Get strategy by name string.

        Args:
            name: Strategy name (e.g., 'cost_optimized')

        Returns:
            ProviderStrategy configuration
        """
        try:
            strategy_type = StrategyType(name.lower())
            return self.get_strategy(strategy_type)
        except ValueError:
            raise ValueError(f"Unknown strategy name: {name}")

    def register_custom_strategy(self, strategy: ProviderStrategy):
        """Register a custom user-defined strategy.

        Args:
            strategy: Custom strategy configuration
        """
        self._custom_strategies[strategy.strategy_type] = strategy

    def list_available_strategies(self) -> List[str]:
        """List all available strategy names."""
        built_in = [s.value for s in StrategyType if s != StrategyType.CUSTOM]
        custom = [s.name for s in self._custom_strategies.values()]
        return built_in + custom

    def calculate_provider_score(
        self, strategy: ProviderStrategy, provider_metrics: Dict[str, float]
    ) -> float:
        """Calculate provider score based on strategy.

        Args:
            strategy: Strategy to use for scoring
            provider_metrics: Dict with cost, speed, quality, reliability scores (0-1)

        Returns:
            Weighted score (0-1, higher is better)
        """
        score = 0.0

        for factor, weight in strategy.priority_factors.items():
            if factor in provider_metrics:
                score += weight * provider_metrics[factor]

        return min(1.0, max(0.0, score))

    def adapt_strategy_for_region(
        self, strategy: ProviderStrategy, region: str
    ) -> ProviderStrategy:
        """Adapt strategy based on user region.

        Args:
            strategy: Base strategy to adapt
            region: User region code

        Returns:
            Adapted strategy with region-specific requirements
        """
        from ...utils.region_detector import is_gdpr_region  # Avoid circular import

        # Create a copy to avoid modifying the original
        adapted = ProviderStrategy(
            name=f"{strategy.name} ({region})",
            strategy_type=strategy.strategy_type,
            description=f"{strategy.description} - adapted for {region}",
            priority_factors=strategy.priority_factors.copy(),
            max_cost_per_million_input=strategy.max_cost_per_million_input,
            max_cost_per_million_output=strategy.max_cost_per_million_output,
            prefer_free_tier=strategy.prefer_free_tier,
            require_streaming=strategy.require_streaming,
            require_gdpr_compliance=strategy.require_gdpr_compliance,
            min_context_length=strategy.min_context_length,
            max_daily_cost_usd=strategy.max_daily_cost_usd,
        )

        # Apply region-specific adaptations
        if is_gdpr_region(region):
            adapted.require_gdpr_compliance = True
            adapted.prefer_free_tier = False  # EU requires paid services
            # Increase cost limits since free tiers aren't available
            adapted.max_cost_per_million_input *= 2
            adapted.max_cost_per_million_output *= 2
            adapted.max_daily_cost_usd *= 2

        return adapted
