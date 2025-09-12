"""
Unit tests for provider strategy selection logic.

Tests the strategy selection algorithms used to choose optimal providers
for different tasks based on cost, performance, availability, and capabilities.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from tests.base import BaseProviderTest


class TestProviderSelectionStrategies(BaseProviderTest):
    """Test different provider selection strategies."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    def test_cost_optimized_strategy(self):
        """Test cost-optimized provider selection strategy."""
        # Arrange
        providers = [
            self.create_mock_provider("openai", cost_per_token=0.03),
            self.create_mock_provider("anthropic", cost_per_token=0.015),
            self.create_mock_provider("groq", cost_per_token=0.001),
            self.create_mock_provider("ollama", cost_per_token=0.0)  # Local
        ]
        
        # All providers available
        for provider in providers:
            provider.is_available.return_value = True
        
        # Act - Cost-optimized selection
        cost_sorted = sorted(providers, key=lambda p: getattr(p, 'cost_per_token', float('inf')))
        selected = cost_sorted[0]
        
        # Assert
        assert selected.name == "ollama"  # Cheapest (free local)
        assert selected.cost_per_token == 0.0
        
        # Test excluding local providers
        cloud_providers = [p for p in providers if p.name != "ollama"]
        cloud_selected = min(cloud_providers, key=lambda p: p.cost_per_token)
        
        assert cloud_selected.name == "groq"
        assert cloud_selected.cost_per_token == 0.001
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_performance_optimized_strategy(self):
        """Test performance-optimized provider selection strategy."""
        # Arrange
        providers = [
            self.create_mock_provider("openai", avg_response_time=2.5, quality_score=0.95),
            self.create_mock_provider("anthropic", avg_response_time=1.8, quality_score=0.93),
            self.create_mock_provider("groq", avg_response_time=0.8, quality_score=0.85),
            self.create_mock_provider("ollama", avg_response_time=5.0, quality_score=0.75)
        ]
        
        # All providers available
        for provider in providers:
            provider.is_available.return_value = True
        
        # Act - Speed-optimized selection
        speed_selected = min(providers, key=lambda p: getattr(p, 'avg_response_time', float('inf')))
        
        # Assert
        assert speed_selected.name == "groq"
        assert speed_selected.avg_response_time == 0.8
        
        # Act - Quality-optimized selection
        quality_selected = max(providers, key=lambda p: getattr(p, 'quality_score', 0))
        
        # Assert
        assert quality_selected.name == "openai"
        assert quality_selected.quality_score == 0.95
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_balanced_strategy(self):
        """Test balanced provider selection strategy."""
        # Arrange
        providers = [
            self.create_mock_provider("openai", cost_per_token=0.03, quality_score=0.95, avg_response_time=2.5),
            self.create_mock_provider("anthropic", cost_per_token=0.015, quality_score=0.93, avg_response_time=1.8),
            self.create_mock_provider("groq", cost_per_token=0.001, quality_score=0.85, avg_response_time=0.8)
        ]
        
        # All providers available
        for provider in providers:
            provider.is_available.return_value = True
        
        # Act - Calculate balanced score (quality/cost ratio with speed bonus)
        def balanced_score(provider):
            cost = getattr(provider, 'cost_per_token', 1)
            quality = getattr(provider, 'quality_score', 0.5)
            speed = getattr(provider, 'avg_response_time', 10)
            
            # Higher is better (quality/cost, with speed penalty)
            return (quality / max(cost, 0.001)) * (1 / max(speed, 0.1))
        
        balanced_selected = max(providers, key=balanced_score)
        
        # Assert - Groq should win with good balance of cost/quality/speed
        assert balanced_selected.name == "groq"
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_capability_based_strategy(self):
        """Test capability-based provider selection strategy."""
        # Arrange
        providers = {
            'openai': self.create_mock_provider("openai", capabilities=['EMBEDDING', 'COMPLETION', 'CHAT']),
            'groq': self.create_mock_provider("groq", capabilities=['COMPLETION', 'CHAT']),
            'ollama': self.create_mock_provider("ollama", capabilities=['EMBEDDING', 'COMPLETION'])
        }
        
        # All providers available
        for provider in providers.values():
            provider.is_available.return_value = True
        
        # Act & Assert - Test embedding capability requirement
        embedding_providers = [p for p in providers.values() if 'EMBEDDING' in getattr(p, 'capabilities', [])]
        assert len(embedding_providers) == 2
        assert any(p.name == 'openai' for p in embedding_providers)
        assert any(p.name == 'ollama' for p in embedding_providers)
        
        # Act & Assert - Test chat capability requirement
        chat_providers = [p for p in providers.values() if 'CHAT' in getattr(p, 'capabilities', [])]
        assert len(chat_providers) == 2
        assert any(p.name == 'openai' for p in chat_providers)
        assert any(p.name == 'groq' for p in chat_providers)
        
        # Act & Assert - Test multi-capability requirement
        multi_capable = [p for p in providers.values() 
                        if all(cap in getattr(p, 'capabilities', []) for cap in ['EMBEDDING', 'CHAT'])]
        assert len(multi_capable) == 1
        assert multi_capable[0].name == 'openai'
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_availability_first_strategy(self):
        """Test availability-first provider selection strategy."""
        # Arrange
        available_provider = self.create_mock_provider("groq")
        available_provider.is_available.return_value = True
        available_provider.priority = 2
        
        unavailable_provider = self.create_mock_provider("openai")
        unavailable_provider.is_available.return_value = False
        unavailable_provider.priority = 1  # Higher priority but unavailable
        
        providers = [unavailable_provider, available_provider]
        
        # Act - Availability-first selection
        available = [p for p in providers if p.is_available()]
        selected = available[0] if available else None
        
        # Assert
        assert selected is not None
        assert selected.name == "groq"
        assert selected.is_available()
        
        # Test fallback when no providers available
        for provider in providers:
            provider.is_available.return_value = False
            
        available_fallback = [p for p in providers if p.is_available()]
        assert len(available_fallback) == 0


class TestFallbackChainLogic(BaseProviderTest):
    """Test fallback chain construction and execution logic."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    def test_priority_based_fallback_chain(self):
        """Test building fallback chains based on provider priorities."""
        # Arrange
        providers = [
            self.create_mock_provider("openai", priority=1, cost_per_token=0.03),
            self.create_mock_provider("anthropic", priority=2, cost_per_token=0.015),
            self.create_mock_provider("groq", priority=3, cost_per_token=0.001),
            self.create_mock_provider("ollama", priority=4, cost_per_token=0.0)
        ]
        
        # Act - Build priority-ordered chain
        priority_chain = sorted(providers, key=lambda p: getattr(p, 'priority', 999))
        
        # Assert
        assert len(priority_chain) == 4
        assert priority_chain[0].name == "openai"     # Highest priority (1)
        assert priority_chain[1].name == "anthropic"  # Second priority (2)
        assert priority_chain[2].name == "groq"       # Third priority (3)
        assert priority_chain[3].name == "ollama"     # Lowest priority (4)
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_cost_aware_fallback_chain(self):
        """Test building cost-aware fallback chains."""
        # Arrange
        providers = [
            self.create_mock_provider("openai", cost_per_token=0.03, quality_score=0.95),
            self.create_mock_provider("groq", cost_per_token=0.001, quality_score=0.85),
            self.create_mock_provider("ollama", cost_per_token=0.0, quality_score=0.75)
        ]
        
        # Act - Build cost-optimized chain (cheapest first)
        cost_chain = sorted(providers, key=lambda p: getattr(p, 'cost_per_token', float('inf')))
        
        # Assert
        assert cost_chain[0].name == "ollama"   # Free
        assert cost_chain[1].name == "groq"     # Very cheap
        assert cost_chain[2].name == "openai"   # Expensive
        
        # Act - Build quality-adjusted cost chain (value optimization)
        def value_score(provider):
            cost = getattr(provider, 'cost_per_token', 1)
            quality = getattr(provider, 'quality_score', 0.5)
            return quality / max(cost, 0.001)  # Quality per dollar
            
        value_chain = sorted(providers, key=value_score, reverse=True)
        
        # Assert - Groq should provide best value (good quality, low cost)
        assert value_chain[0].name == "groq"
        
    @pytest.mark.unit
    @pytest.mark.providers
    async def test_fallback_chain_execution(self):
        """Test executing fallback chain when providers fail."""
        # Arrange
        failing_provider = self.create_mock_provider("openai")
        failing_provider.is_available.return_value = True
        failing_provider.generate_response = AsyncMock(side_effect=Exception("Rate limited"))
        
        working_provider = self.create_mock_provider("groq")
        working_provider.is_available.return_value = True
        working_provider.generate_response = AsyncMock(return_value={
            "content": "Fallback response",
            "model": "llama-3.1-8b-instant"
        })
        
        fallback_chain = [failing_provider, working_provider]
        
        # Act - Execute fallback chain
        result = None
        last_error = None
        
        for provider in fallback_chain:
            try:
                if provider.is_available():
                    result = await provider.generate_response("test query")
                    break
            except Exception as e:
                last_error = e
                continue
                
        # Assert
        assert result is not None
        assert result["content"] == "Fallback response"
        assert result["model"] == "llama-3.1-8b-instant"
        assert last_error is not None  # First provider did fail
        
        # Verify call order
        failing_provider.generate_response.assert_called_once_with("test query")
        working_provider.generate_response.assert_called_once_with("test query")
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_circular_fallback_prevention(self):
        """Test prevention of circular fallbacks in chain."""
        # Arrange
        provider_a = self.create_mock_provider("openai")
        provider_b = self.create_mock_provider("groq")
        
        # Create circular reference (bad)
        circular_chain = [provider_a, provider_b, provider_a]  # provider_a appears twice
        
        # Act - Remove duplicates to prevent circular fallback
        unique_chain = []
        seen_names = set()
        
        for provider in circular_chain:
            if provider.name not in seen_names:
                unique_chain.append(provider)
                seen_names.add(provider.name)
                
        # Assert
        assert len(unique_chain) == 2
        assert unique_chain[0].name == "openai"
        assert unique_chain[1].name == "groq"
        assert len(seen_names) == 2
        

class TestDynamicProviderSelection(BaseProviderTest):
    """Test dynamic provider selection based on runtime conditions."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    def test_load_based_provider_selection(self):
        """Test selecting providers based on current load."""
        # Arrange
        providers = [
            self.create_mock_provider("openai", current_load=0.9, max_requests_per_minute=100),
            self.create_mock_provider("anthropic", current_load=0.5, max_requests_per_minute=80),
            self.create_mock_provider("groq", current_load=0.2, max_requests_per_minute=200)
        ]
        
        # All providers available
        for provider in providers:
            provider.is_available.return_value = True
        
        # Act - Select provider with lowest load
        least_loaded = min(providers, key=lambda p: getattr(p, 'current_load', 1.0))
        
        # Assert
        assert least_loaded.name == "groq"
        assert least_loaded.current_load == 0.2
        
        # Act - Select provider with highest capacity remaining
        def remaining_capacity(provider):
            load = getattr(provider, 'current_load', 1.0)
            max_rpm = getattr(provider, 'max_requests_per_minute', 1)
            return max_rpm * (1 - load)
            
        highest_capacity = max(providers, key=remaining_capacity)
        
        # Assert - Groq has highest remaining capacity
        assert highest_capacity.name == "groq"
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_time_based_provider_selection(self):
        """Test selecting providers based on time of day or usage patterns."""
        # Arrange
        import datetime
        
        providers = [
            self.create_mock_provider("openai", peak_hours=[9, 10, 11, 14, 15, 16]),
            self.create_mock_provider("groq", peak_hours=[12, 13, 17, 18, 19, 20]),
            self.create_mock_provider("ollama", peak_hours=[])  # Local, no peak hours
        ]
        
        # Mock current time
        current_hour = 15  # 3 PM - OpenAI peak time
        
        # Act - Find providers not in peak hours
        non_peak_providers = [p for p in providers 
                             if current_hour not in getattr(p, 'peak_hours', [])]
        
        # Assert
        assert len(non_peak_providers) == 2
        provider_names = [p.name for p in non_peak_providers]
        assert 'groq' in provider_names
        assert 'ollama' in provider_names
        assert 'openai' not in provider_names
        
    @pytest.mark.unit
    @pytest.mark.providers  
    def test_budget_constrained_provider_selection(self):
        """Test selecting providers based on budget constraints."""
        # Arrange
        daily_budget = 10.0  # $10 per day
        current_spent = 7.5   # $7.50 already spent today
        remaining_budget = daily_budget - current_spent  # $2.50 remaining
        
        providers = [
            self.create_mock_provider("openai", cost_per_token=0.03),      # Expensive
            self.create_mock_provider("anthropic", cost_per_token=0.015),  # Moderate
            self.create_mock_provider("groq", cost_per_token=0.001),       # Cheap
            self.create_mock_provider("ollama", cost_per_token=0.0)        # Free
        ]
        
        # Estimated tokens for request
        estimated_tokens = 200
        
        # Act - Filter providers by budget constraint
        affordable_providers = []
        for provider in providers:
            cost_per_token = getattr(provider, 'cost_per_token', 0)
            estimated_cost = estimated_tokens * cost_per_token
            if estimated_cost <= remaining_budget:
                affordable_providers.append(provider)
        
        # Assert
        affordable_names = [p.name for p in affordable_providers]
        assert 'ollama' in affordable_names    # Free, always affordable
        assert 'groq' in affordable_names      # $0.20 cost, under budget
        assert 'anthropic' in affordable_names # $3.00 cost, over budget but close
        assert 'openai' not in affordable_names # $6.00 cost, way over budget
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_geographic_provider_selection(self):
        """Test selecting providers based on geographic preferences."""
        # Arrange
        user_region = "us-east"
        
        providers = [
            self.create_mock_provider("openai", regions=["us-east", "us-west", "eu-west"]),
            self.create_mock_provider("anthropic", regions=["us-east", "us-west"]),
            self.create_mock_provider("groq", regions=["us-west", "eu-west"]),
            self.create_mock_provider("ollama", regions=["local"])  # Local deployment
        ]
        
        # Act - Filter providers by region support
        regional_providers = [p for p in providers 
                            if user_region in getattr(p, 'regions', []) or 'local' in getattr(p, 'regions', [])]
        
        # Assert
        regional_names = [p.name for p in regional_providers]
        assert 'openai' in regional_names
        assert 'anthropic' in regional_names
        assert 'ollama' in regional_names  # Local is always available
        assert 'groq' not in regional_names  # Doesn't support us-east


class TestProviderHealthMonitoring(BaseProviderTest):
    """Test provider health monitoring and circuit breaker logic."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    def test_provider_health_scoring(self):
        """Test calculating provider health scores."""
        # Arrange
        provider = self.create_mock_provider("openai")
        provider.metrics = {
            'success_rate': 0.95,     # 95% success rate
            'avg_response_time': 1.2,  # 1.2 second average
            'error_rate': 0.05,       # 5% error rate
            'uptime': 0.99            # 99% uptime
        }
        
        # Act - Calculate health score
        def calculate_health_score(provider):
            metrics = getattr(provider, 'metrics', {})
            success_rate = metrics.get('success_rate', 0.5)
            uptime = metrics.get('uptime', 0.5)
            error_rate = metrics.get('error_rate', 0.5)
            response_time = metrics.get('avg_response_time', 10.0)
            
            # Higher is better (0-100 scale)
            score = (
                (success_rate * 40) +      # Success rate weight: 40%
                (uptime * 30) +            # Uptime weight: 30%
                ((1 - error_rate) * 20) +  # Error rate weight: 20% (inverted)
                (max(0, (5 - response_time) / 5) * 10)  # Speed weight: 10%
            )
            return min(100, max(0, score))
            
        health_score = calculate_health_score(provider)
        
        # Assert
        assert health_score > 85  # Should be high with good metrics
        assert health_score <= 100
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_circuit_breaker_logic(self):
        """Test circuit breaker pattern for failing providers."""
        # Arrange
        provider = self.create_mock_provider("openai")
        
        # Circuit breaker states
        circuit_states = ['CLOSED', 'OPEN', 'HALF_OPEN']
        
        # Mock circuit breaker
        provider.circuit_breaker = {
            'state': 'CLOSED',
            'failure_count': 0,
            'failure_threshold': 5,
            'recovery_timeout': 60,  # seconds
            'last_failure_time': None
        }
        
        # Act - Simulate failures
        def record_failure(provider):
            cb = provider.circuit_breaker
            cb['failure_count'] += 1
            cb['last_failure_time'] = 1234567890  # Mock timestamp
            
            if cb['failure_count'] >= cb['failure_threshold']:
                cb['state'] = 'OPEN'
                
        def can_execute(provider):
            cb = provider.circuit_breaker
            if cb['state'] == 'CLOSED':
                return True
            elif cb['state'] == 'OPEN':
                # Check if recovery timeout has passed
                current_time = 1234567890 + 61  # 61 seconds later
                if current_time - cb['last_failure_time'] > cb['recovery_timeout']:
                    cb['state'] = 'HALF_OPEN'
                    return True
                return False
            elif cb['state'] == 'HALF_OPEN':
                return True
            return False
        
        # Test initial state
        assert provider.circuit_breaker['state'] == 'CLOSED'
        assert can_execute(provider) is True
        
        # Simulate failures
        for _ in range(5):
            record_failure(provider)
            
        # Assert circuit is now open
        assert provider.circuit_breaker['state'] == 'OPEN'
        assert can_execute(provider) is False
        
        # Simulate recovery timeout passing
        assert can_execute(provider) is True  # Should transition to HALF_OPEN
        
    @pytest.mark.unit
    @pytest.mark.providers
    def test_provider_blacklist_logic(self):
        """Test provider blacklisting for consistently failing providers."""
        # Arrange
        providers = [
            self.create_mock_provider("openai", reliability_score=0.95),
            self.create_mock_provider("groq", reliability_score=0.40),  # Poor reliability
            self.create_mock_provider("ollama", reliability_score=0.85)
        ]
        
        blacklist_threshold = 0.5  # Blacklist providers below 50% reliability
        
        # Act - Filter out blacklisted providers
        reliable_providers = [p for p in providers 
                            if getattr(p, 'reliability_score', 1.0) >= blacklist_threshold]
        
        # Assert
        assert len(reliable_providers) == 2
        reliable_names = [p.name for p in reliable_providers]
        assert 'openai' in reliable_names
        assert 'ollama' in reliable_names
        assert 'groq' not in reliable_names  # Blacklisted due to poor reliability