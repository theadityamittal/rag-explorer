"""Comprehensive resilience module with retry logic, circuit breakers, and error handling."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorType(Enum):
    """Classification of errors for retry policies."""
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Set[Type[Exception]] = field(default_factory=set)
    non_retryable_exceptions: Set[Type[Exception]] = field(default_factory=lambda: {ProviderPermanentError})


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    success_threshold: int = 3
    reset_timeout: float = 60.0
    half_open_max_calls: int = 5


@dataclass
class TimeoutConfig:
    """Configuration for timeouts."""
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    total_timeout: float = 60.0


class RetryExhaustedException(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, attempts: int, last_error: Exception):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(message)


class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, circuit_name: str):
        self.circuit_name = circuit_name
        super().__init__(message)


class ProviderPermanentError(Exception):
    """Raised for non-retryable provider errors (auth, permissions, model not found)."""

    def __init__(self, message: str, provider: str, original_error: Exception = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(message)


class WebFetchError(Exception):
    """Raised for web fetching errors."""

    def __init__(self, message: str, url: str, status_code: int = None, original_error: Exception = None):
        self.url = url
        self.status_code = status_code
        self.original_error = original_error
        super().__init__(message)


class CircuitBreaker:
    """Circuit breaker implementation for preventing cascading failures."""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self._lock = Lock()

    def __enter__(self):
        """Context manager entry."""
        if not self._can_execute():
            raise CircuitBreakerOpenException(
                f"Circuit breaker {self.name} is open",
                self.name
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self._on_success()
        else:
            self._on_failure()

    def _can_execute(self) -> bool:
        """Check if circuit allows execution."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.config.reset_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return self.half_open_calls < self.config.half_open_max_calls
            return False

    def _on_success(self):
        """Handle successful execution."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} closed after recovery")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

    def _on_failure(self):
        """Handle failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker {self.name} opened after half-open failure")
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")

            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "half_open_calls": self.half_open_calls,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "reset_timeout": self.config.reset_timeout,
                    "half_open_max_calls": self.config.half_open_max_calls
                }
            }


class ErrorClassifier:
    """Utility for classifying errors into retryable and non-retryable categories."""

    # Default retryable exception patterns
    RETRYABLE_PATTERNS = {
        "ConnectionError", "TimeoutError", "HTTPError", "RequestException",
        "ServiceUnavailable", "BadGateway", "GatewayTimeout", "TooManyRequests",
        "InternalServerError", "NetworkError", "TemporaryFailure"
    }

    # Default non-retryable exception patterns
    NON_RETRYABLE_PATTERNS = {
        "AuthenticationError", "Unauthorized", "Forbidden", "NotFound",
        "BadRequest", "InvalidInput", "ValidationError", "PermissionError",
        "ConfigurationError", "ValueError", "TypeError"
    }

    @classmethod
    def classify_error(cls, error: Exception) -> ErrorType:
        """Classify an error for retry decisions."""
        error_name = type(error).__name__
        error_msg = str(error).lower()

        # Check for permanent auth/permission/model errors in message content
        permanent_message_patterns = [
            "unauthorized", "forbidden", "api key", "permission",
            "invalid model", "model not found", "authentication failed",
            "invalid api key", "access denied", "permission denied",
            "invalid credentials", "account disabled", "quota exceeded"
        ]
        if any(pattern in error_msg for pattern in permanent_message_patterns):
            return ErrorType.PERMANENT

        # Check for rate limiting
        if any(pattern in error_msg for pattern in ['rate limit', 'too many requests', '429']):
            return ErrorType.RATE_LIMIT

        # Check for timeout
        if any(pattern in error_msg for pattern in ['timeout', 'timed out', 'deadline exceeded']):
            return ErrorType.TIMEOUT

        # Check for connection issues
        if any(pattern in error_msg for pattern in ['connection', 'network', 'dns', 'host']):
            return ErrorType.CONNECTION

        # Check against known patterns
        if any(pattern in error_name for pattern in cls.NON_RETRYABLE_PATTERNS):
            return ErrorType.PERMANENT

        if any(pattern in error_name for pattern in cls.RETRYABLE_PATTERNS):
            return ErrorType.TRANSIENT

        # Default to transient for unknown errors
        return ErrorType.TRANSIENT

    @classmethod
    def is_retryable(cls, error: Exception, policy: RetryPolicy) -> bool:
        """Determine if an error should trigger a retry."""
        error_type = type(error)

        # Check explicit non-retryable exceptions
        if policy.non_retryable_exceptions and error_type in policy.non_retryable_exceptions:
            return False

        # Check explicit retryable exceptions
        if policy.retryable_exceptions and error_type in policy.retryable_exceptions:
            return True

        # Use classification
        classification = cls.classify_error(error)
        return classification in {ErrorType.TRANSIENT, ErrorType.RATE_LIMIT, ErrorType.TIMEOUT, ErrorType.CONNECTION}


def calculate_backoff_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> float:
    """Calculate delay for exponential backoff with jitter."""
    delay = min(base_delay * (exponential_base ** attempt), max_delay)

    if jitter:
        # Add random jitter to prevent thundering herd
        delay *= (0.5 + random.random() * 0.5)

    return delay


def retry_with_backoff(
    policy: Optional[RetryPolicy] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    rethrow_last_error: bool = True
):
    """Decorator for adding retry logic with exponential backoff to functions."""

    if policy is None:
        policy = RetryPolicy()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None

            for attempt in range(policy.max_retries + 1):
                try:
                    # Use circuit breaker if provided
                    if circuit_breaker:
                        with circuit_breaker:
                            return func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)

                except Exception as error:
                    last_error = error

                    # Don't retry on last attempt
                    if attempt == policy.max_retries:
                        break

                    # Check if error is retryable
                    if not ErrorClassifier.is_retryable(error, policy):
                        logger.warning(f"Non-retryable error in {func.__name__}: {error}")
                        raise error

                    # Calculate delay
                    delay = calculate_backoff_delay(
                        attempt,
                        policy.base_delay,
                        policy.max_delay,
                        policy.exponential_base,
                        policy.jitter
                    )

                    # Special handling for rate limits
                    error_type = ErrorClassifier.classify_error(error)
                    if error_type == ErrorType.RATE_LIMIT:
                        delay = max(delay, 5.0)  # Minimum 5 second delay for rate limits

                    logger.warning(
                        f"Attempt {attempt + 1}/{policy.max_retries + 1} failed for {func.__name__}: {error}. "
                        f"Retrying in {delay:.2f}s"
                    )

                    time.sleep(delay)

            # All retries exhausted
            if rethrow_last_error and last_error:
                raise last_error
            else:
                raise RetryExhaustedException(
                    f"All {policy.max_retries + 1} attempts failed for {func.__name__}",
                    policy.max_retries + 1,
                    last_error
                )

        return wrapper
    return decorator


class ConnectionPool:
    """Generic connection pool for managing persistent connections."""

    def __init__(self, max_connections: int = 10, connection_timeout: float = 30.0):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self._connections: List[Any] = []
        self._in_use: Set[Any] = set()
        self._lock = Lock()
        self._last_cleanup = time.time()

    def get_connection(self, factory: Callable[[], Any]) -> Any:
        """Get a connection from the pool or create a new one."""
        with self._lock:
            # Clean up stale connections periodically
            if time.time() - self._last_cleanup > 60:
                self._cleanup_stale_connections()

            # Try to reuse existing connection
            for conn in self._connections[:]:
                if conn not in self._in_use:
                    self._in_use.add(conn)
                    return conn

            # Create new connection if under limit
            if len(self._connections) < self.max_connections:
                conn = factory()
                self._connections.append(conn)
                self._in_use.add(conn)
                return conn

            # Pool exhausted, create temporary connection
            logger.warning(f"Connection pool exhausted, creating temporary connection")
            return factory()

    def return_connection(self, connection: Any):
        """Return a connection to the pool."""
        with self._lock:
            if connection in self._in_use:
                self._in_use.remove(connection)

    def _cleanup_stale_connections(self):
        """Remove stale connections from the pool."""
        # Implementation depends on connection type
        # For now, just update cleanup time
        self._last_cleanup = time.time()

    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._connections:
                try:
                    if hasattr(conn, 'close'):
                        conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")
            self._connections.clear()
            self._in_use.clear()


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_circuit_breaker_lock = Lock()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    with _circuit_breaker_lock:
        if name not in _circuit_breakers:
            if config is None:
                config = CircuitBreakerConfig()
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all registered circuit breakers."""
    with _circuit_breaker_lock:
        return _circuit_breakers.copy()


def reset_circuit_breaker(name: str):
    """Reset a circuit breaker to closed state."""
    with _circuit_breaker_lock:
        if name in _circuit_breakers:
            cb = _circuit_breakers[name]
            with cb._lock:
                cb.state = CircuitState.CLOSED
                cb.failure_count = 0
                cb.success_count = 0
                cb.half_open_calls = 0
            logger.info(f"Circuit breaker {name} manually reset")


# Utility functions for specific error types
def is_rate_limit_error(error: Exception) -> bool:
    """Check if error is a rate limit error."""
    return ErrorClassifier.classify_error(error) == ErrorType.RATE_LIMIT


def is_transient_error(error: Exception) -> bool:
    """Check if error is transient and retryable."""
    classification = ErrorClassifier.classify_error(error)
    return classification in {ErrorType.TRANSIENT, ErrorType.CONNECTION, ErrorType.TIMEOUT}


def get_provider_retry_policy(provider_name: str) -> RetryPolicy:
    """Get retry policy configured for specific provider from settings."""
    try:
        from ...utils.settings import (
            OPENAI_MAX_RETRIES, GOOGLE_MAX_RETRIES, OLLAMA_MAX_RETRIES,
            RETRY_BACKOFF_BASE, RETRY_BACKOFF_MAX, RETRY_EXPONENTIAL_BASE
        )
    except ImportError:
        # Fallback to defaults if settings not available
        OPENAI_MAX_RETRIES = 3
        GOOGLE_MAX_RETRIES = 3
        OLLAMA_MAX_RETRIES = 2
        RETRY_BACKOFF_BASE = 1.0
        RETRY_BACKOFF_MAX = 30.0
        RETRY_EXPONENTIAL_BASE = 2.0

    # Provider-specific policies using settings
    policies = {
        "openai": RetryPolicy(
            max_retries=OPENAI_MAX_RETRIES,
            base_delay=RETRY_BACKOFF_BASE,
            max_delay=RETRY_BACKOFF_MAX,
            exponential_base=RETRY_EXPONENTIAL_BASE
        ),
        "google": RetryPolicy(
            max_retries=GOOGLE_MAX_RETRIES,
            base_delay=RETRY_BACKOFF_BASE,
            max_delay=RETRY_BACKOFF_MAX,
            exponential_base=RETRY_EXPONENTIAL_BASE
        ),
        "ollama": RetryPolicy(
            max_retries=OLLAMA_MAX_RETRIES,  # Fewer retries for local service
            base_delay=RETRY_BACKOFF_BASE * 0.5,  # Shorter delay for local
            max_delay=10.0,
            exponential_base=1.5
        )
    }

    return policies.get(provider_name.lower(), RetryPolicy())


def get_provider_circuit_breaker_config(provider_name: str) -> CircuitBreakerConfig:
    """Get circuit breaker config for specific provider from settings."""
    try:
        from ...utils.settings import (
            OPENAI_CIRCUIT_BREAKER_THRESHOLD, GOOGLE_CIRCUIT_BREAKER_THRESHOLD,
            OLLAMA_CIRCUIT_BREAKER_THRESHOLD, OLLAMA_RESET_TIMEOUT,
            CB_SUCCESS_THRESHOLD, CB_RESET_TIMEOUT, CB_HALF_OPEN_MAX_CALLS
        )
    except ImportError:
        # Fallback to defaults if settings not available
        OPENAI_CIRCUIT_BREAKER_THRESHOLD = 5
        GOOGLE_CIRCUIT_BREAKER_THRESHOLD = 5
        OLLAMA_CIRCUIT_BREAKER_THRESHOLD = 3
        OLLAMA_RESET_TIMEOUT = 30.0
        CB_SUCCESS_THRESHOLD = 3
        CB_RESET_TIMEOUT = 60.0
        CB_HALF_OPEN_MAX_CALLS = 5

    configs = {
        "openai": CircuitBreakerConfig(
            failure_threshold=OPENAI_CIRCUIT_BREAKER_THRESHOLD,
            success_threshold=CB_SUCCESS_THRESHOLD,
            reset_timeout=CB_RESET_TIMEOUT,
            half_open_max_calls=CB_HALF_OPEN_MAX_CALLS
        ),
        "google": CircuitBreakerConfig(
            failure_threshold=GOOGLE_CIRCUIT_BREAKER_THRESHOLD,
            success_threshold=CB_SUCCESS_THRESHOLD,
            reset_timeout=CB_RESET_TIMEOUT,
            half_open_max_calls=CB_HALF_OPEN_MAX_CALLS
        ),
        "ollama": CircuitBreakerConfig(
            failure_threshold=OLLAMA_CIRCUIT_BREAKER_THRESHOLD,  # Faster failure detection for local service
            success_threshold=CB_SUCCESS_THRESHOLD,
            reset_timeout=OLLAMA_RESET_TIMEOUT,
            half_open_max_calls=CB_HALF_OPEN_MAX_CALLS // 2  # Fewer calls for local service
        )
    }

    return configs.get(provider_name.lower(), CircuitBreakerConfig())