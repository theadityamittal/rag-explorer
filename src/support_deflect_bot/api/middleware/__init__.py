"""API middleware for Support Deflect Bot."""

from .cors import add_cors_middleware, configure_development_cors, configure_production_cors
from .error_handling import add_error_handlers, ErrorHandlingMiddleware
from .rate_limiting import add_rate_limiting, RateLimitMiddleware
from .authentication import add_authentication_middleware, AuthenticationMiddleware
from .logging import add_logging_middleware, LoggingMiddleware

__all__ = [
    "add_cors_middleware",
    "configure_development_cors",
    "configure_production_cors",
    "add_error_handlers",
    "ErrorHandlingMiddleware",
    "add_rate_limiting",
    "RateLimitMiddleware",
    "add_authentication_middleware",
    "AuthenticationMiddleware",
    "add_logging_middleware",
    "LoggingMiddleware",
]