"""
Authentication middleware for the FastAPI application.

This middleware supports API key authentication and optional JWT token authentication
with flexible configuration and proper error responses.
"""

import logging
import os
from typing import Callable, Optional, Set, Dict, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timezone

try:
    import jwt  # type: ignore
except Exception:
    jwt = None

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle authentication using API keys and optional JWT tokens.
    
    Supports multiple authentication schemes and provides user context
    for authenticated requests with proper error responses for unauthorized access.
    """
    
    def __init__(
        self,
        app,
        api_keys: Optional[Set[str]] = None,
        jwt_secret: Optional[str] = None,
        jwt_algorithm: str = "HS256",
        require_auth: bool = True,
        public_paths: Optional[Set[str]] = None
    ):
        super().__init__(app)
        
        # API key configuration
        self.api_keys = api_keys or set()
        
        # Load API keys from environment if not provided
        if not self.api_keys:
            env_keys = os.getenv("API_KEYS", "")
            if env_keys:
                self.api_keys = set(key.strip() for key in env_keys.split(",") if key.strip())
        
        # JWT configuration
        self.jwt_secret = jwt_secret or os.getenv("JWT_SECRET")
        self.jwt_algorithm = jwt_algorithm
        
        # Authentication requirements
        self.require_auth = require_auth
        
        # Public paths that don't require authentication
        self.public_paths = public_paths or {
            "/", "/health", "/docs", "/redoc", "/openapi.json"
        }
        
        # Add common health check and documentation paths
        self.public_paths.update({
            "/favicon.ico", "/robots.txt", "/sitemap.xml"
        })
        
        logger.info(
            f"Authentication middleware initialized: "
            f"API keys: {len(self.api_keys)}, "
            f"JWT enabled: {bool(self.jwt_secret)}, "
            f"Require auth: {require_auth}, "
            f"Public paths: {len(self.public_paths)}"
        )
    
    def _is_public_path(self, path: str) -> bool:
        """Check if the request path is public (doesn't require authentication)."""
        # Exact match
        if path in self.public_paths:
            return True
        
        # Pattern matching for common public paths
        public_patterns = [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/static/",
            "/.well-known/"
        ]
        
        return any(path.startswith(pattern) for pattern in public_patterns)
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request headers or query parameters."""
        # Check Authorization header (ApiKey scheme only)
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("ApiKey "):
            return auth_header[7:]  # Remove "ApiKey " prefix
        
        # Check X-API-Key header
        api_key_header = request.headers.get("x-api-key")
        if api_key_header:
            return api_key_header
        
        # Check query parameter
        api_key_param = request.query_params.get("api_key")
        if api_key_param:
            return api_key_param
        
        return None
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate the provided API key."""
        return api_key in self.api_keys
    
    def _extract_jwt_token(self, request: Request) -> Optional[str]:
        """Extract JWT token from request headers."""
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            # Simple check to see if it looks like a JWT (has dots)
            if token.count('.') == 2:
                return token
        
        return None
    
    def _validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return payload if valid."""
        if not self.jwt_secret or jwt is None:
            return None
        
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            
            # Check expiration
            if 'exp' in payload:
                exp_timestamp = payload['exp']
                if datetime.now(timezone.utc).timestamp() > exp_timestamp:
                    logger.warning("JWT token has expired")
                    return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error validating JWT token: {e}")
            return None
    
    def _create_auth_error_response(self, message: str, details: Optional[str] = None) -> JSONResponse:
        """Create standardized authentication error response."""
        content = {
            "error": {
                "type": "AuthenticationError",
                "status_code": 401,
                "message": message
            }
        }
        
        if details:
            content["error"]["details"] = details
        
        return JSONResponse(
            status_code=401,
            content=content,
            headers={
                "WWW-Authenticate": 'Bearer realm="API", ApiKey realm="API"'
            }
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip authentication for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)
        
        # Skip authentication if not required
        if not self.require_auth:
            return await call_next(request)
        
        # Initialize user context
        user_context = {
            "authenticated": False,
            "auth_method": None,
            "user_id": None,
            "api_key": None,
            "jwt_payload": None
        }
        
        # Try API key authentication first
        api_key = self._extract_api_key(request)
        if api_key:
            if self._validate_api_key(api_key):
                user_context.update({
                    "authenticated": True,
                    "auth_method": "api_key",
                    "api_key": api_key[:8] + "..." if len(api_key) > 8 else api_key  # Truncated for logging
                })
                
                # Add user context to request state
                request.state.user = user_context
                
                logger.debug(f"API key authentication successful for {request.url.path}")
                return await call_next(request)
            else:
                logger.warning(f"Invalid API key attempted for {request.url.path}")
                return self._create_auth_error_response(
                    "Invalid API key",
                    "The provided API key is not valid"
                )
        
        # Try JWT authentication if JWT secret is configured
        if self.jwt_secret:
            jwt_token = self._extract_jwt_token(request)
            if jwt_token:
                jwt_payload = self._validate_jwt_token(jwt_token)
                if jwt_payload:
                    user_context.update({
                        "authenticated": True,
                        "auth_method": "jwt",
                        "user_id": jwt_payload.get("sub"),
                        "jwt_payload": jwt_payload
                    })
                    
                    # Add user context to request state
                    request.state.user = user_context
                    
                    logger.debug(f"JWT authentication successful for {request.url.path}")
                    return await call_next(request)
                else:
                    logger.warning(f"Invalid JWT token attempted for {request.url.path}")
                    return self._create_auth_error_response(
                        "Invalid or expired token",
                        "The provided JWT token is invalid or has expired"
                    )
        
        # No valid authentication found
        logger.warning(f"Unauthenticated request to {request.url.path}")
        
        # Determine appropriate error message based on available auth methods
        if self.api_keys and self.jwt_secret:
            message = "Authentication required. Provide a valid API key or JWT token."
            details = "Use 'Authorization: Bearer <jwt_token>' or 'Authorization: ApiKey <api_key>' or 'X-API-Key: <key>' header"
        elif self.api_keys:
            message = "API key required"
            details = "Use 'Authorization: ApiKey <api_key>' or 'X-API-Key: <key>' header"
        elif self.jwt_secret:
            message = "JWT token required"
            details = "Use 'Authorization: Bearer <jwt_token>' header"
        else:
            message = "Authentication is required but no authentication methods are configured"
            details = None
        
        return self._create_auth_error_response(message, details)


def authentication_middleware(
    app,
    api_keys: Optional[Set[str]] = None,
    jwt_secret: Optional[str] = None,
    jwt_algorithm: str = "HS256",
    require_auth: bool = True,
    public_paths: Optional[Set[str]] = None
):
    """
    Factory function to create and configure the authentication middleware.
    
    Args:
        app: FastAPI application instance
        api_keys: Set of valid API keys for authentication
        jwt_secret: Secret key for JWT token validation
        jwt_algorithm: Algorithm used for JWT token validation
        require_auth: Whether authentication is required for non-public paths
        public_paths: Set of paths that don't require authentication
        
    Returns:
        Configured AuthenticationMiddleware instance
    """
    return AuthenticationMiddleware(
        app,
        api_keys=api_keys,
        jwt_secret=jwt_secret,
        jwt_algorithm=jwt_algorithm,
        require_auth=require_auth,
        public_paths=public_paths
    )
