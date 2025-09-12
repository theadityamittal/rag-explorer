"""Security dependencies for FastAPI application."""

from typing import Optional
from fastapi import HTTPException, status, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time

# Simple rate limiting storage (in production, use Redis or similar)
_rate_limit_storage = {}

security = HTTPBearer(auto_error=False)

def get_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """Optional API key authentication."""
    # For now, just return the key if provided
    # In production, validate against a database or configuration
    return x_api_key

async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = security,
    api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """Verify API key from header or bearer token."""
    # Check X-API-Key header first
    if api_key:
        return api_key
    
    # Check Authorization header
    if credentials:
        return credentials.credentials
    
    # No authentication required for now
    return None

def rate_limiter(
    request: Request,
    limit_per_minute: int = 60
) -> None:
    """Simple rate limiting based on client IP."""
    client_ip = request.client.host if request.client else "unknown"
    current_time = int(time.time() / 60)  # Current minute
    
    # Initialize or get current count for this IP and minute
    key = f"{client_ip}:{current_time}"
    current_count = _rate_limit_storage.get(key, 0)
    
    if current_count >= limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit_per_minute} requests per minute"
        )
    
    # Increment counter
    _rate_limit_storage[key] = current_count + 1
    
    # Cleanup old entries (keep only last 2 minutes)
    cleanup_keys = []
    for stored_key in _rate_limit_storage.keys():
        if stored_key.split(':')[1] != str(current_time) and stored_key.split(':')[1] != str(current_time - 1):
            cleanup_keys.append(stored_key)
    
    for cleanup_key in cleanup_keys:
        _rate_limit_storage.pop(cleanup_key, None)

def check_content_type(request: Request) -> None:
    """Validate content type for POST requests."""
    if request.method in ["POST", "PUT", "PATCH"]:
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Content-Type must be application/json"
            )

def validate_request_size(request: Request, max_size_mb: int = 10) -> None:
    """Validate request body size."""
    content_length = request.headers.get("content-length")
    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request too large: {size_mb:.2f}MB > {max_size_mb}MB"
            )

class SecurityHeaders:
    """Security headers dependency."""
    
    def __init__(self, request: Request):
        self.request = request
    
    def apply_security_headers(self, response):
        """Apply security headers to response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response