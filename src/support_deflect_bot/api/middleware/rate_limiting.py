"""
Rate limiting middleware for the FastAPI application.

This middleware implements a token bucket algorithm to limit requests per IP address
with configurable limits and time windows.
"""

import logging
import time
import threading
from typing import Callable, Dict, Tuple
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
import os

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket implementation for rate limiting.
    
    Each bucket has a capacity and refill rate. Tokens are consumed for each request
    and refilled at a constant rate over time.
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens the bucket can hold
            refill_rate: Number of tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough tokens available
        """
        with self.lock:
            now = time.time()
            
            # Add tokens based on time elapsed
            time_passed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + (time_passed * self.refill_rate)
            )
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_status(self) -> Dict[str, float]:
        """Get current bucket status."""
        with self.lock:
            now = time.time()
            time_passed = now - self.last_refill
            current_tokens = min(
                self.capacity,
                self.tokens + (time_passed * self.refill_rate)
            )
            
            return {
                "tokens": current_tokens,
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
                "time_to_refill": max(0, (1 - current_tokens) / self.refill_rate) if current_tokens < 1 else 0
            }


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to implement rate limiting using token bucket algorithm.
    
    Tracks requests per IP address and enforces configurable rate limits
    with proper HTTP 429 responses when limits are exceeded.
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        cleanup_interval: int = 300  # 5 minutes
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.cleanup_interval = cleanup_interval
        
        # Storage for token buckets per IP
        self.buckets: Dict[str, TokenBucket] = {}
        self.bucket_lock = threading.Lock()
        
        # Track last access time for cleanup
        self.last_access: Dict[str, float] = defaultdict(float)
        self.last_cleanup = time.time()
        
        # Whitelist for IPs that should bypass rate limiting
        self.whitelist = set()
        whitelist_env = os.getenv("RATE_LIMIT_WHITELIST", "")
        if whitelist_env:
            self.whitelist = set(ip.strip() for ip in whitelist_env.split(","))
        
        logger.info(
            f"Rate limiting initialized: {requests_per_minute} requests/minute, "
            f"burst size: {burst_size}, whitelist: {len(self.whitelist)} IPs"
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first (for reverse proxies)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_or_create_bucket(self, ip: str) -> TokenBucket:
        """Get existing bucket or create new one for IP address."""
        with self.bucket_lock:
            if ip not in self.buckets:
                # Create new bucket with burst capacity and refill rate
                refill_rate = self.requests_per_minute / 60.0  # tokens per second
                self.buckets[ip] = TokenBucket(
                    capacity=self.burst_size,
                    refill_rate=refill_rate
                )
            
            self.last_access[ip] = time.time()
            return self.buckets[ip]
    
    def _cleanup_old_buckets(self):
        """Remove buckets for IPs that haven't been seen recently."""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        with self.bucket_lock:
            # Remove buckets older than cleanup interval
            old_ips = [
                ip for ip, last_time in self.last_access.items()
                if now - last_time > self.cleanup_interval
            ]
            
            for ip in old_ips:
                self.buckets.pop(ip, None)
                self.last_access.pop(ip, None)
            
            if old_ips:
                logger.debug(f"Cleaned up {len(old_ips)} old rate limit buckets")
            
            self.last_cleanup = now
    
    def _is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted for rate limiting."""
        return ip in self.whitelist or ip == "127.0.0.1" or ip == "localhost"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Periodic cleanup of old buckets
        self._cleanup_old_buckets()
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Skip rate limiting for whitelisted IPs
        if self._is_whitelisted(client_ip):
            return await call_next(request)
        
        # Get or create token bucket for this IP
        bucket = self._get_or_create_bucket(client_ip)
        
        # Try to consume a token
        if not bucket.consume(1):
            # Rate limit exceeded
            bucket_status = bucket.get_status()
            
            logger.warning(
                f"Rate limit exceeded for IP {client_ip}: "
                f"{request.method} {request.url.path}"
            )
            
            # Calculate retry-after header
            retry_after = int(bucket_status["time_to_refill"]) + 1
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "type": "RateLimitExceeded",
                        "status_code": 429,
                        "message": "Too many requests. Please try again later.",
                        "details": {
                            "limit": self.requests_per_minute,
                            "window": "1 minute",
                            "retry_after_seconds": retry_after
                        }
                    }
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + retry_after))
                }
            )
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers to response
        bucket_status = bucket.get_status()
        remaining = max(0, int(bucket_status["tokens"]))
        
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(
            int(time.time() + (60 - (time.time() % 60)))  # Next minute boundary
        )
        
        return response


def rate_limiting_middleware(
    app,
    requests_per_minute: int = 60,
    burst_size: int = 10,
    cleanup_interval: int = 300
):
    """
    Factory function to create and configure the rate limiting middleware.
    
    Args:
        app: FastAPI application instance
        requests_per_minute: Maximum requests allowed per minute per IP
        burst_size: Maximum burst requests allowed (token bucket capacity)
        cleanup_interval: Seconds between cleanup of old IP buckets
        
    Returns:
        Configured RateLimitingMiddleware instance
    """
    return RateLimitingMiddleware(
        app,
        requests_per_minute=requests_per_minute,
        burst_size=burst_size,
        cleanup_interval=cleanup_interval
    )
