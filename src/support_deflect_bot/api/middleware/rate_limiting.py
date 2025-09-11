"""Rate limiting middleware for Support Deflect Bot API."""

import time
from collections import defaultdict
from fastapi import FastAPI, Request, HTTPException, status

# Simple in-memory rate limiting (use Redis in production)
_rate_limit_storage = defaultdict(list)

class RateLimitMiddleware:
    """Simple rate limiting middleware."""
    
    def __init__(self, app: FastAPI, calls_per_minute: int = 60):
        self.app = app
        self.calls_per_minute = calls_per_minute
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Get client IP
        client_ip = "unknown"
        for header_name, header_value in scope.get("headers", []):
            if header_name == b"x-forwarded-for":
                client_ip = header_value.decode("utf-8").split(",")[0]
                break
            elif header_name == b"x-real-ip":
                client_ip = header_value.decode("utf-8")
                break
        
        if client_ip == "unknown":
            client_ip = scope.get("client", ["unknown", 0])[0]
        
        current_time = time.time()
        
        # Clean old entries and count recent requests
        _rate_limit_storage[client_ip] = [
            timestamp for timestamp in _rate_limit_storage[client_ip]
            if current_time - timestamp < 60  # Keep last minute
        ]
        
        # Check rate limit
        if len(_rate_limit_storage[client_ip]) >= self.calls_per_minute:
            response = {
                "error": f"Rate limit exceeded: {self.calls_per_minute} requests per minute",
                "error_type": "rate_limit_exceeded",
                "status_code": 429,
                "retry_after": 60
            }
            
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"retry-after", b"60"]
                ]
            })
            await send({
                "type": "http.response.body",
                "body": str(response).encode()
            })
            return
        
        # Add current request
        _rate_limit_storage[client_ip].append(current_time)
        
        await self.app(scope, receive, send)

def add_rate_limiting(app: FastAPI, calls_per_minute: int = 60) -> None:
    """Add rate limiting middleware to FastAPI application."""
    app.add_middleware(RateLimitMiddleware, calls_per_minute=calls_per_minute)