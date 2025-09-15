"""
FastAPI application with middleware for error handling, logging, rate limiting, and authentication.
"""

import os
import time
import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class HealthResponse(BaseModel):
    status: str
    timestamp: float

class ErrorResponse(BaseModel):
    error: Dict[str, Any]

# Middleware classes
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for standardized error handling."""

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "type": "HTTPException",
                        "status_code": exc.status_code,
                        "message": exc.detail,
                        "path": str(request.url.path)
                    }
                }
            )
        except Exception as exc:
            logger.exception("Unhandled exception occurred")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "type": "InternalServerError",
                        "status_code": 500,
                        "message": "Internal server error",
                        "path": str(request.url.path)
                    }
                }
            )

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging and correlation tracking."""

    async def dispatch(self, request: Request, call_next):
        correlation_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.info(f"Request started - {request.method} {request.url.path} - ID: {correlation_id}")

        response = await call_next(request)

        process_time = time.time() - start_time

        logger.info(f"Request completed - {request.method} {request.url.path} - "
                   f"Status: {response.status_code} - Time: {process_time:.4f}s - ID: {correlation_id}")

        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time"] = str(process_time)

        return response

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting with whitelist support."""

    def __init__(self, app, requests_per_minute: int = 60, burst_size: int = 10):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.request_counts = {}
        self.whitelist = {"127.0.0.1", "localhost"}

        # Add whitelist from environment
        env_whitelist = os.getenv("RATE_LIMIT_WHITELIST", "")
        if env_whitelist:
            self.whitelist.update(ip.strip() for ip in env_whitelist.split(","))

    def get_client_ip(self, request: Request) -> str:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        client_ip = self.get_client_ip(request)

        # Skip rate limiting for whitelisted IPs
        if client_ip in self.whitelist:
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
            return response

        current_time = time.time()
        minute_window = int(current_time // 60)

        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = {}

        # Clean old windows
        self.request_counts[client_ip] = {
            window: count for window, count in self.request_counts[client_ip].items()
            if window >= minute_window - 1
        }

        current_count = self.request_counts[client_ip].get(minute_window, 0)

        if current_count >= self.requests_per_minute:
            retry_after = 60 - (current_time % 60)
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "type": "RateLimitExceeded",
                        "status_code": 429,
                        "message": "Rate limit exceeded",
                        "path": str(request.url.path)
                    }
                },
                headers={"Retry-After": str(int(retry_after))}
            )

        # Update count
        self.request_counts[client_ip][minute_window] = current_count + 1

        response = await call_next(request)

        # Add rate limit headers
        remaining = max(0, self.requests_per_minute - current_count - 1)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str((minute_window + 1) * 60)

        return response

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""

    def __init__(self, app, require_auth: bool = False):
        super().__init__(app)
        self.require_auth = require_auth
        self.api_keys = set()

        # Load API keys from environment
        env_keys = os.getenv("API_KEYS", "")
        if env_keys:
            self.api_keys.update(key.strip() for key in env_keys.split(","))

        self.public_paths = {"/", "/health", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths or if auth is disabled
        if not self.require_auth or request.url.path in self.public_paths:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")

        if not api_key or api_key not in self.api_keys:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "type": "AuthenticationRequired",
                        "status_code": 401,
                        "message": "Valid API key required",
                        "path": str(request.url.path)
                    }
                }
            )

        return await call_next(request)

# Create FastAPI app
app = FastAPI(
    title="RAG Explorer API",
    description="API for RAG Explorer with comprehensive middleware",
    version="1.0.0",
    debug=True
)

# Add middleware (order matters - last added is executed first)
app.add_middleware(AuthenticationMiddleware, require_auth=False)  # Disabled for development
app.add_middleware(RateLimitingMiddleware, requests_per_minute=60, burst_size=10)
app.add_middleware(LoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)

# Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "RAG Explorer API", "status": "running"}

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="healthy", timestamp=time.time())

@app.post("/query")
async def query(request: QueryRequest):
    """Query endpoint for RAG operations."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Simulate processing time
    await asyncio.sleep(0.1)

    return {
        "question": request.question,
        "answer": "This is a simulated response from the RAG system.",
        "sources": [],
        "timestamp": time.time()
    }

@app.get("/docs")
async def docs():
    """Custom docs endpoint."""
    return {"message": "API documentation available at /docs"}

@app.get("/redoc")
async def redoc():
    """Custom redoc endpoint."""
    return {"message": "API documentation available at /redoc"}

@app.get("/openapi.json")
async def openapi():
    """OpenAPI schema endpoint."""
    return app.openapi()

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "type": "HTTPException",
                "status_code": 404,
                "message": "Not Found",
                "path": str(request.url.path)
            }
        }
    )

@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "ValidationError",
                "status_code": 422,
                "message": "Validation failed",
                "path": str(request.url.path),
                "details": getattr(exc, 'errors', lambda: [])()
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)