"""
Logging middleware for the FastAPI application.

This middleware logs all incoming requests and outgoing responses with detailed
information including timing, headers, and request correlation IDs.
"""

import logging
import time
import uuid
import os
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import json

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all HTTP requests and responses with detailed information.
    
    Provides structured logging with correlation IDs, request/response details,
    and processing time measurements while filtering sensitive data.
    """
    
    def __init__(self, app, log_level: str = "INFO"):
        super().__init__(app)
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)

        # Check if body logging is enabled via environment variable
        self.log_body = os.getenv("LOG_BODY", "true").lower() == "true"

        # Sensitive headers to filter from logs
        self.sensitive_headers = {
            'authorization', 'x-api-key', 'cookie', 'set-cookie',
            'x-auth-token', 'x-access-token', 'bearer'
        }

        # Sensitive query parameters to filter
        self.sensitive_params = {
            'api_key', 'token', 'password', 'secret', 'key'
        }

        # Sensitive body keys to redact in JSON bodies
        self.sensitive_body_keys = {
            "password", "secret", "token", "api_key", "access_token"
        }
    
    def _filter_sensitive_data(self, data: dict) -> dict:
        """Filter sensitive information from headers or query parameters."""
        if not data:
            return data

        filtered = {}
        for key, value in data.items():
            if key.lower() in self.sensitive_headers or key.lower() in self.sensitive_params:
                filtered[key] = "[FILTERED]"
            else:
                filtered[key] = value
        return filtered

    def _redact_sensitive_body(self, body_str: str) -> str:
        """Redact sensitive keys from JSON body content."""
        try:
            parsed = json.loads(body_str)
            redacted = self._redact_dict(parsed)
            return json.dumps(redacted, indent=2)
        except json.JSONDecodeError:
            # If it's not JSON, return as-is
            return body_str

    def _redact_dict(self, data) -> dict:
        """Recursively redact sensitive keys from dictionary."""
        if isinstance(data, dict):
            redacted = {}
            for key, value in data.items():
                if key.lower() in self.sensitive_body_keys:
                    redacted[key] = "[REDACTED]"
                elif isinstance(value, (dict, list)):
                    redacted[key] = self._redact_dict(value)
                else:
                    redacted[key] = value
            return redacted
        elif isinstance(data, list):
            return [self._redact_dict(item) for item in data]
        else:
            return data
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request headers."""
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
    
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Use incoming request ID when present, otherwise generate a new one
        correlation_id = (request.headers.get("x-correlation-id") or
                         request.headers.get("x-request-id") or
                         str(uuid.uuid4()))[:8]
        
        # Add correlation ID to request state for use in other parts of the app
        request.state.correlation_id = correlation_id
        
        # Record start time
        start_time = time.time()
        
        # Extract request information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Filter sensitive headers
        filtered_headers = self._filter_sensitive_data(dict(request.headers))
        
        # Filter sensitive query parameters
        filtered_params = self._filter_sensitive_data(dict(request.query_params))
        
        # Get request body (this consumes the stream, so we need to be careful)
        request_body = ""
        original_receive = None
        if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            received = False
            original_receive = request._receive

            async def receive():
                nonlocal received
                if not received:
                    received = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return {"type": "http.request", "body": b"", "more_body": False}

            request._receive = receive

            if body:
                try:
                    if len(body) <= 10000:  # Only log reasonable sized bodies
                        content_type = request.headers.get("content-type", "")
                        if "json" in content_type.lower():
                            body_str = json.dumps(json.loads(body.decode()), indent=2)
                            request_body = self._redact_sensitive_body(body_str)
                        elif any(ct in content_type.lower() for ct in ["form", "text"]):
                            request_body = body.decode("utf-8")
                        else:
                            request_body = f"[BINARY_DATA_{len(body)}_BYTES]"
                    else:
                        request_body = f"[LARGE_BODY_{len(body)}_BYTES]"
                except Exception:
                    request_body = "[ERROR_PARSING_BODY]"
        
        # Log incoming request
        request_log = {
            "event": "request_started",
            "correlation_id": correlation_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "headers": filtered_headers,
            "query_params": filtered_params,
        }

        # Only include body in log if logging is enabled and body exists
        if self.log_body and request_body:
            request_log["body"] = request_body
        
        logger.log(self.log_level, f"Request started: {request.method} {request.url.path}", 
                  extra={"request_data": request_log})
        
        # Process request
        if request.method in ["POST", "PUT", "PATCH"] and original_receive is not None:
            try:
                response = await call_next(request)
            finally:
                request._receive = original_receive
        else:
            response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        response_log = {
            "event": "request_completed",
            "correlation_id": correlation_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time_seconds": round(process_time, 4),
            "response_headers": dict(response.headers) if hasattr(response, 'headers') else {}
        }
        
        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = logging.ERROR
            log_msg = f"Request failed: {request.method} {request.url.path} - {response.status_code}"
        elif response.status_code >= 400:
            log_level = logging.WARNING
            log_msg = f"Request error: {request.method} {request.url.path} - {response.status_code}"
        else:
            log_level = self.log_level
            log_msg = f"Request completed: {request.method} {request.url.path} - {response.status_code}"
        
        logger.log(log_level, log_msg, extra={"response_data": response_log})
        
        # Add correlation ID to response headers for client tracking
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time"] = str(round(process_time, 4))
        
        return response


def logging_middleware(app, log_level: str = "INFO"):
    """
    Factory function to create and configure the logging middleware.
    
    Args:
        app: FastAPI application instance
        log_level: Logging level for request/response logs (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured LoggingMiddleware instance
    """
    return LoggingMiddleware(app, log_level=log_level)
