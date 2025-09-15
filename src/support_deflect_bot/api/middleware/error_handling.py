"""
Error handling middleware for the FastAPI application.

This middleware catches all unhandled exceptions, formats them appropriately,
and returns proper HTTP error responses with detailed logging.
"""

import logging
import traceback
import uuid
from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import os

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle all unhandled exceptions in the application.
    
    Catches exceptions, logs them appropriately, and returns formatted
    error responses while sanitizing sensitive information in production.
    """
    
    def __init__(self, app, debug: bool = False):
        super().__init__(app)
        self.debug = debug or os.getenv("DEBUG", "false").lower() == "true"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except RequestValidationError as exc:
            logger.warning(
                f"Request validation error for {request.method} {request.url}",
                exc_info=self.debug
            )
            correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4())[:8])
            return JSONResponse(
                status_code=422,
                content={
                    "detail": "Request validation failed",
                    "error": {
                        "type": "RequestValidationError",
                        "status_code": 422,
                        "message": "Request validation failed",
                        **({"details": exc.errors()} if self.debug else {}),
                        "path": str(request.url.path),
                    }
                },
                headers={"X-Correlation-ID": correlation_id}
            )
        except HTTPException as exc:
            # FastAPI HTTP exceptions - pass through with proper formatting
            logger.warning(
                f"HTTP Exception: {exc.status_code} - {exc.detail} "
                f"for {request.method} {request.url}"
            )
            correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4())[:8])
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "detail": exc.detail,
                    "error": {
                        "type": "HTTPException",
                        "status_code": exc.status_code,
                        "message": exc.detail,
                        "path": str(request.url.path),
                    },
                },
                headers={"X-Correlation-ID": correlation_id}
            )
        except ValueError as exc:
            # Validation and value errors
            logger.error(
                f"Validation Error: {str(exc)} for {request.method} {request.url}",
                exc_info=self.debug
            )
            correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4())[:8])
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "ValidationError",
                        "status_code": 400,
                        "message": str(exc) if self.debug else "Invalid request data",
                        "path": str(request.url.path)
                    }
                },
                headers={"X-Correlation-ID": correlation_id}
            )
        except PermissionError as exc:
            # Permission/authorization errors
            logger.error(
                f"Permission Error: {str(exc)} for {request.method} {request.url}",
                exc_info=self.debug
            )
            correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4())[:8])
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "type": "PermissionError",
                        "status_code": 403,
                        "message": "Access denied",
                        "path": str(request.url.path)
                    }
                },
                headers={"X-Correlation-ID": correlation_id}
            )
        except Exception as exc:
            # All other unhandled exceptions
            error_id = id(exc)  # Simple error ID for tracking
            logger.error(
                f"Unhandled Exception (ID: {error_id}): {type(exc).__name__}: {str(exc)} "
                f"for {request.method} {request.url}",
                exc_info=True
            )

            correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4())[:8])

            # Detailed error info for debug mode
            error_detail = {
                "error": {
                    "type": "InternalServerError",
                    "status_code": 500,
                    "message": "An internal server error occurred",
                    "error_id": error_id,
                    "path": str(request.url.path)
                }
            }

            if self.debug:
                error_detail["error"].update({
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc().split('\n')
                })

            return JSONResponse(
                status_code=500,
                content=error_detail,
                headers={"X-Correlation-ID": correlation_id}
            )


def error_handler_middleware(app, debug: bool = False):
    """
    Factory function to create and configure the error handler middleware.
    
    Args:
        app: FastAPI application instance
        debug: Whether to include detailed error information in responses
        
    Returns:
        Configured ErrorHandlerMiddleware instance
    """
    return ErrorHandlerMiddleware(app, debug=debug)
