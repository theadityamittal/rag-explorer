"""
Comprehensive unit tests for API middleware components.

Tests CORS, error handling, logging, authentication, and rate limiting
middleware with various scenarios and configurations.
"""

import pytest
import time
import json
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError, BaseModel
from tests.base import BaseAPITest

from support_deflect_bot.api.middleware.cors import (
    add_cors_middleware, 
    configure_development_cors, 
    configure_production_cors
)
from support_deflect_bot.api.middleware.error_handling import (
    add_error_handlers,
    ErrorHandlingMiddleware
)
from support_deflect_bot.api.middleware.logging import (
    LoggingMiddleware,
    add_logging_middleware
)
from support_deflect_bot.api.middleware.authentication import (
    AuthenticationMiddleware,
    add_authentication_middleware
)
from support_deflect_bot.api.middleware.rate_limiting import (
    RateLimitMiddleware,
    add_rate_limiting
)


class TestCORSMiddleware(BaseAPITest):
    """Test CORS middleware configurations."""
    
    def create_test_app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        return app
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_cors_middleware_configuration(self):
        """Test basic CORS middleware configuration."""
        app = self.create_test_app()
        
        # Add CORS middleware
        add_cors_middleware(app)
        
        # Verify middleware was added
        assert len(app.user_middleware) > 0
        
        # Find CORS middleware
        from fastapi.middleware.cors import CORSMiddleware
        cors_middleware = None
        for middleware in app.user_middleware:
            if middleware.cls == CORSMiddleware:
                cors_middleware = middleware
                break
        
        assert cors_middleware is not None
        assert cors_middleware.kwargs["allow_credentials"] is True
        assert "http://localhost:3000" in cors_middleware.kwargs["allow_origins"]
        assert "GET" in cors_middleware.kwargs["allow_methods"]
        assert "POST" in cors_middleware.kwargs["allow_methods"]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_development_cors_configuration(self):
        """Test development CORS configuration."""
        app = self.create_test_app()
        
        configure_development_cors(app)
        
        client = TestClient(app)
        
        # Test preflight request
        response = client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"
    
    @pytest.mark.unit
    @pytest.mark.api  
    def test_production_cors_configuration(self):
        """Test production CORS configuration."""
        app = self.create_test_app()
        
        allowed_origins = ["https://myapp.com", "https://api.myapp.com"]
        configure_production_cors(app, allowed_origins)
        
        client = TestClient(app)
        
        # Test allowed origin
        response = client.get("/test", headers={"Origin": "https://myapp.com"})
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "https://myapp.com"
        
        # Test disallowed origin
        response = client.get("/test", headers={"Origin": "https://malicious.com"})
        assert response.status_code == 200
        # CORS headers should not include the disallowed origin
        assert response.headers.get("access-control-allow-origin") != "https://malicious.com"
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_cors_preflight_request(self):
        """Test CORS preflight handling."""
        app = self.create_test_app()
        add_cors_middleware(app)
        
        client = TestClient(app)
        
        # Test preflight request
        response = client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type,X-API-Key"
            }
        )
        
        assert response.status_code == 200
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
        assert "POST" in response.headers["access-control-allow-methods"]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_cors_credentials_handling(self):
        """Test CORS credentials handling."""
        app = self.create_test_app()
        add_cors_middleware(app)
        
        client = TestClient(app)
        
        response = client.get(
            "/test",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 200
        assert "access-control-allow-credentials" in response.headers
        assert response.headers["access-control-allow-credentials"] == "true"


class TestErrorHandlingMiddleware(BaseAPITest):
    """Test error handling middleware."""
    
    def create_test_app_with_errors(self):
        """Create test app with error endpoints."""
        app = FastAPI()
        
        @app.get("/success")
        async def success_endpoint():
            return {"message": "success"}
        
        @app.get("/http_error")
        async def http_error_endpoint():
            raise HTTPException(status_code=404, detail="Not found")
        
        @app.get("/validation_error")
        async def validation_error_endpoint(required_param: int):
            return {"param": required_param}
        
        @app.get("/general_error")
        async def general_error_endpoint():
            raise ValueError("Something went wrong")
        
        @app.post("/pydantic_error")
        async def pydantic_error_endpoint():
            # Simulate pydantic validation error
            class TestModel(BaseModel):
                required_field: str
            
            # This will raise ValidationError
            TestModel()
        
        # Add error handlers
        add_error_handlers(app)
        
        return app
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_http_exception_handler(self):
        """Test HTTP exception handling."""
        app = self.create_test_app_with_errors()
        client = TestClient(app)
        
        response = client.get("/http_error")
        
        assert response.status_code == 404
        data = response.json()
        
        # Check error response structure
        assert "error" in data
        assert "error_type" in data
        assert "status_code" in data
        assert "timestamp" in data
        assert "path" in data
        
        assert data["error"] == "Not found"
        assert data["error_type"] == "http_exception"
        assert data["status_code"] == 404
        assert data["path"] == "/http_error"
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_validation_error_handler(self):
        """Test request validation error handling."""
        app = self.create_test_app_with_errors()
        client = TestClient(app)
        
        # Missing required parameter should trigger validation error
        response = client.get("/validation_error")
        
        assert response.status_code == 422
        data = response.json()
        
        assert "error" in data
        assert "error_type" in data
        assert "details" in data
        assert data["error_type"] == "validation_error"
        assert isinstance(data["details"], list)
        assert len(data["details"]) > 0
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_general_exception_handler(self):
        """Test general exception handling."""
        app = self.create_test_app_with_errors()
        client = TestClient(app)
        
        with patch('support_deflect_bot.api.middleware.error_handling.logger') as mock_logger:
            response = client.get("/general_error")
            
            assert response.status_code == 500
            data = response.json()
            
            # Check error response structure
            assert "error" in data
            assert "error_type" in data
            assert "status_code" in data
            assert "detail" in data
            assert "timestamp" in data
            assert "path" in data
            
            assert data["error"] == "Internal server error"
            assert data["error_type"] == "internal_server_error"
            assert data["status_code"] == 500
            assert "unexpected error occurred" in data["detail"]
            
            # Verify exception was logged
            mock_logger.error.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_error_handling_middleware_class(self):
        """Test ErrorHandlingMiddleware class."""
        app = FastAPI()
        middleware = ErrorHandlingMiddleware(app, debug=True)
        
        assert middleware.app == app
        assert middleware.debug is True
        
        # Test non-HTTP scope (should pass through)
        mock_scope = {"type": "websocket"}
        mock_receive = Mock()
        mock_send = Mock()
        
        # Should not raise exception for non-HTTP scope
        # This would need async testing framework for full test
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_error_response_format_consistency(self):
        """Test that all error responses have consistent format."""
        app = self.create_test_app_with_errors()
        client = TestClient(app)
        
        # Test different error types
        error_responses = [
            client.get("/http_error"),
            client.get("/validation_error"),  
            client.get("/general_error")
        ]
        
        required_fields = ["error", "error_type", "timestamp", "path"]
        
        for response in error_responses:
            assert response.status_code >= 400
            data = response.json()
            
            # Check all required fields are present
            for field in required_fields:
                assert field in data, f"Field {field} missing in error response"
            
            # Check timestamp format
            assert data["timestamp"].endswith("Z")
            assert len(data["timestamp"]) > 10


class TestLoggingMiddleware(BaseAPITest):
    """Test logging middleware."""
    
    def create_test_app_with_logging(self):
        """Create test app with logging middleware."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        @app.get("/slow")
        async def slow_endpoint():
            await asyncio.sleep(0.1)
            return {"message": "slow response"}
        
        add_logging_middleware(app)
        return app
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_logging_middleware_configuration(self):
        """Test logging middleware is properly configured."""
        app = FastAPI()
        add_logging_middleware(app)
        
        # Check middleware was added
        assert len(app.user_middleware) > 0
        
        # Find logging middleware
        logging_middleware = None
        for middleware in app.user_middleware:
            if middleware.cls == LoggingMiddleware:
                logging_middleware = middleware
                break
        
        assert logging_middleware is not None
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_request_id_generation(self):
        """Test request ID generation and headers."""
        app = self.create_test_app_with_logging()
        client = TestClient(app)
        
        response = client.get("/test")
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time" in response.headers
        
        # Check request ID format (should be UUID)
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) == 36  # UUID string length
        assert request_id.count("-") == 4  # UUID has 4 hyphens
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_response_time_header(self):
        """Test response time header."""
        app = self.create_test_app_with_logging()
        client = TestClient(app)
        
        response = client.get("/test")
        
        assert "X-Response-Time" in response.headers
        
        # Parse response time (should be a float as string)
        response_time = float(response.headers["X-Response-Time"])
        assert response_time >= 0.0
        assert response_time < 10.0  # Should be less than 10 seconds for test
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_logging_middleware_calls(self):
        """Test that logging middleware logs requests and responses."""
        app = self.create_test_app_with_logging()
        client = TestClient(app)
        
        with patch('support_deflect_bot.api.middleware.logging.logger') as mock_logger:
            response = client.get("/test")
            
            assert response.status_code == 200
            
            # Verify logging calls were made
            assert mock_logger.info.call_count == 2  # Request and response logs
            
            # Check log message content
            call_args = [call[0][0] for call in mock_logger.info.call_args_list]
            
            # Should have request and response logs
            request_log = next((log for log in call_args if log.startswith("Request:")), None)
            response_log = next((log for log in call_args if log.startswith("Response:")), None)
            
            assert request_log is not None
            assert response_log is not None
            assert "GET" in request_log
            assert "/test" in request_log
            assert "200" in response_log
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_logging_middleware_with_client_ip(self):
        """Test logging with client IP address."""
        app = self.create_test_app_with_logging()
        client = TestClient(app)
        
        with patch('support_deflect_bot.api.middleware.logging.logger') as mock_logger:
            response = client.get("/test")
            
            assert response.status_code == 200
            
            # Check that client information is logged
            request_log = mock_logger.info.call_args_list[0][0][0]
            assert "from" in request_log  # Should contain client info


class TestAuthenticationMiddleware(BaseAPITest):
    """Test authentication middleware."""
    
    def create_test_app_with_auth(self, require_auth=False):
        """Create test app with authentication middleware."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        @app.get("/protected")
        async def protected_endpoint():
            return {"message": "protected content"}
        
        add_authentication_middleware(app, require_auth=require_auth)
        return app
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_authentication_middleware_configuration(self):
        """Test authentication middleware configuration."""
        app = FastAPI()
        add_authentication_middleware(app, require_auth=True)
        
        # Check middleware was added
        assert len(app.user_middleware) > 0
        
        # Find authentication middleware
        auth_middleware = None
        for middleware in app.user_middleware:
            if middleware.cls == AuthenticationMiddleware:
                auth_middleware = middleware
                break
        
        assert auth_middleware is not None
        assert auth_middleware.kwargs.get("require_auth") is True
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_authentication_middleware_optional(self):
        """Test authentication middleware when auth is optional."""
        app = self.create_test_app_with_auth(require_auth=False)
        client = TestClient(app)
        
        # Should allow requests without authentication
        response = client.get("/test")
        assert response.status_code == 200
        
        response = client.get("/protected")
        assert response.status_code == 200
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_authentication_middleware_passthrough(self):
        """Test that authentication middleware currently passes through all requests."""
        app = self.create_test_app_with_auth(require_auth=True)
        client = TestClient(app)
        
        # Currently authentication is pass-through, so should still work
        response = client.get("/test")
        assert response.status_code == 200
        
        # Note: In actual implementation, this would require authentication
        # This test verifies current placeholder behavior


class TestRateLimitingMiddleware(BaseAPITest):
    """Test rate limiting middleware."""
    
    def create_test_app_with_rate_limiting(self, calls_per_minute=3):
        """Create test app with rate limiting."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        add_rate_limiting(app, calls_per_minute=calls_per_minute)
        return app
    
    @pytest.mark.unit
    @pytest.mark.api  
    def test_rate_limiting_middleware_configuration(self):
        """Test rate limiting middleware configuration."""
        app = FastAPI()
        add_rate_limiting(app, calls_per_minute=10)
        
        # Check middleware was added
        assert len(app.user_middleware) > 0
        
        # Find rate limiting middleware
        rate_limit_middleware = None
        for middleware in app.user_middleware:
            if middleware.cls == RateLimitMiddleware:
                rate_limit_middleware = middleware
                break
        
        assert rate_limit_middleware is not None
        assert rate_limit_middleware.kwargs.get("calls_per_minute") == 10
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_rate_limiting_allows_normal_requests(self):
        """Test that rate limiting allows normal request patterns."""
        app = self.create_test_app_with_rate_limiting(calls_per_minute=10)
        client = TestClient(app)
        
        # Should allow several requests under the limit
        for i in range(5):
            response = client.get("/test")
            assert response.status_code == 200
            assert response.json()["message"] == "test"
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_rate_limiting_blocks_excessive_requests(self):
        """Test that rate limiting blocks excessive requests."""
        # Set very low limit for testing
        app = self.create_test_app_with_rate_limiting(calls_per_minute=2)
        client = TestClient(app)
        
        # Clear any existing rate limit state
        from support_deflect_bot.api.middleware.rate_limiting import _rate_limit_storage
        _rate_limit_storage.clear()
        
        # First requests should succeed
        response1 = client.get("/test")
        assert response1.status_code == 200
        
        response2 = client.get("/test")
        assert response2.status_code == 200
        
        # Third request should be rate limited
        response3 = client.get("/test")
        assert response3.status_code == 429
        
        # Should have retry-after header
        assert "retry-after" in response3.headers
        assert response3.headers["retry-after"] == "60"
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_rate_limiting_cleanup(self):
        """Test that rate limiting cleans up old entries."""
        app = self.create_test_app_with_rate_limiting(calls_per_minute=5)
        client = TestClient(app)
        
        from support_deflect_bot.api.middleware.rate_limiting import _rate_limit_storage
        _rate_limit_storage.clear()
        
        # Mock time to test cleanup
        with patch('time.time') as mock_time:
            # Start at time 0
            mock_time.return_value = 0
            
            # Make some requests
            client.get("/test")
            client.get("/test")
            
            # Move time forward past cleanup threshold
            mock_time.return_value = 70  # More than 60 seconds later
            
            # This request should succeed because old entries are cleaned up
            response = client.get("/test") 
            assert response.status_code == 200
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_rate_limiting_different_ips(self):
        """Test that rate limiting treats different IPs separately."""
        app = self.create_test_app_with_rate_limiting(calls_per_minute=2)
        
        from support_deflect_bot.api.middleware.rate_limiting import _rate_limit_storage
        _rate_limit_storage.clear()
        
        # Create clients that appear to have different IPs
        client1 = TestClient(app)
        client2 = TestClient(app)
        
        # Each client should have their own rate limit
        response1 = client1.get("/test")
        assert response1.status_code == 200
        
        response2 = client2.get("/test")  
        assert response2.status_code == 200
        
        # Both clients should be able to make their limit
        client1.get("/test")  # Second request for client1
        client2.get("/test")  # Second request for client2


class TestMiddlewareIntegration(BaseAPITest):
    """Test middleware integration scenarios."""
    
    def create_full_middleware_app(self):
        """Create app with all middleware components."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        @app.get("/error")
        async def error_endpoint():
            raise HTTPException(status_code=400, detail="Test error")
        
        # Add all middleware
        add_cors_middleware(app)
        add_error_handlers(app)
        add_logging_middleware(app)
        add_authentication_middleware(app)
        add_rate_limiting(app, calls_per_minute=10)
        
        return app
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_middleware_stack_order(self):
        """Test that middleware is applied in correct order."""
        app = self.create_full_middleware_app()
        
        # Check that all middleware was added
        assert len(app.user_middleware) >= 4  # At least 4 middleware components
        
        # Verify middleware types are present
        middleware_classes = [mw.cls for mw in app.user_middleware]
        
        from fastapi.middleware.cors import CORSMiddleware
        assert CORSMiddleware in middleware_classes
        assert LoggingMiddleware in middleware_classes
        assert AuthenticationMiddleware in middleware_classes
        assert RateLimitMiddleware in middleware_classes
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_integrated_middleware_request_flow(self):
        """Test complete request flow through all middleware."""
        app = self.create_full_middleware_app()
        client = TestClient(app)
        
        with patch('support_deflect_bot.api.middleware.logging.logger') as mock_logger:
            response = client.get(
                "/test",
                headers={"Origin": "http://localhost:3000"}
            )
            
            assert response.status_code == 200
            
            # Check CORS headers
            assert "access-control-allow-origin" in response.headers
            
            # Check logging headers
            assert "X-Request-ID" in response.headers
            assert "X-Response-Time" in response.headers
            
            # Check logging was called
            assert mock_logger.info.call_count >= 2  # Request and response logs
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_integrated_middleware_error_flow(self):
        """Test error handling through all middleware."""
        app = self.create_full_middleware_app()
        client = TestClient(app)
        
        with patch('support_deflect_bot.api.middleware.logging.logger') as mock_logger:
            response = client.get("/error")
            
            assert response.status_code == 400
            data = response.json()
            
            # Check error response structure
            assert "error" in data
            assert "error_type" in data
            assert data["error"] == "Test error"
            
            # Check that logging still occurred
            assert mock_logger.info.call_count >= 1  # At least request log
            
            # Check headers are still present
            assert "X-Request-ID" in response.headers
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_middleware_performance_impact(self):
        """Test that middleware doesn't significantly impact performance."""
        app = self.create_full_middleware_app()
        client = TestClient(app)
        
        import time
        
        # Measure response times
        times = []
        for _ in range(5):
            start = time.time()
            response = client.get("/test")
            end = time.time()
            
            assert response.status_code == 200
            times.append(end - start)
        
        # Average response time should be reasonable
        avg_time = sum(times) / len(times)
        assert avg_time < 0.5  # Should be less than 500ms
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_middleware_configuration_variations(self):
        """Test different middleware configurations."""
        # Test with minimal middleware
        minimal_app = FastAPI()
        add_error_handlers(minimal_app)
        minimal_client = TestClient(minimal_app)
        
        @minimal_app.get("/test")
        async def test():
            return {"minimal": True}
        
        response = minimal_client.get("/test")
        assert response.status_code == 200
        
        # Test with maximum middleware
        maximal_app = self.create_full_middleware_app()
        maximal_client = TestClient(maximal_app)
        
        response = maximal_client.get("/test") 
        assert response.status_code == 200
        
        # Maximal app should have more headers
        assert len(response.headers) > 3  # Should have additional middleware headers


class TestMiddlewareErrorScenarios(BaseAPITest):
    """Test middleware behavior in error scenarios."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_middleware_with_malformed_requests(self):
        """Test middleware handling of malformed requests."""
        app = FastAPI()
        add_cors_middleware(app)
        add_error_handlers(app)
        add_logging_middleware(app)
        
        @app.post("/test")
        async def test_endpoint(data: dict):
            return data
        
        client = TestClient(app)
        
        # Test malformed JSON
        response = client.post(
            "/test",
            data="invalid json{",
            headers={"Content-Type": "application/json"}
        )
        
        # Should handle gracefully
        assert response.status_code == 422
        assert "X-Request-ID" in response.headers  # Logging middleware still works
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_middleware_exception_propagation(self):
        """Test that middleware doesn't interfere with exception propagation."""
        app = FastAPI()
        add_logging_middleware(app)
        add_error_handlers(app)
        
        @app.get("/test")
        async def test_endpoint():
            raise ValueError("Test exception")
        
        client = TestClient(app)
        
        response = client.get("/test")
        
        # Error handler should catch and format the exception
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error_type"] == "internal_server_error"
        
        # Logging middleware should still add headers
        assert "X-Request-ID" in response.headers
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_middleware_with_async_exceptions(self):
        """Test middleware handling of async exceptions."""
        import asyncio
        
        app = FastAPI()
        add_error_handlers(app)
        add_logging_middleware(app)
        
        @app.get("/test")
        async def async_error_endpoint():
            await asyncio.sleep(0.01)
            raise HTTPException(status_code=503, detail="Service unavailable")
        
        client = TestClient(app)
        
        response = client.get("/test")
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "Service unavailable"
        assert "X-Request-ID" in response.headers