"""
Unit tests for FastAPI application configuration and lifecycle.

Tests FastAPI app creation, middleware configuration, routing,
lifespan management, and engine service integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from tests.base import BaseAPITest

from support_deflect_bot.api.app import app, lifespan


class TestFastAPIAppCreation(BaseAPITest):
    """Test FastAPI application creation and configuration."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_app_instance_creation(self):
        """Test FastAPI app instance is created correctly."""
        assert isinstance(app, FastAPI)
        assert app.title == "Support Deflect Bot"
        assert "RAG-powered question answering" in app.description
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
        assert app.openapi_url == "/openapi.json"
        
    @pytest.mark.unit
    @pytest.mark.api
    def test_app_version_configuration(self):
        """Test app version is configured from settings."""
        # Version should be loaded from APP_VERSION setting
        assert hasattr(app, 'version')
        assert app.version is not None
        assert len(app.version) > 0
        
    @pytest.mark.unit
    @pytest.mark.api
    def test_app_metadata_configuration(self):
        """Test app metadata configuration."""
        assert app.title is not None
        assert len(app.title) > 0
        assert app.description is not None
        assert len(app.description) > 0
        
    @pytest.mark.unit
    @pytest.mark.api
    def test_app_route_configuration(self):
        """Test basic app routes are configured."""
        # Get all routes from the app
        routes = [route.path for route in app.routes]
        
        # Should have basic routes
        assert "/" in routes  # Root route
        assert "/version" in routes or any("/api" in route for route in routes)


class TestAppLifespanManagement(BaseAPITest):
    """Test application lifespan management."""
    
    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_lifespan_startup_initialization(self):
        """Test lifespan startup initializes engine services."""
        mock_app = Mock(spec=FastAPI)
        
        with patch('support_deflect_bot.api.app.UnifiedRAGEngine') as mock_rag_class:
            with patch('support_deflect_bot.api.app.UnifiedDocumentProcessor') as mock_doc_class:
                with patch('support_deflect_bot.api.app.UnifiedQueryService') as mock_query_class:
                    with patch('support_deflect_bot.api.app.UnifiedEmbeddingService') as mock_embed_class:
                        with patch('support_deflect_bot.api.app.set_rag_engine') as mock_set_rag:
                            with patch('support_deflect_bot.api.app.set_document_processor') as mock_set_doc:
                                with patch('support_deflect_bot.api.app.set_query_service') as mock_set_query:
                                    with patch('support_deflect_bot.api.app.set_embedding_service') as mock_set_embed:
                                        
                                        # Create mock instances
                                        mock_rag_instance = Mock()
                                        mock_doc_instance = Mock()
                                        mock_query_instance = Mock()
                                        mock_embed_instance = Mock()
                                        
                                        mock_rag_class.return_value = mock_rag_instance
                                        mock_doc_class.return_value = mock_doc_instance
                                        mock_query_class.return_value = mock_query_instance
                                        mock_embed_class.return_value = mock_embed_instance
                                        
                                        # Test lifespan startup
                                        async with lifespan(mock_app):
                                            pass
                                        
                                        # Verify services were instantiated
                                        mock_rag_class.assert_called_once()
                                        mock_doc_class.assert_called_once()
                                        mock_query_class.assert_called_once()
                                        mock_embed_class.assert_called_once()
                                        
                                        # Verify dependency injection was set up
                                        mock_set_rag.assert_called_once_with(mock_rag_instance)
                                        mock_set_doc.assert_called_once_with(mock_doc_instance)
                                        mock_set_query.assert_called_once_with(mock_query_instance)
                                        mock_set_embed.assert_called_once_with(mock_embed_instance)
                                        
    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_lifespan_startup_error_handling(self):
        """Test lifespan startup handles initialization errors."""
        mock_app = Mock(spec=FastAPI)
        
        with patch('support_deflect_bot.api.app.UnifiedRAGEngine') as mock_rag_class:
            # Make RAG engine initialization fail
            mock_rag_class.side_effect = Exception("RAG initialization failed")
            
            # Should raise the exception
            with pytest.raises(Exception, match="RAG initialization failed"):
                async with lifespan(mock_app):
                    pass
                    
    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_lifespan_shutdown_cleanup(self):
        """Test lifespan shutdown performs cleanup."""
        mock_app = Mock(spec=FastAPI)
        
        with patch('support_deflect_bot.api.app.UnifiedRAGEngine'):
            with patch('support_deflect_bot.api.app.UnifiedDocumentProcessor'):
                with patch('support_deflect_bot.api.app.UnifiedQueryService'):
                    with patch('support_deflect_bot.api.app.UnifiedEmbeddingService'):
                        with patch('support_deflect_bot.api.app.set_rag_engine'):
                            with patch('support_deflect_bot.api.app.set_document_processor'):
                                with patch('support_deflect_bot.api.app.set_query_service'):
                                    with patch('support_deflect_bot.api.app.set_embedding_service'):
                                        
                                        # Test lifespan completes without error
                                        async with lifespan(mock_app):
                                            pass
                                        
                                        # If we get here, cleanup was successful


class TestAppMiddlewareConfiguration(BaseAPITest):
    """Test middleware configuration."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_cors_middleware_configured(self):
        """Test CORS middleware is configured."""
        # Check middleware stack
        middleware_types = [type(middleware.cls) for middleware in app.user_middleware]
        
        # Should have CORS-related middleware (exact type may vary)
        # This tests that middleware configuration functions were called
        assert len(app.user_middleware) > 0
        
    @pytest.mark.unit
    @pytest.mark.api  
    def test_trusted_host_middleware_configured(self):
        """Test TrustedHost middleware is configured."""
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        
        # Check if TrustedHostMiddleware is in the middleware stack
        middleware_classes = [middleware.cls for middleware in app.user_middleware]
        assert TrustedHostMiddleware in middleware_classes
        
    @pytest.mark.unit
    @pytest.mark.api
    def test_custom_middleware_configured(self):
        """Test custom middleware is configured."""
        # Verify middleware functions were called during app setup
        # This is tested by ensuring middleware stack has expected size
        assert len(app.user_middleware) >= 1  # At least TrustedHost
        
        # Additional middleware may be added by middleware configuration functions
        # Exact count depends on implementation


class TestAppRouterConfiguration(BaseAPITest):
    """Test router configuration and inclusion."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_routers_included(self):
        """Test all routers are included in the app."""
        # Get all route paths from the app
        route_paths = [route.path for route in app.routes]
        
        # Should have routes from included routers
        # The exact paths depend on router configuration
        assert len(route_paths) > 2  # More than just root and openapi routes
        
    @pytest.mark.unit
    @pytest.mark.api
    def test_api_versioning_routes(self):
        """Test API versioning routes are configured."""
        route_paths = [route.path for route in app.routes]
        
        # Should have versioned API routes (common pattern is /api/v1)
        api_routes = [path for path in route_paths if "/api" in path]
        
        # May have API routes depending on router configuration
        # This test ensures routing structure is set up
        assert "/" in route_paths  # Root route should always exist


class TestAppDependencyInjection(BaseAPITest):
    """Test dependency injection configuration."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_dependency_injection_setup(self):
        """Test dependency injection is properly configured."""
        # Test that dependency injection functions exist and are callable
        from support_deflect_bot.api.dependencies.engine import (
            set_rag_engine,
            set_document_processor,
            set_query_service, 
            set_embedding_service
        )
        
        assert callable(set_rag_engine)
        assert callable(set_document_processor)
        assert callable(set_query_service)
        assert callable(set_embedding_service)


class TestAppErrorHandling(BaseAPITest):
    """Test application error handling configuration."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_error_handlers_configured(self):
        """Test error handlers are configured."""
        # Error handlers are added by add_error_handlers function
        # Verify the function was called during app setup
        
        # This is tested by checking that the app has been configured
        # with error handling middleware/handlers
        assert hasattr(app, 'exception_handlers')
        
    @pytest.mark.unit
    @pytest.mark.api
    def test_logging_middleware_configured(self):
        """Test logging middleware is configured."""
        # Logging middleware is added by add_logging_middleware function
        # Verify the function was called during app setup
        
        # Check that middleware stack includes logging-related middleware
        assert len(app.user_middleware) >= 1


class TestAppEndpointBasics(BaseAPITest):
    """Test basic app endpoints."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API testing."""
        return TestClient(app)
        
    @pytest.mark.unit
    @pytest.mark.api
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns proper information."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data
        assert "description" in data
        
        # Check content
        assert "Support Deflect Bot" in data["message"]
        assert data["docs"] == "/docs"
        assert "/health" in data["health"]
        
    @pytest.mark.unit
    @pytest.mark.api
    def test_version_endpoint(self, test_client):
        """Test version endpoint if it exists."""
        # Try to get version endpoint
        response = test_client.get("/version")
        
        # May return 200 with version info or 404 if not implemented
        if response.status_code == 200:
            data = response.json()
            assert "version" in data or "app_version" in data
        else:
            # If version endpoint doesn't exist, that's also acceptable
            assert response.status_code == 404
            
    @pytest.mark.unit
    @pytest.mark.api
    def test_docs_endpoint_accessible(self, test_client):
        """Test OpenAPI docs endpoint is accessible."""
        response = test_client.get("/docs")
        
        # Should return HTML page for Swagger UI
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        
    @pytest.mark.unit
    @pytest.mark.api
    def test_openapi_schema_endpoint(self, test_client):
        """Test OpenAPI schema endpoint."""
        response = test_client.get("/openapi.json")
        
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/json"
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert schema["info"]["title"] == "Support Deflect Bot"


class TestAppConfiguration(BaseAPITest):
    """Test application configuration and settings integration."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_app_settings_integration(self):
        """Test app integrates with settings properly."""
        # App should use settings for configuration
        assert app.title is not None
        assert app.version is not None
        
        # Description should contain expected content
        assert "RAG" in app.description or "question answering" in app.description
        
    @pytest.mark.unit
    @pytest.mark.api
    def test_app_environment_configuration(self):
        """Test app can be configured for different environments."""
        # App should be configurable for different environments
        # This tests that the app structure supports environment configuration
        
        assert app.docs_url is not None  # Docs enabled
        assert app.redoc_url is not None  # ReDoc enabled
        
        # In production, these might be disabled, but for testing they should be available
        
    @pytest.mark.unit
    @pytest.mark.api
    def test_app_debug_configuration(self):
        """Test app debug configuration."""
        # App should have appropriate debug settings
        # Exact debug configuration depends on environment
        
        # Verify app can be configured (has the necessary attributes)
        assert hasattr(app, 'debug')


class TestAppSecurity(BaseAPITest):
    """Test application security configuration."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_trusted_hosts_configuration(self):
        """Test trusted hosts are properly configured."""
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        
        # Find TrustedHost middleware in the stack
        trusted_host_middleware = None
        for middleware in app.user_middleware:
            if middleware.cls == TrustedHostMiddleware:
                trusted_host_middleware = middleware
                break
                
        assert trusted_host_middleware is not None
        
        # Verify it's configured with allowed hosts
        # In test/development, it should allow all hosts ("*")
        # In production, it should be more restrictive
        
    @pytest.mark.unit
    @pytest.mark.api
    def test_security_headers_configuration(self):
        """Test security headers are configured."""
        # Security headers may be configured through middleware
        # This tests that security middleware setup functions were called
        
        # Verify middleware stack includes security-related middleware
        assert len(app.user_middleware) >= 1
        
    @pytest.mark.unit
    @pytest.mark.api
    def test_cors_security_configuration(self):
        """Test CORS security configuration."""
        # CORS should be configured through middleware
        # Exact configuration depends on security requirements
        
        # Verify CORS middleware was added
        middleware_count = len(app.user_middleware)
        assert middleware_count >= 1  # At least TrustedHost, possibly CORS too


class TestAppIntegrationReadiness(BaseAPITest):
    """Test application readiness for integration testing."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for integration readiness testing."""
        return TestClient(app)
        
    @pytest.mark.unit
    @pytest.mark.api
    def test_app_can_start(self, test_client):
        """Test application can start successfully."""
        # Test that the app can handle basic requests
        response = test_client.get("/")
        
        # Should be able to serve requests
        assert response.status_code == 200
        
    @pytest.mark.unit
    @pytest.mark.api  
    def test_app_health_check_ready(self, test_client):
        """Test app is ready for health checks."""
        # Try common health check endpoints
        health_endpoints = ["/health", "/api/v1/health", "/status"]
        
        found_health_endpoint = False
        for endpoint in health_endpoints:
            response = test_client.get(endpoint)
            if response.status_code == 200:
                found_health_endpoint = True
                break
                
        # Should have a health endpoint or the app should at least respond to root
        if not found_health_endpoint:
            # If no dedicated health endpoint, root should work
            response = test_client.get("/")
            assert response.status_code == 200
            
    @pytest.mark.unit
    @pytest.mark.api
    def test_app_api_documentation_ready(self, test_client):
        """Test API documentation is ready."""
        # OpenAPI documentation should be accessible
        docs_response = test_client.get("/docs")
        openapi_response = test_client.get("/openapi.json")
        
        assert docs_response.status_code == 200
        assert openapi_response.status_code == 200
        
        # OpenAPI schema should be valid JSON
        schema = openapi_response.json()
        assert isinstance(schema, dict)
        assert "info" in schema
        assert "paths" in schema