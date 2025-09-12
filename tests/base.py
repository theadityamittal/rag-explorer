"""
Base test classes for Support Deflect Bot unified architecture testing.

This module provides base classes that establish common testing patterns
and utilities for different components of the system.
"""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch
import pytest


class BaseEngineTest:
    """Base test class for unified engine services."""
    
    def setup_method(self):
        """Set up test method with common mocks."""
        self.mock_settings = {}
        self.mock_chroma_client = Mock()
        self.mock_provider = Mock()
        
    def teardown_method(self):
        """Clean up after test method."""
        pass
        
    def create_mock_provider(self, name: str = "test_provider") -> Mock:
        """Create a mock provider with standard interface."""
        mock_provider = Mock()
        mock_provider.name = name
        mock_provider.is_available = Mock(return_value=True)
        mock_provider.generate_response = AsyncMock()
        mock_provider.generate_embeddings = AsyncMock()
        return mock_provider
        
    def assert_async_called_with(self, mock_method, *args, **kwargs):
        """Assert that an async mock was called with specific arguments."""
        mock_method.assert_called_with(*args, **kwargs)


class BaseProviderTest:
    """Base test class for AI provider implementations."""
    
    def setup_method(self):
        """Set up test method with provider-specific mocks."""
        self.mock_http_client = Mock()
        self.mock_response = Mock()
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {"status": "success"}
        
    def teardown_method(self):
        """Clean up after test method."""
        pass
        
    def create_mock_response(self, content: str, model: str = "test-model") -> Dict[str, Any]:
        """Create a mock API response."""
        return {
            "choices": [{
                "message": {
                    "content": content,
                    "role": "assistant"
                }
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": len(content.split()),
                "total_tokens": 100 + len(content.split())
            },
            "model": model
        }
        
    def assert_provider_available(self, provider, expected: bool = True):
        """Assert provider availability status."""
        assert provider.is_available() == expected
        
    async def assert_provider_response(self, provider, query: str, expected_content: str):
        """Assert provider generates expected response."""
        response = await provider.generate_response(query)
        assert response["content"] == expected_content


class BaseAPITest:
    """Base test class for API endpoint testing."""
    
    def setup_method(self):
        """Set up test method with FastAPI test client."""
        from fastapi.testclient import TestClient
        # Will be imported in actual test implementations
        self.test_client = None
        
    def teardown_method(self):
        """Clean up after test method."""
        pass
        
    def assert_status_code(self, response, expected: int):
        """Assert HTTP response status code."""
        assert response.status_code == expected
        
    def assert_json_response(self, response, expected_keys: List[str]):
        """Assert response contains expected JSON keys."""
        json_data = response.json()
        for key in expected_keys:
            assert key in json_data


class BaseCLITest:
    """Base test class for CLI command testing."""
    
    def setup_method(self):
        """Set up test method with CLI runner."""
        from click.testing import CliRunner
        self.runner = CliRunner()
        
    def teardown_method(self):
        """Clean up after test method."""
        pass
        
    def assert_cli_success(self, result, expected_output: Optional[str] = None):
        """Assert CLI command executed successfully."""
        assert result.exit_code == 0
        if expected_output:
            assert expected_output in result.output
            
    def assert_cli_error(self, result, expected_exit_code: int = 1):
        """Assert CLI command failed with expected exit code."""
        assert result.exit_code == expected_exit_code


class BaseIntegrationTest:
    """Base test class for integration tests."""
    
    def setup_method(self):
        """Set up integration test with multiple components."""
        self.temp_dir = None
        self.mock_services = {}
        
    def teardown_method(self):
        """Clean up integration test resources."""
        pass
        
    async def setup_mock_services(self):
        """Set up mock services for integration testing."""
        from tests.conftest import (
            mock_rag_engine,
            mock_document_processor,
            mock_query_service,
            mock_embedding_service
        )
        
        # This would typically be called by pytest fixtures
        # Implementation depends on specific test needs
        pass


# =============================================================================
# TEST UTILITIES
# =============================================================================

class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def create_test_document(content: str, source: str = "test.md") -> Dict[str, Any]:
        """Create a test document with metadata."""
        return {
            "content": content,
            "metadata": {
                "source": source,
                "type": "markdown",
                "chunk_count": len(content.split("\n\n"))
            }
        }
        
    @staticmethod
    def create_test_embedding(dimension: int = 384) -> List[float]:
        """Create a test embedding vector."""
        return [0.1] * dimension
        
    @staticmethod
    def create_test_chunk(content: str, source: str = "test.md", chunk_id: str = "chunk_1") -> Dict[str, Any]:
        """Create a test document chunk."""
        return {
            "content": content,
            "metadata": {
                "source": source,
                "chunk_id": chunk_id,
                "section": "Test Section"
            },
            "embedding": TestUtils.create_test_embedding()
        }
        
    @staticmethod
    async def wait_for_async(coro, timeout: float = 5.0):
        """Wait for async operation with timeout."""
        return await asyncio.wait_for(coro, timeout=timeout)


# =============================================================================
# MOCK FACTORIES
# =============================================================================

class MockFactory:
    """Factory for creating consistent mocks across tests."""
    
    @staticmethod
    def create_engine_service_mock(service_name: str) -> Mock:
        """Create a mock for any engine service."""
        mock_service = Mock()
        mock_service.name = service_name
        
        # Add common async methods
        for method_name in ['process', 'query', 'generate', 'search']:
            setattr(mock_service, method_name, AsyncMock())
            
        return mock_service
        
    @staticmethod
    def create_provider_mock(provider_name: str, available: bool = True) -> Mock:
        """Create a mock for any AI provider."""
        mock_provider = Mock()
        mock_provider.name = provider_name
        mock_provider.is_available = Mock(return_value=available)
        mock_provider.generate_response = AsyncMock()
        mock_provider.generate_embeddings = AsyncMock()
        
        # Set up default return values
        mock_provider.generate_response.return_value = {
            "content": f"Test response from {provider_name}",
            "model": f"{provider_name}-model",
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }
        
        mock_provider.generate_embeddings.return_value = [[0.1] * 384]
        
        return mock_provider


# =============================================================================
# ASYNC TEST HELPERS
# =============================================================================

def async_test(coro):
    """Decorator to run async tests."""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper


# =============================================================================
# TEST DATA BUILDERS
# =============================================================================

class TestDataBuilder:
    """Builder for creating test data with realistic structure."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset builder to initial state."""
        self._documents = []
        self._chunks = []
        self._embeddings = []
        return self
        
    def with_documents(self, count: int = 3) -> 'TestDataBuilder':
        """Add test documents."""
        for i in range(count):
            doc = TestUtils.create_test_document(
                content=f"Test document {i+1} content",
                source=f"test_doc_{i+1}.md"
            )
            self._documents.append(doc)
        return self
        
    def with_chunks(self, count: int = 5) -> 'TestDataBuilder':
        """Add test chunks."""
        for i in range(count):
            chunk = TestUtils.create_test_chunk(
                content=f"Test chunk {i+1} content",
                chunk_id=f"chunk_{i+1}"
            )
            self._chunks.append(chunk)
        return self
        
    def with_embeddings(self, count: int = 5, dimension: int = 384) -> 'TestDataBuilder':
        """Add test embeddings."""
        for i in range(count):
            embedding = [0.1 + i * 0.1] * dimension
            self._embeddings.append(embedding)
        return self
        
    def build(self) -> Dict[str, Any]:
        """Build the test data structure."""
        return {
            "documents": self._documents,
            "chunks": self._chunks,
            "embeddings": self._embeddings
        }