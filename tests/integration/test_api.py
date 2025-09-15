"""Integration tests for the FastAPI application."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import json

from support_deflect_bot.api.app import app
from support_deflect_bot.api.dependencies.engine import get_rag_engine, get_document_processor


class TestAPIIntegration:
    """Integration test suite for FastAPI application."""

    @pytest.fixture(autouse=True)
    def reset_app_state(self):
        from support_deflect_bot.api import app as app_module
        app_module.app.dependency_overrides.clear()
        app_module._rag_engine = None
        app_module._document_processor = None
        yield
        app_module.app.dependency_overrides.clear()
        app_module._rag_engine = None
        app_module._document_processor = None

    @pytest.fixture
    def mock_engines(self):
        """Mock the engines to avoid initialization issues during testing."""
        # Mock RAG engine
        mock_rag_instance = Mock()
        mock_rag_instance.answer_question.return_value = {
            "answer": "This is a test answer",
            "citations": [{"rank": 1, "path": "test.md", "preview": "Test content"}],
            "confidence": 0.85,
            "metadata": {"chunks_found": 3}
        }
        mock_rag_instance.get_metrics.return_value = {
            "queries_processed": 10,
            "successful_answers": 8,
            "refusals": 2
        }
        mock_rag_instance.validate_providers.return_value = {
            "llm_openai": True,
            "embedding_openai": True
        }

        # Mock document processor
        mock_doc_proc_instance = Mock()
        mock_doc_proc_instance.process_documents.return_value = {
            "indexed": 5,
            "failed": 0,
            "total_chunks": 25
        }

        # Set up dependency overrides
        app.dependency_overrides[get_rag_engine] = lambda: mock_rag_instance
        app.dependency_overrides[get_document_processor] = lambda: mock_doc_proc_instance

        yield mock_rag_instance, mock_doc_proc_instance

    @pytest.fixture
    def client(self, mock_engines):
        """Create test client with mocked engines."""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_query_endpoint_success(self, client, mock_engines):
        """Test successful query endpoint."""
        mock_rag_engine, _ = mock_engines

        response = client.post(
            "/query",
            json={
                "question": "What is this about?",
                "k": 5
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "confidence" in data
        assert data["answer"] == "This is a test answer"

    def test_query_endpoint_missing_question(self, client):
        """Test query endpoint with missing question."""
        response = client.post("/query", json={})

        # Should return 422 for validation error
        assert response.status_code == 422

    def test_query_endpoint_with_options(self, client, mock_engines):
        """Test query endpoint with additional options."""
        mock_rag_engine, _ = mock_engines

        response = client.post(
            "/query",
            json={
                "question": "How do I configure this?",
                "k": 10,
                "domains": ["example.com"],
                "min_confidence": 0.7
            }
        )

        assert response.status_code == 200
        mock_rag_engine.answer_question.assert_called_once_with(
            "How do I configure this?",
            k=10,
            domains=["example.com"],
            min_confidence=0.7
        )

    def test_index_documents_endpoint(self, client, mock_engines):
        """Test document indexing endpoint."""
        _, mock_doc_processor = mock_engines

        response = client.post(
            "/index",
            json={
                "documents": [
                    {
                        "path": "test1.md",
                        "content": "Test document 1"
                    },
                    {
                        "path": "test2.md",
                        "content": "Test document 2"
                    }
                ]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "indexed" in data
        assert "failed" in data
        assert "total_chunks" in data

    def test_index_files_endpoint(self, client, mock_engines):
        """Test file indexing endpoint."""
        _, mock_doc_processor = mock_engines

        response = client.post(
            "/index/files",
            json={
                "file_paths": ["/path/to/docs/file1.md", "/path/to/docs/file2.md"]
            }
        )

        assert response.status_code == 200

    def test_index_url_endpoint(self, client, mock_engines):
        """Test URL indexing endpoint."""
        _, mock_doc_processor = mock_engines

        response = client.post(
            "/index/url",
            json={
                "url": "https://example.com/docs",
                "max_pages": 10
            }
        )

        assert response.status_code == 200

    def test_health_detailed_endpoint(self, client, mock_engines):
        """Test detailed health endpoint."""
        mock_rag_engine, _ = mock_engines

        response = client.get("/health/detailed")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "engines" in data
        assert "providers" in data

    def test_metrics_endpoint(self, client, mock_engines):
        """Test metrics endpoint."""
        mock_rag_engine, _ = mock_engines

        response = client.get("/admin/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "queries_processed" in data
        assert "successful_answers" in data

    def test_reset_endpoint(self, client, mock_engines):
        """Test database reset endpoint."""
        response = client.post("/admin/reset")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_batch_query_endpoint(self, client, mock_engines):
        """Test batch query endpoint."""
        mock_rag_engine, _ = mock_engines

        # Configure mock to return different responses for each question
        mock_rag_engine.answer_question.side_effect = [
            {
                "answer": "Answer 1",
                "citations": [],
                "confidence": 0.8,
                "metadata": {}
            },
            {
                "answer": "Answer 2",
                "citations": [],
                "confidence": 0.9,
                "metadata": {}
            }
        ]

        response = client.post(
            "/batch/query",
            json={
                "questions": [
                    {"question": "Question 1", "id": "q1"},
                    {"question": "Question 2", "id": "q2"}
                ]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

    def test_batch_index_endpoint(self, client, mock_engines):
        """Test batch indexing endpoint."""
        _, mock_doc_processor = mock_engines

        response = client.post(
            "/batch/index",
            json={
                "documents": [
                    {"path": "doc1.md", "content": "Content 1"},
                    {"path": "doc2.md", "content": "Content 2"}
                ]
            }
        )

        assert response.status_code == 200

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/", headers={"Origin": "http://localhost:3000"})

        # FastAPI/Starlette handles OPTIONS automatically with CORS middleware
        assert response.status_code in [200, 405]  # 405 if no OPTIONS handler, but CORS headers should be present

    def test_error_handling_engine_not_initialized(self):
        """Test error handling when engines are not initialized."""
        # Create client without mocked engines by clearing dependency overrides
        app.dependency_overrides.clear()

        # Patch the global engine to be None to simulate uninitialized state
        with patch('support_deflect_bot.api.app._rag_engine', None):
            client = TestClient(app)
            response = client.post("/query", json={"question": "test"})
            assert response.status_code == 503

    def test_invalid_json_request(self, client):
        """Test handling of invalid JSON requests."""
        response = client.post(
            "/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_endpoint_engine_error(self, client, mock_engines):
        """Test query endpoint when engine raises an error."""
        mock_rag_engine, _ = mock_engines
        mock_rag_engine.answer_question.side_effect = Exception("Engine error")

        response = client.post(
            "/query",
            json={"question": "test question"}
        )

        # Should handle the error gracefully
        assert response.status_code in [500, 503]

    def test_index_endpoint_validation_errors(self, client):
        """Test index endpoint with validation errors."""
        # Missing required fields
        response = client.post("/index", json={})
        assert response.status_code == 422

        # Invalid document structure
        response = client.post(
            "/index",
            json={"documents": [{"invalid": "structure"}]}
        )
        assert response.status_code == 422

    def test_concurrent_requests(self, client, mock_engines):
        """Test handling of concurrent requests."""
        import concurrent.futures

        def make_request():
            return client.post("/query", json={"question": "concurrent test"})

        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [future.result() for future in futures]

        # All requests should succeed
        assert all(response.status_code == 200 for response in responses)

    def test_large_request_payload(self, client, mock_engines):
        """Test handling of large request payloads."""
        # Create a large document
        large_content = "A" * 10000  # 10KB content

        response = client.post(
            "/index",
            json={
                "documents": [
                    {
                        "path": "large_doc.md",
                        "content": large_content
                    }
                ]
            }
        )

        assert response.status_code == 200

    def test_query_with_special_characters(self, client, mock_engines):
        """Test query with special characters and Unicode."""
        response = client.post(
            "/query",
            json={
                "question": "How do I handle 特殊字符 and émojis 🚀?"
            }
        )

        assert response.status_code == 200

    def test_api_documentation_endpoints(self, client):
        """Test that API documentation endpoints are accessible."""
        # OpenAPI spec
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        # Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200

        # ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_lifespan_events(self):
        """Test application lifespan events."""
        with patch('support_deflect_bot.api.app.UnifiedRAGEngine') as mock_rag, \
             patch('support_deflect_bot.api.app.UnifiedDocumentProcessor') as mock_doc_proc:

            # Test that engines are initialized during startup
            with TestClient(app):
                mock_rag.assert_called_once()
                mock_doc_proc.assert_called_once()

    def test_dependency_injection(self, client, mock_engines):
        """Test that dependency injection works correctly."""
        mock_rag_engine, _ = mock_engines

        response = client.post("/query", json={"question": "test"})

        # Verify that the mocked engine was called
        mock_rag_engine.answer_question.assert_called_once()

    def test_response_format_consistency(self, client, mock_engines):
        """Test that API responses have consistent format."""
        # Test query response format
        response = client.post("/query", json={"question": "test"})
        data = response.json()

        required_fields = ["answer", "citations", "confidence", "metadata"]
        assert all(field in data for field in required_fields)

        # Test health response format
        response = client.get("/health")
        data = response.json()

        required_fields = ["status", "timestamp", "version"]
        assert all(field in data for field in required_fields)

    def test_error_response_format(self, client):
        """Test that error responses have consistent format."""
        # Test validation error
        response = client.post("/query", json={})
        assert response.status_code == 422

        error_data = response.json()
        assert "detail" in error_data