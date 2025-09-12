"""
Comprehensive unit tests for all API endpoints.

Tests all endpoint functionality including query, health, indexing, 
admin, and batch processing endpoints with proper mocking and
error handling scenarios.
"""

import pytest
import time
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status
from tests.base import BaseAPITest

from support_deflect_bot.api.app import app


class TestQueryEndpoints(BaseAPITest):
    """Test query-related API endpoints."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API testing."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_ask_request(self):
        """Sample ask request data."""
        return {
            "question": "What is the purpose of this system?",
            "domains": ["docs"],
            "max_chunks": 5,
            "min_confidence": 0.25,
            "use_context": True
        }
    
    @pytest.fixture
    def sample_search_request(self):
        """Sample search request data."""
        return {
            "query": "system purpose",
            "k": 5,
            "domains": ["docs"]
        }
    
    @pytest.fixture
    def sample_batch_ask_request(self):
        """Sample batch ask request data."""
        return {
            "questions": [
                "What is the system purpose?",
                "How does it work?",
                "What are the main features?"
            ],
            "domains": ["docs"],
            "max_chunks": 5,
            "min_confidence": 0.25
        }
    
    @pytest.fixture
    def mock_rag_result(self):
        """Mock RAG engine result."""
        return {
            "answer": "This system is designed to provide intelligent question answering.",
            "confidence": 0.85,
            "citations": [
                {
                    "id": "doc1",
                    "text": "System overview content",
                    "metadata": {"source": "docs/overview.md"},
                    "distance": 0.15
                }
            ],
            "provider_used": "openai",
            "metadata": {"tokens_used": 150}
        }
    
    @pytest.fixture
    def mock_search_result(self):
        """Mock search service result."""
        return [
            {
                "id": "chunk1",
                "text": "Relevant content for search query",
                "metadata": {"source": "docs/intro.md"},
                "distance": 0.2
            },
            {
                "id": "chunk2", 
                "text": "Additional relevant content",
                "metadata": {"source": "docs/features.md"},
                "distance": 0.25
            }
        ]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_ask_endpoint_success(self, test_client, sample_ask_request, mock_rag_result):
        """Test successful ask endpoint request."""
        with patch('support_deflect_bot.api.dependencies.engine.get_rag_engine') as mock_get_rag:
            mock_engine = Mock()
            mock_engine.answer_question.return_value = mock_rag_result
            mock_get_rag.return_value = mock_engine
            
            response = test_client.post("/api/v1/ask", json=sample_ask_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "answer" in data
            assert "confidence" in data
            assert "sources" in data
            assert "chunks_used" in data
            assert "response_time" in data
            assert "provider_used" in data
            assert "metadata" in data
            
            # Check response content
            assert data["answer"] == mock_rag_result["answer"]
            assert data["confidence"] == mock_rag_result["confidence"]
            assert data["provider_used"] == mock_rag_result["provider_used"]
            assert len(data["sources"]) == len(mock_rag_result["citations"])
            
            # Verify engine was called correctly
            mock_engine.answer_question.assert_called_once_with(
                question=sample_ask_request["question"],
                domains=sample_ask_request["domains"],
                k=sample_ask_request["max_chunks"],
                min_confidence=sample_ask_request["min_confidence"]
            )
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_ask_endpoint_validation_error(self, test_client):
        """Test ask endpoint with invalid request data."""
        invalid_request = {
            "question": "",  # Empty question should fail validation
            "max_chunks": 25,  # Exceeds maximum limit
            "min_confidence": 1.5  # Exceeds maximum value
        }
        
        response = test_client.post("/api/v1/ask", json=invalid_request)
        assert response.status_code == 422  # Validation error
        
        # Test missing required field
        response = test_client.post("/api/v1/ask", json={})
        assert response.status_code == 422
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_ask_endpoint_engine_error(self, test_client, sample_ask_request):
        """Test ask endpoint when engine raises exception."""
        with patch('support_deflect_bot.api.dependencies.engine.get_rag_engine') as mock_get_rag:
            mock_engine = Mock()
            mock_engine.answer_question.side_effect = Exception("Engine processing failed")
            mock_get_rag.return_value = mock_engine
            
            response = test_client.post("/api/v1/ask", json=sample_ask_request)
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Question processing failed" in data["detail"]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_search_endpoint_success(self, test_client, sample_search_request, mock_search_result):
        """Test successful search endpoint request."""
        with patch('support_deflect_bot.api.dependencies.engine.get_query_service') as mock_get_query:
            mock_query_service = Mock()
            mock_query_service.search_similar_chunks.return_value = mock_search_result
            mock_get_query.return_value = mock_query_service
            
            response = test_client.post("/api/v1/search", json=sample_search_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "results" in data
            assert "total_count" in data
            assert "query" in data
            assert "response_time" in data
            
            # Check response content
            assert len(data["results"]) == len(mock_search_result)
            assert data["total_count"] == len(mock_search_result)
            assert data["query"] == sample_search_request["query"]
            
            # Check result structure
            for result in data["results"]:
                assert "id" in result
                assert "content" in result
                assert "metadata" in result
                assert "distance" in result
                assert "score" in result
            
            # Verify query service was called correctly
            mock_query_service.search_similar_chunks.assert_called_once_with(
                query=sample_search_request["query"],
                k=sample_search_request["k"],
                domain_filter=sample_search_request["domains"]
            )
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_search_endpoint_validation_error(self, test_client):
        """Test search endpoint with invalid request data."""
        invalid_request = {
            "query": "",  # Empty query should fail validation
            "k": 0  # Below minimum limit
        }
        
        response = test_client.post("/api/v1/search", json=invalid_request)
        assert response.status_code == 422
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_batch_ask_endpoint_success(self, test_client, sample_batch_ask_request, mock_rag_result):
        """Test successful batch ask endpoint request."""
        with patch('support_deflect_bot.api.dependencies.engine.get_rag_engine') as mock_get_rag:
            mock_engine = Mock()
            mock_engine.answer_question.return_value = mock_rag_result
            mock_get_rag.return_value = mock_engine
            
            response = test_client.post("/api/v1/batch_ask", json=sample_batch_ask_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "results" in data
            assert "total_questions" in data
            assert "successful_answers" in data
            assert "processing_time" in data
            
            # Check response content
            assert len(data["results"]) == len(sample_batch_ask_request["questions"])
            assert data["total_questions"] == len(sample_batch_ask_request["questions"])
            assert data["successful_answers"] == len(sample_batch_ask_request["questions"])
            
            # Verify engine was called for each question
            assert mock_engine.answer_question.call_count == len(sample_batch_ask_request["questions"])
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_batch_ask_endpoint_partial_failure(self, test_client, sample_batch_ask_request, mock_rag_result):
        """Test batch ask endpoint with partial failures."""
        with patch('support_deflect_bot.api.dependencies.engine.get_rag_engine') as mock_get_rag:
            mock_engine = Mock()
            
            # First call succeeds, second fails, third succeeds
            mock_engine.answer_question.side_effect = [
                mock_rag_result,
                Exception("Processing failed for question 2"),
                mock_rag_result
            ]
            mock_get_rag.return_value = mock_engine
            
            response = test_client.post("/api/v1/batch_ask", json=sample_batch_ask_request)
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data["results"]) == 3
            assert data["total_questions"] == 3
            assert data["successful_answers"] == 2  # Two successful, one failed
            
            # Check that error response is included for failed question
            error_result = data["results"][1]  # Second question failed
            assert "Error processing question" in error_result["answer"]
            assert error_result["confidence"] == 0.0
            assert error_result["provider_used"] == "error"


class TestHealthEndpoints(BaseAPITest):
    """Test health-related API endpoints."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API testing."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_healthy_services(self):
        """Mock healthy service states."""
        return {
            "rag_status": {"overall_health": "ok", "providers": ["openai"]},
            "doc_status": {"connected": True, "total_chunks": 1500, "collections": ["docs"]},
            "query_status": {"connected": True, "queries_processed": 250},
            "embedding_status": {"openai": True, "local": True}
        }
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_health_endpoint_success(self, test_client, mock_healthy_services):
        """Test successful health check endpoint."""
        with patch('support_deflect_bot.api.endpoints.health.get_rag_engine') as mock_rag, \
             patch('support_deflect_bot.api.endpoints.health.get_document_processor') as mock_doc, \
             patch('support_deflect_bot.api.endpoints.health.get_query_service') as mock_query, \
             patch('support_deflect_bot.api.endpoints.health.get_embedding_service') as mock_embed:
            
            # Mock service instances
            mock_rag_engine = Mock()
            mock_rag_engine.get_system_status.return_value = mock_healthy_services["rag_status"]
            mock_rag.return_value = mock_rag_engine
            
            mock_doc_processor = Mock()
            mock_doc_processor.get_status.return_value = mock_healthy_services["doc_status"]
            mock_doc.return_value = mock_doc_processor
            
            mock_query_service = Mock()
            mock_query_service.get_status.return_value = mock_healthy_services["query_status"]
            mock_query.return_value = mock_query_service
            
            mock_embed_service = Mock()
            mock_embed_service.get_provider_status.return_value = mock_healthy_services["embedding_status"]
            mock_embed.return_value = mock_embed_service
            
            response = test_client.get("/api/v1/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "status" in data
            assert "timestamp" in data
            assert "version" in data
            assert "providers" in data
            assert "database" in data
            
            # Check healthy status
            assert data["status"] == "healthy"
            assert data["database"]["connected"] is True
            
    @pytest.mark.unit
    @pytest.mark.api
    def test_health_endpoint_degraded_status(self, test_client):
        """Test health endpoint with degraded system status."""
        with patch('support_deflect_bot.api.endpoints.health.get_rag_engine') as mock_rag, \
             patch('support_deflect_bot.api.endpoints.health.get_document_processor') as mock_doc, \
             patch('support_deflect_bot.api.endpoints.health.get_query_service') as mock_query, \
             patch('support_deflect_bot.api.endpoints.health.get_embedding_service') as mock_embed:
            
            # Mock degraded service states
            mock_rag_engine = Mock()
            mock_rag_engine.get_system_status.return_value = {"overall_health": "ok"}
            mock_rag.return_value = mock_rag_engine
            
            mock_doc_processor = Mock()
            mock_doc_processor.get_status.return_value = {"connected": False}  # Database disconnected
            mock_doc.return_value = mock_doc_processor
            
            mock_query_service = Mock()
            mock_query_service.get_status.return_value = {"connected": False}
            mock_query.return_value = mock_query_service
            
            mock_embed_service = Mock()
            mock_embed_service.get_provider_status.return_value = {}
            mock_embed.return_value = mock_embed_service
            
            response = test_client.get("/api/v1/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"  # Should be degraded due to database issue
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_health_endpoint_error_handling(self, test_client):
        """Test health endpoint when services raise exceptions."""
        with patch('support_deflect_bot.api.endpoints.health.get_rag_engine') as mock_rag:
            mock_rag.side_effect = Exception("Service unavailable")
            
            response = test_client.get("/api/v1/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "error" in data["providers"]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_ping_endpoint(self, test_client):
        """Test simple ping endpoint."""
        response = test_client.get("/api/v1/ping")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "service" in data
        assert data["status"] == "ok"
        assert data["service"] == "Support Deflect Bot API"
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_readiness_endpoint_ready(self, test_client):
        """Test readiness endpoint when system is ready."""
        with patch('support_deflect_bot.api.endpoints.health.get_rag_engine') as mock_rag, \
             patch('support_deflect_bot.api.endpoints.health.get_document_processor') as mock_doc:
            
            mock_rag_engine = Mock()
            mock_rag.return_value = mock_rag_engine
            
            mock_doc_processor = Mock()
            mock_doc_processor.get_status.return_value = {"connected": True}
            mock_doc.return_value = mock_doc_processor
            
            response = test_client.get("/api/v1/readiness")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "ready" in data
            assert "timestamp" in data
            assert "checks" in data
            assert data["ready"] is True
            assert data["checks"]["database"] is True
            assert data["checks"]["engine"] is True
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_readiness_endpoint_not_ready(self, test_client):
        """Test readiness endpoint when system is not ready."""
        with patch('support_deflect_bot.api.endpoints.health.get_rag_engine') as mock_rag, \
             patch('support_deflect_bot.api.endpoints.health.get_document_processor') as mock_doc:
            
            mock_rag_engine = Mock()
            mock_rag.return_value = mock_rag_engine
            
            mock_doc_processor = Mock()
            mock_doc_processor.get_status.return_value = {"connected": False}
            mock_doc.return_value = mock_doc_processor
            
            response = test_client.get("/api/v1/readiness")
            
            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is False
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_readiness_endpoint_service_error(self, test_client):
        """Test readiness endpoint when services are unavailable."""
        with patch('support_deflect_bot.api.endpoints.health.get_rag_engine') as mock_rag:
            mock_rag.side_effect = Exception("Service unavailable")
            
            response = test_client.get("/api/v1/readiness")
            
            assert response.status_code == 503
            data = response.json()
            assert "detail" in data
            assert "Service not ready" in data["detail"]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_liveness_endpoint(self, test_client):
        """Test liveness endpoint."""
        response = test_client.get("/api/v1/liveness")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "alive" in data
        assert "timestamp" in data
        assert data["alive"] is True


class TestIndexingEndpoints(BaseAPITest):
    """Test indexing-related API endpoints."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API testing."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_index_request(self):
        """Sample index request data."""
        return {
            "directory": "./test_docs",
            "force": True,
            "recursive": True,
            "file_patterns": ["*.md", "*.txt"]
        }
    
    @pytest.fixture
    def sample_crawl_request(self):
        """Sample crawl request data."""
        return {
            "urls": ["https://example.com/docs", "https://example.com/help"],
            "depth": 2,
            "max_pages": 20,
            "same_domain": True
        }
    
    @pytest.fixture
    def mock_index_result(self):
        """Mock indexing result."""
        return {
            "processed_files": ["./test_docs/file1.md", "./test_docs/file2.txt"],
            "errors": []
        }
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_index_endpoint_success(self, test_client, sample_index_request, mock_index_result):
        """Test successful index endpoint request."""
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True), \
             patch('support_deflect_bot.api.dependencies.engine.get_document_processor') as mock_get_doc:
            
            mock_processor = Mock()
            mock_processor.process_directory.return_value = mock_index_result
            mock_get_doc.return_value = mock_processor
            
            response = test_client.post("/api/v1/index", json=sample_index_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "success" in data
            assert "processed_count" in data
            assert "failed_count" in data
            assert "processing_time" in data
            assert "directory" in data
            assert "details" in data
            assert "error_messages" in data
            
            # Check response content
            assert data["success"] is True
            assert data["processed_count"] == 2
            assert data["failed_count"] == 0
            assert data["directory"] == sample_index_request["directory"]
            
            # Verify processor was called correctly
            mock_processor.process_directory.assert_called_once_with(
                directory_path=sample_index_request["directory"],
                recursive=sample_index_request["recursive"],
                force_reprocess=sample_index_request["force"],
                file_patterns=sample_index_request["file_patterns"]
            )
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_index_endpoint_directory_not_found(self, test_client, sample_index_request):
        """Test index endpoint with non-existent directory."""
        with patch('os.path.exists', return_value=False):
            response = test_client.post("/api/v1/index", json=sample_index_request)
            
            assert response.status_code == 400
            data = response.json()
            assert "Directory not found" in data["detail"]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_index_endpoint_not_directory(self, test_client, sample_index_request):
        """Test index endpoint with path that is not a directory."""
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=False):
            
            response = test_client.post("/api/v1/index", json=sample_index_request)
            
            assert response.status_code == 400
            data = response.json()
            assert "Path is not a directory" in data["detail"]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_index_endpoint_with_errors(self, test_client, sample_index_request):
        """Test index endpoint with processing errors."""
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True), \
             patch('support_deflect_bot.api.dependencies.engine.get_document_processor') as mock_get_doc:
            
            mock_result = {
                "processed_files": ["./test_docs/file1.md"],
                "errors": [{"path": "./test_docs/bad_file.md", "error": "Parsing failed"}]
            }
            
            mock_processor = Mock()
            mock_processor.process_directory.return_value = mock_result
            mock_get_doc.return_value = mock_processor
            
            response = test_client.post("/api/v1/index", json=sample_index_request)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["processed_count"] == 1
            assert data["failed_count"] == 1
            assert len(data["error_messages"]) > 0
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_crawl_endpoint_success(self, test_client, sample_crawl_request):
        """Test successful crawl endpoint request."""
        with patch('support_deflect_bot.api.dependencies.validation.validate_crawl_urls') as mock_validate, \
             patch('support_deflect_bot.api.dependencies.engine.get_document_processor') as mock_get_doc:
            
            mock_validate.return_value = sample_crawl_request["urls"]
            
            mock_processor = Mock()
            mock_processor.process_web_content.return_value = {"pages_processed": 5}
            mock_get_doc.return_value = mock_processor
            
            response = test_client.post("/api/v1/crawl", json=sample_crawl_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "success" in data
            assert "processed_count" in data
            assert "failed_count" in data
            assert "processing_time" in data
            assert "urls" in data
            assert "crawl_details" in data
            
            # Check response content
            assert data["success"] is True
            assert data["processed_count"] == 2  # Two URLs processed successfully
            assert data["failed_count"] == 0
            assert data["urls"] == sample_crawl_request["urls"]
            
            # Verify crawl details
            assert data["crawl_details"]["depth"] == sample_crawl_request["depth"]
            assert data["crawl_details"]["max_pages"] == sample_crawl_request["max_pages"]
            assert data["crawl_details"]["same_domain"] == sample_crawl_request["same_domain"]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_crawl_endpoint_validation_error(self, test_client):
        """Test crawl endpoint with invalid URLs."""
        invalid_request = {
            "urls": ["not-a-valid-url", "ftp://invalid-protocol.com"],
            "depth": 1,
            "max_pages": 10
        }
        
        response = test_client.post("/api/v1/crawl", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_clear_index_endpoint_success(self, test_client):
        """Test successful clear index endpoint request."""
        with patch('support_deflect_bot.api.dependencies.engine.get_document_processor') as mock_get_doc:
            mock_processor = Mock()
            mock_processor.clear_database.return_value = True
            mock_get_doc.return_value = mock_processor
            
            response = test_client.delete("/api/v1/index")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "success" in data
            assert "message" in data
            assert "timestamp" in data
            assert data["success"] is True
            assert "cleared successfully" in data["message"]
            
            mock_processor.clear_database.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_clear_index_endpoint_error(self, test_client):
        """Test clear index endpoint when operation fails."""
        with patch('support_deflect_bot.api.dependencies.engine.get_document_processor') as mock_get_doc:
            mock_processor = Mock()
            mock_processor.clear_database.side_effect = Exception("Database error")
            mock_get_doc.return_value = mock_processor
            
            response = test_client.delete("/api/v1/index")
            
            assert response.status_code == 500
            data = response.json()
            assert "Failed to clear index" in data["detail"]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_index_stats_endpoint_success(self, test_client):
        """Test successful index stats endpoint request."""
        with patch('support_deflect_bot.api.dependencies.engine.get_document_processor') as mock_get_doc:
            mock_status = {
                "connected": True,
                "total_chunks": 1250,
                "collections": ["docs", "help"],
                "database_path": "/data/vector_db",
                "last_updated": "2024-01-15T10:30:00Z"
            }
            
            mock_processor = Mock()
            mock_processor.get_status.return_value = mock_status
            mock_get_doc.return_value = mock_processor
            
            response = test_client.get("/api/v1/index/stats")
            
            assert response.status_code == 200
            data = response.json()
            
            # Check all expected fields are present
            assert "connected" in data
            assert "total_chunks" in data
            assert "collections" in data
            assert "database_path" in data
            assert "last_updated" in data
            
            # Check values match mock
            assert data["connected"] == mock_status["connected"]
            assert data["total_chunks"] == mock_status["total_chunks"]
            assert data["collections"] == mock_status["collections"]


class TestAdminEndpoints(BaseAPITest):
    """Test admin API endpoints."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API testing."""
        return TestClient(app)
    
    @pytest.fixture
    def valid_api_key(self):
        """Mock valid API key for testing."""
        return "test-api-key-123"
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_admin_metrics_endpoint_success(self, test_client, valid_api_key):
        """Test successful admin metrics endpoint request."""
        with patch('support_deflect_bot.api.dependencies.security.verify_api_key') as mock_verify, \
             patch('support_deflect_bot.api.dependencies.engine.get_rag_engine') as mock_rag, \
             patch('support_deflect_bot.api.dependencies.engine.get_query_service') as mock_query, \
             patch('support_deflect_bot.api.dependencies.engine.get_document_processor') as mock_doc, \
             patch('support_deflect_bot.api.dependencies.engine.get_embedding_service') as mock_embed:
            
            # Mock API key verification
            mock_verify.return_value = valid_api_key
            
            # Mock service instances and their metrics
            mock_rag_engine = Mock()
            mock_rag_engine.get_metrics.return_value = {"queries_processed": 150, "avg_response_time": 0.85}
            mock_rag.return_value = mock_rag_engine
            
            mock_query_service = Mock()
            mock_query_service.get_query_analytics.return_value = {"total_searches": 75, "cache_hits": 45}
            mock_query.return_value = mock_query_service
            
            mock_doc_processor = Mock()
            mock_doc_processor.get_status.return_value = {"connected": True, "total_chunks": 1000}
            mock_doc.return_value = mock_doc_processor
            
            mock_embed_service = Mock()
            mock_embed_service.get_provider_status.return_value = {"openai": True, "local": False}
            mock_embed.return_value = mock_embed_service
            
            # Add API key to headers
            response = test_client.get("/api/v1/admin/metrics", headers={"X-API-Key": valid_api_key})
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "timestamp" in data
            assert "version" in data
            assert "application" in data
            assert "rag_engine" in data
            assert "query_service" in data
            assert "document_processor" in data
            assert "embedding_service" in data
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_admin_metrics_endpoint_unauthorized(self, test_client):
        """Test admin metrics endpoint without API key."""
        response = test_client.get("/api/v1/admin/metrics")
        
        # Should require authentication
        assert response.status_code in [401, 403]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_admin_reset_endpoint_success(self, test_client, valid_api_key):
        """Test successful admin reset endpoint request."""
        with patch('support_deflect_bot.api.dependencies.security.verify_api_key') as mock_verify:
            mock_verify.return_value = valid_api_key
            
            response = test_client.post("/api/v1/admin/reset", headers={"X-API-Key": valid_api_key})
            
            assert response.status_code == 200
            data = response.json()
            
            assert "success" in data
            assert "message" in data
            assert "timestamp" in data
            assert data["success"] is True
            assert "reset completed" in data["message"]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_admin_status_endpoint_success(self, test_client):
        """Test successful admin status endpoint request."""
        with patch('support_deflect_bot.api.dependencies.engine.get_rag_engine') as mock_rag, \
             patch('support_deflect_bot.api.dependencies.engine.get_document_processor') as mock_doc:
            
            mock_rag_engine = Mock()
            mock_rag_engine.get_system_status.return_value = {"overall_health": "ok", "components": ["llm", "embeddings"]}
            mock_rag.return_value = mock_rag_engine
            
            mock_doc_processor = Mock()
            mock_doc_processor.get_status.return_value = {"connected": True, "database_type": "chroma"}
            mock_doc.return_value = mock_doc_processor
            
            response = test_client.get("/api/v1/admin/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "timestamp" in data
            assert "version" in data
            assert "system_status" in data
            assert "database_status" in data
            assert "uptime" in data
            assert "memory_usage" in data
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_admin_providers_endpoint_success(self, test_client):
        """Test successful admin providers endpoint request."""
        with patch('support_deflect_bot.api.dependencies.engine.get_embedding_service') as mock_embed:
            mock_embed_service = Mock()
            mock_provider_status = {"openai": True, "huggingface": False, "local": True}
            mock_embed_service.get_provider_status.return_value = mock_provider_status
            mock_embed.return_value = mock_embed_service
            
            response = test_client.get("/api/v1/admin/providers")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "provider_status" in data
            assert "validation_results" in data
            assert "timestamp" in data
            
            # Check provider status
            assert data["provider_status"] == mock_provider_status
            
            # Check validation results structure
            for provider in mock_provider_status.keys():
                assert provider in data["validation_results"]
                assert "available" in data["validation_results"][provider]
                assert "last_tested" in data["validation_results"][provider]


class TestBatchEndpoints(BaseAPITest):
    """Test batch processing API endpoints."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API testing.""" 
        return TestClient(app)
    
    @pytest.fixture
    def sample_batch_ask_request(self):
        """Sample batch ask request data."""
        return {
            "questions": [
                "What is machine learning?",
                "How does neural networks work?",
                "What are the types of AI?"
            ],
            "domains": ["ml", "ai"],
            "max_chunks": 5,
            "min_confidence": 0.3
        }
    
    @pytest.fixture
    def sample_batch_crawl_request(self):
        """Sample batch crawl request data."""
        return {
            "urls": ["https://docs.example.com", "https://help.example.com", "https://api.example.com"],
            "depth": 2,
            "max_pages": 15,
            "same_domain": True
        }
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_batch_ask_endpoint_success(self, test_client, sample_batch_ask_request):
        """Test successful batch ask endpoint request."""
        with patch('support_deflect_bot.api.dependencies.engine.get_rag_engine') as mock_get_rag:
            mock_result = {
                "answer": "Machine learning is a method of data analysis...",
                "confidence": 0.78,
                "citations": [{"id": "ml1", "text": "ML content", "metadata": {}, "distance": 0.22}],
                "provider_used": "openai",
                "metadata": {}
            }
            
            mock_engine = Mock()
            mock_engine.answer_question.return_value = mock_result
            mock_get_rag.return_value = mock_engine
            
            response = test_client.post("/api/v1/batch/ask", json=sample_batch_ask_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "results" in data
            assert "total_questions" in data
            assert "successful_answers" in data
            assert "processing_time" in data
            
            # Check response content
            assert len(data["results"]) == len(sample_batch_ask_request["questions"])
            assert data["total_questions"] == len(sample_batch_ask_request["questions"])
            assert data["successful_answers"] == len(sample_batch_ask_request["questions"])
            
            # Verify each result has proper structure
            for result in data["results"]:
                assert "answer" in result
                assert "confidence" in result
                assert "sources" in result
                assert "response_time" in result
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_batch_crawl_endpoint_success(self, test_client, sample_batch_crawl_request):
        """Test successful batch crawl endpoint request."""
        with patch('support_deflect_bot.api.dependencies.engine.get_document_processor') as mock_get_doc:
            mock_processor = Mock()
            mock_processor.process_web_content.return_value = {"pages": 3, "success": True}
            mock_get_doc.return_value = mock_processor
            
            response = test_client.post("/api/v1/batch/crawl", json=sample_batch_crawl_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "success" in data
            assert "processed_count" in data
            assert "failed_count" in data
            assert "processing_time" in data
            assert "urls" in data
            assert "crawl_details" in data
            
            # Check response content
            assert data["success"] is True
            assert data["processed_count"] == 3  # Three URLs processed
            assert data["failed_count"] == 0
            assert data["crawl_details"]["batch_mode"] is True
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_batch_status_endpoint(self, test_client):
        """Test batch status endpoint."""
        response = test_client.get("/api/v1/batch/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "batch_processing" in data
        assert "queue_status" in data
        assert "active_jobs" in data
        assert "completed_jobs" in data
        assert "timestamp" in data
        
        # Check default values
        assert data["batch_processing"] == "available"
        assert data["active_jobs"] == 0
        assert data["queue_status"] == "not_implemented"


class TestErrorHandlingAndEdgeCases(BaseAPITest):
    """Test error handling and edge cases across all endpoints."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API testing."""
        return TestClient(app)
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_endpoint_not_found(self, test_client):
        """Test 404 for non-existent endpoints."""
        response = test_client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_method_not_allowed(self, test_client):
        """Test 405 for incorrect HTTP methods."""
        # POST endpoint called with GET
        response = test_client.get("/api/v1/ask")
        assert response.status_code == 405
        
        # GET endpoint called with POST
        response = test_client.post("/api/v1/ping")
        assert response.status_code == 405
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_malformed_json_request(self, test_client):
        """Test handling of malformed JSON requests."""
        response = test_client.post(
            "/api/v1/ask",
            data="invalid json{",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_empty_request_body(self, test_client):
        """Test handling of empty request bodies."""
        response = test_client.post("/api/v1/ask", json={})
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_service_dependency_failures(self, test_client):
        """Test handling when service dependencies are unavailable."""
        sample_request = {"question": "test question"}
        
        # Mock dependency injection to return None
        with patch('support_deflect_bot.api.dependencies.engine.get_rag_engine') as mock_get_rag:
            mock_get_rag.side_effect = Exception("Service initialization failed")
            
            response = test_client.post("/api/v1/ask", json=sample_request)
            
            # Should handle dependency injection failure gracefully
            assert response.status_code == 500
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_timeout_handling(self, test_client):
        """Test handling of service timeouts."""
        sample_request = {"question": "test question"}
        
        with patch('support_deflect_bot.api.dependencies.engine.get_rag_engine') as mock_get_rag:
            mock_engine = Mock()
            # Simulate timeout exception
            mock_engine.answer_question.side_effect = TimeoutError("Request timed out")
            mock_get_rag.return_value = mock_engine
            
            response = test_client.post("/api/v1/ask", json=sample_request)
            
            assert response.status_code == 500
            data = response.json()
            assert "Question processing failed" in data["detail"]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_large_request_handling(self, test_client):
        """Test handling of oversized requests."""
        # Create request with very large question
        large_request = {
            "question": "x" * 2000,  # Exceeds max_length=1000
            "max_chunks": 5
        }
        
        response = test_client.post("/api/v1/ask", json=large_request)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_concurrent_request_handling(self, test_client):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = test_client.get("/api/v1/ping")
            results.append(response.status_code)
        
        # Create multiple threads to simulate concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5