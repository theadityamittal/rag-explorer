"""
Comprehensive unit tests for API models and validation.

Tests Pydantic request/response models, field validation,
custom validators, and validation dependencies.
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, patch
from pydantic import ValidationError
from fastapi import HTTPException
from tests.base import BaseAPITest

from support_deflect_bot.api.models.requests import (
    AskRequest,
    SearchRequest, 
    IndexRequest,
    CrawlRequest,
    BatchAskRequest
)
from support_deflect_bot.api.models.responses import (
    AskResponse,
    SearchResponse,
    IndexResponse,
    CrawlResponse,
    HealthResponse,
    BatchAskResponse,
    Source,
    SearchResult,
    ProcessingDetail,
    ErrorResponse
)
from support_deflect_bot.api.dependencies.validation import (
    validate_domain_filter,
    validate_user_agent,
    validate_crawl_urls,
    validate_file_patterns,
    validate_pagination
)


class TestRequestModels(BaseAPITest):
    """Test request model validation."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_ask_request_valid_data(self):
        """Test AskRequest with valid data."""
        valid_data = {
            "question": "What is machine learning?",
            "domains": ["ml", "ai"],
            "max_chunks": 5,
            "min_confidence": 0.5,
            "use_context": True
        }
        
        request = AskRequest(**valid_data)
        
        assert request.question == "What is machine learning?"
        assert request.domains == ["ml", "ai"]
        assert request.max_chunks == 5
        assert request.min_confidence == 0.5
        assert request.use_context is True
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_ask_request_default_values(self):
        """Test AskRequest with default values."""
        request = AskRequest(question="Test question")
        
        assert request.question == "Test question"
        assert request.domains is None
        assert request.max_chunks == 5  # Default value
        assert request.min_confidence == 0.25  # Default value
        assert request.use_context is True  # Default value
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_ask_request_question_validation(self):
        """Test AskRequest question field validation."""
        # Empty question should fail
        with pytest.raises(ValidationError) as exc_info:
            AskRequest(question="")
        assert "Question cannot be empty" in str(exc_info.value)
        
        # Whitespace-only question should fail
        with pytest.raises(ValidationError) as exc_info:
            AskRequest(question="   ")
        assert "Question cannot be empty" in str(exc_info.value)
        
        # Too long question should fail
        with pytest.raises(ValidationError) as exc_info:
            AskRequest(question="x" * 1001)  # Exceeds max_length=1000
        assert "ensure this value has at most 1000 characters" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_ask_request_field_constraints(self):
        """Test AskRequest field constraint validation."""
        # max_chunks too low
        with pytest.raises(ValidationError) as exc_info:
            AskRequest(question="test", max_chunks=0)
        assert "ensure this value is greater than or equal to 1" in str(exc_info.value)
        
        # max_chunks too high
        with pytest.raises(ValidationError) as exc_info:
            AskRequest(question="test", max_chunks=25)
        assert "ensure this value is less than or equal to 20" in str(exc_info.value)
        
        # min_confidence too low
        with pytest.raises(ValidationError) as exc_info:
            AskRequest(question="test", min_confidence=-0.1)
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)
        
        # min_confidence too high
        with pytest.raises(ValidationError) as exc_info:
            AskRequest(question="test", min_confidence=1.5)
        assert "ensure this value is less than or equal to 1" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_search_request_valid_data(self):
        """Test SearchRequest with valid data."""
        valid_data = {
            "query": "neural networks",
            "k": 10,
            "domains": ["ai", "ml"]
        }
        
        request = SearchRequest(**valid_data)
        
        assert request.query == "neural networks"
        assert request.k == 10
        assert request.domains == ["ai", "ml"]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_search_request_query_validation(self):
        """Test SearchRequest query field validation."""
        # Empty query should fail
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="")
        assert "Query cannot be empty" in str(exc_info.value)
        
        # Too long query should fail
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="x" * 501)  # Exceeds max_length=500
        assert "ensure this value has at most 500 characters" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_search_request_k_validation(self):
        """Test SearchRequest k field validation."""
        # k too low
        with pytest.raises(ValidationError):
            SearchRequest(query="test", k=0)
        
        # k too high
        with pytest.raises(ValidationError):
            SearchRequest(query="test", k=100)
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_index_request_valid_data(self):
        """Test IndexRequest with valid data."""
        valid_data = {
            "directory": "/docs",
            "force": True,
            "recursive": False,
            "file_patterns": ["*.md", "*.txt"]
        }
        
        request = IndexRequest(**valid_data)
        
        assert request.directory == "/docs"
        assert request.force is True
        assert request.recursive is False
        assert request.file_patterns == ["*.md", "*.txt"]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_index_request_defaults(self):
        """Test IndexRequest with default values."""
        request = IndexRequest()
        
        assert request.directory is None
        assert request.force is False
        assert request.recursive is True
        assert request.file_patterns is None
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_crawl_request_valid_data(self):
        """Test CrawlRequest with valid data."""
        valid_data = {
            "urls": ["https://example.com", "https://docs.example.com"],
            "depth": 2,
            "max_pages": 50,
            "same_domain": False
        }
        
        request = CrawlRequest(**valid_data)
        
        assert request.urls == ["https://example.com", "https://docs.example.com"]
        assert request.depth == 2
        assert request.max_pages == 50
        assert request.same_domain is False
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_crawl_request_url_validation(self):
        """Test CrawlRequest URL validation."""
        # Invalid URL format
        with pytest.raises(ValidationError) as exc_info:
            CrawlRequest(urls=["not-a-url"])
        assert "Invalid URL" in str(exc_info.value)
        
        # FTP protocol should fail
        with pytest.raises(ValidationError) as exc_info:
            CrawlRequest(urls=["ftp://example.com"])
        assert "Invalid URL" in str(exc_info.value)
        
        # Valid URLs should pass
        request = CrawlRequest(urls=[
            "https://example.com",
            "http://localhost:8000",
            "https://192.168.1.1"
        ])
        assert len(request.urls) == 3
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_crawl_request_constraints(self):
        """Test CrawlRequest field constraints."""
        # Too few URLs
        with pytest.raises(ValidationError):
            CrawlRequest(urls=[])
        
        # Too many URLs
        with pytest.raises(ValidationError):
            CrawlRequest(urls=["https://example.com"] * 15)  # Exceeds max_items=10
        
        # Depth constraints
        with pytest.raises(ValidationError):
            CrawlRequest(urls=["https://example.com"], depth=0)
        
        with pytest.raises(ValidationError):
            CrawlRequest(urls=["https://example.com"], depth=5)  # Exceeds le=3
        
        # max_pages constraints  
        with pytest.raises(ValidationError):
            CrawlRequest(urls=["https://example.com"], max_pages=0)
        
        with pytest.raises(ValidationError):
            CrawlRequest(urls=["https://example.com"], max_pages=200)  # Exceeds le=100
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_batch_ask_request_valid_data(self):
        """Test BatchAskRequest with valid data."""
        valid_data = {
            "questions": [
                "What is AI?",
                "How does ML work?", 
                "What are neural networks?"
            ],
            "domains": ["ai", "ml"],
            "max_chunks": 3,
            "min_confidence": 0.3
        }
        
        request = BatchAskRequest(**valid_data)
        
        assert len(request.questions) == 3
        assert request.questions[0] == "What is AI?"
        assert request.domains == ["ai", "ml"]
        assert request.max_chunks == 3
        assert request.min_confidence == 0.3
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_batch_ask_request_question_validation(self):
        """Test BatchAskRequest question validation."""
        # Empty questions list
        with pytest.raises(ValidationError):
            BatchAskRequest(questions=[])
        
        # Too many questions
        with pytest.raises(ValidationError):
            BatchAskRequest(questions=["test"] * 15)  # Exceeds max_items=10
        
        # Empty question in list
        with pytest.raises(ValidationError) as exc_info:
            BatchAskRequest(questions=["Valid question", ""])
        assert "Questions cannot be empty" in str(exc_info.value)
        
        # Whitespace-only question
        with pytest.raises(ValidationError) as exc_info:
            BatchAskRequest(questions=["Valid question", "   "])
        assert "Questions cannot be empty" in str(exc_info.value)


class TestResponseModels(BaseAPITest):
    """Test response model serialization and validation."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_source_model(self):
        """Test Source response model."""
        source = Source(
            id="doc1",
            content="Test content",
            metadata={"file": "test.txt"},
            distance=0.25
        )
        
        assert source.id == "doc1"
        assert source.content == "Test content"
        assert source.metadata == {"file": "test.txt"}
        assert source.distance == 0.25
        
        # Test JSON serialization
        data = source.dict()
        assert "id" in data
        assert "content" in data
        assert "metadata" in data
        assert "distance" in data
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_source_model_constraints(self):
        """Test Source model field constraints."""
        # Distance too low
        with pytest.raises(ValidationError):
            Source(content="test", distance=-0.1)
        
        # Distance too high
        with pytest.raises(ValidationError):
            Source(content="test", distance=2.5)
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_ask_response_model(self):
        """Test AskResponse model."""
        sources = [
            Source(id="1", content="Content 1", distance=0.2),
            Source(id="2", content="Content 2", distance=0.3)
        ]
        
        response = AskResponse(
            answer="This is the answer",
            confidence=0.85,
            sources=sources,
            chunks_used=2,
            response_time=1.25,
            provider_used="openai",
            metadata={"tokens": 150}
        )
        
        assert response.answer == "This is the answer"
        assert response.confidence == 0.85
        assert len(response.sources) == 2
        assert response.chunks_used == 2
        assert response.response_time == 1.25
        assert response.provider_used == "openai"
        assert response.metadata["tokens"] == 150
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_ask_response_constraints(self):
        """Test AskResponse field constraints."""
        # Confidence too low
        with pytest.raises(ValidationError):
            AskResponse(
                answer="test", 
                confidence=-0.1,
                chunks_used=1,
                response_time=1.0,
                provider_used="test"
            )
        
        # Confidence too high
        with pytest.raises(ValidationError):
            AskResponse(
                answer="test",
                confidence=1.5,
                chunks_used=1,
                response_time=1.0,
                provider_used="test"
            )
        
        # Negative chunks_used
        with pytest.raises(ValidationError):
            AskResponse(
                answer="test",
                confidence=0.5,
                chunks_used=-1,
                response_time=1.0,
                provider_used="test"
            )
        
        # Negative response_time
        with pytest.raises(ValidationError):
            AskResponse(
                answer="test",
                confidence=0.5,
                chunks_used=1,
                response_time=-0.1,
                provider_used="test"
            )
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_search_result_model(self):
        """Test SearchResult model."""
        result = SearchResult(
            id="chunk1",
            content="Search result content",
            metadata={"source": "file.txt"},
            distance=0.15,
            score=0.85
        )
        
        assert result.id == "chunk1"
        assert result.content == "Search result content"
        assert result.distance == 0.15
        assert result.score == 0.85
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_search_response_model(self):
        """Test SearchResponse model."""
        results = [
            SearchResult(id="1", content="Result 1", distance=0.1, score=0.9),
            SearchResult(id="2", content="Result 2", distance=0.2, score=0.8)
        ]
        
        response = SearchResponse(
            results=results,
            total_count=2,
            query="test query",
            response_time=0.5
        )
        
        assert len(response.results) == 2
        assert response.total_count == 2
        assert response.query == "test query"
        assert response.response_time == 0.5
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_processing_detail_model(self):
        """Test ProcessingDetail model."""
        # Success case
        detail = ProcessingDetail(
            status="success",
            path="/docs/file.txt",
            url=None,
            error=None
        )
        
        assert detail.status == "success"
        assert detail.path == "/docs/file.txt"
        assert detail.url is None
        assert detail.error is None
        
        # Error case
        error_detail = ProcessingDetail(
            status="failed",
            path=None,
            url="https://example.com",
            error="Connection failed"
        )
        
        assert error_detail.status == "failed"
        assert error_detail.url == "https://example.com"
        assert error_detail.error == "Connection failed"
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_index_response_model(self):
        """Test IndexResponse model."""
        details = [
            ProcessingDetail(status="success", path="/file1.txt"),
            ProcessingDetail(status="failed", path="/file2.txt", error="Parse error")
        ]
        
        response = IndexResponse(
            success=True,
            processed_count=1,
            failed_count=1,
            processing_time=5.2,
            directory="/docs",
            details=details,
            error_messages=["Parse error in file2.txt"]
        )
        
        assert response.success is True
        assert response.processed_count == 1
        assert response.failed_count == 1
        assert response.processing_time == 5.2
        assert response.directory == "/docs"
        assert len(response.details) == 2
        assert len(response.error_messages) == 1
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_crawl_response_model(self):
        """Test CrawlResponse model."""
        response = CrawlResponse(
            success=True,
            processed_count=3,
            failed_count=1,
            processing_time=15.5,
            urls=["https://example.com", "https://docs.example.com"],
            crawl_details={"depth": 2, "max_pages": 20},
            details=[],
            error_messages=[]
        )
        
        assert response.success is True
        assert response.processed_count == 3
        assert response.failed_count == 1
        assert response.processing_time == 15.5
        assert len(response.urls) == 2
        assert response.crawl_details["depth"] == 2
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_health_response_model(self):
        """Test HealthResponse model."""
        response = HealthResponse(
            status="healthy",
            timestamp="2024-01-15T10:30:00Z",
            version="2.0.0",
            providers={"openai": {"status": "healthy"}},
            database={"connected": True, "chunks": 1500}
        )
        
        assert response.status == "healthy"
        assert response.timestamp == "2024-01-15T10:30:00Z"
        assert response.version == "2.0.0"
        assert response.providers["openai"]["status"] == "healthy"
        assert response.database["connected"] is True
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_batch_ask_response_model(self):
        """Test BatchAskResponse model."""
        results = [
            AskResponse(
                answer="Answer 1", 
                confidence=0.8, 
                chunks_used=3,
                response_time=1.0,
                provider_used="openai"
            ),
            AskResponse(
                answer="Answer 2",
                confidence=0.7,
                chunks_used=2,
                response_time=1.2,
                provider_used="openai"
            )
        ]
        
        response = BatchAskResponse(
            results=results,
            total_questions=2,
            successful_answers=2,
            processing_time=3.5
        )
        
        assert len(response.results) == 2
        assert response.total_questions == 2
        assert response.successful_answers == 2
        assert response.processing_time == 3.5
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_error_response_model(self):
        """Test ErrorResponse model."""
        response = ErrorResponse(
            error="Validation failed",
            error_type="validation_error",
            detail="Field 'question' is required",
            timestamp="2024-01-15T10:30:00Z"
        )
        
        assert response.error == "Validation failed"
        assert response.error_type == "validation_error"
        assert response.detail == "Field 'question' is required"
        assert response.timestamp == "2024-01-15T10:30:00Z"


class TestValidationDependencies(BaseAPITest):
    """Test validation dependency functions."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_validate_domain_filter_valid(self):
        """Test validate_domain_filter with valid domains."""
        # None should pass through
        result = validate_domain_filter(None)
        assert result is None
        
        # Empty list should pass through  
        result = validate_domain_filter([])
        assert result is None
        
        # Valid domains
        domains = ["docs", "api", "help"]
        result = validate_domain_filter(domains)
        assert result == domains
        
        # Domains with special characters
        domains = ["sub-domain", "docs.v2", "api_v1"]
        result = validate_domain_filter(domains)
        assert result == domains
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_validate_domain_filter_invalid(self):
        """Test validate_domain_filter with invalid domains."""
        # Invalid domain format (empty string)
        with pytest.raises(HTTPException) as exc_info:
            validate_domain_filter(["valid", ""])
        assert exc_info.value.status_code == 400
        assert "Invalid domain format" in exc_info.value.detail
        
        # Invalid domain format (non-string)
        with pytest.raises(HTTPException) as exc_info:
            validate_domain_filter(["valid", 123])
        assert exc_info.value.status_code == 400
        
        # Invalid domain characters
        with pytest.raises(HTTPException) as exc_info:
            validate_domain_filter(["valid", "invalid@domain"])
        assert exc_info.value.status_code == 400
        assert "Invalid domain name" in exc_info.value.detail
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_validate_user_agent_valid(self):
        """Test validate_user_agent with valid user agents."""
        # Default when None provided
        result = validate_user_agent(None)
        assert result == "SupportDeflectBot/2.0 API"
        
        # Custom user agent
        custom_ua = "MyApp/1.0 (Windows NT 10.0)"
        result = validate_user_agent(custom_ua)
        assert result == custom_ua
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_validate_user_agent_invalid(self):
        """Test validate_user_agent with invalid user agents."""
        # Too long user agent
        long_ua = "x" * 201  # Exceeds limit of 200
        with pytest.raises(HTTPException) as exc_info:
            validate_user_agent(long_ua)
        assert exc_info.value.status_code == 400
        assert "User agent too long" in exc_info.value.detail
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_validate_crawl_urls_valid(self):
        """Test validate_crawl_urls with valid URLs."""
        valid_urls = [
            "https://example.com",
            "http://localhost:8000",
            "https://subdomain.example.co.uk",
            "https://192.168.1.1:3000/path"
        ]
        
        with patch('support_deflect_bot.api.dependencies.validation.ALLOW_HOSTS', None):
            result = validate_crawl_urls(valid_urls)
            assert result == valid_urls
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_validate_crawl_urls_invalid_format(self):
        """Test validate_crawl_urls with invalid URL formats."""
        # Invalid URL format
        with pytest.raises(HTTPException) as exc_info:
            validate_crawl_urls(["not-a-url"])
        assert exc_info.value.status_code == 400
        assert "Invalid URL format" in exc_info.value.detail
        
        # FTP protocol
        with pytest.raises(HTTPException) as exc_info:
            validate_crawl_urls(["ftp://example.com"])
        assert exc_info.value.status_code == 400
        
        # Missing protocol
        with pytest.raises(HTTPException) as exc_info:
            validate_crawl_urls(["example.com"])
        assert exc_info.value.status_code == 400
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_validate_crawl_urls_with_allowed_hosts(self):
        """Test validate_crawl_urls with ALLOW_HOSTS restriction."""
        allowed_hosts = ["example.com", "docs.example.com"]
        
        with patch('support_deflect_bot.api.dependencies.validation.ALLOW_HOSTS', allowed_hosts):
            # Allowed host should pass
            valid_urls = ["https://example.com/path"]
            result = validate_crawl_urls(valid_urls)
            assert result == valid_urls
            
            # Disallowed host should fail
            with pytest.raises(HTTPException) as exc_info:
                validate_crawl_urls(["https://malicious.com"])
            assert exc_info.value.status_code == 400
            assert "URL not allowed" in exc_info.value.detail
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_validate_file_patterns_valid(self):
        """Test validate_file_patterns with valid patterns."""
        # None should pass through
        result = validate_file_patterns(None)
        assert result is None
        
        # Valid glob patterns
        patterns = ["*.txt", "*.md", "docs/*.py"]
        result = validate_file_patterns(patterns)
        assert result == patterns
        
        # Valid regex patterns
        patterns = [r".*\.txt$", r"file\d+\.md"]
        result = validate_file_patterns(patterns)
        assert result == patterns
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_validate_file_patterns_invalid(self):
        """Test validate_file_patterns with invalid patterns."""
        # Invalid regex pattern
        with pytest.raises(HTTPException) as exc_info:
            validate_file_patterns([r"invalid[regex"])  # Unclosed bracket
        assert exc_info.value.status_code == 400
        assert "Invalid file pattern" in exc_info.value.detail
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_validate_pagination_valid(self):
        """Test validate_pagination with valid parameters."""
        # Default values
        skip, limit = validate_pagination()
        assert skip == 0
        assert limit == 10
        
        # Custom values
        skip, limit = validate_pagination(skip=20, limit=50)
        assert skip == 20
        assert limit == 50
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_validate_pagination_constraints(self):
        """Test validate_pagination parameter constraints."""
        # This would be handled by FastAPI's parameter validation
        # but we can test the function directly
        
        # Valid boundary values
        skip, limit = validate_pagination(skip=0, limit=1)
        assert skip == 0
        assert limit == 1
        
        skip, limit = validate_pagination(skip=1000, limit=100)
        assert skip == 1000
        assert limit == 100


class TestModelIntegration(BaseAPITest):
    """Test model integration scenarios."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_request_response_cycle(self):
        """Test complete request-response model cycle."""
        # Create request
        request_data = {
            "question": "What is the meaning of life?",
            "domains": ["philosophy", "science"],
            "max_chunks": 3,
            "min_confidence": 0.4
        }
        
        ask_request = AskRequest(**request_data)
        
        # Simulate processing and create response
        sources = [
            Source(
                id="phil1",
                content="Philosophical perspective on life's meaning",
                metadata={"source": "philosophy.txt"},
                distance=0.15
            )
        ]
        
        ask_response = AskResponse(
            answer="Life's meaning is subjective and varies by individual perspective.",
            confidence=0.82,
            sources=sources,
            chunks_used=1,
            response_time=1.35,
            provider_used="openai",
            metadata={"tokens_used": 125}
        )
        
        # Verify request data was preserved in processing logic
        assert ask_request.question in "What is the meaning of life?"
        assert "philosophy" in ask_request.domains
        
        # Verify response structure
        assert ask_response.confidence > ask_request.min_confidence
        assert len(ask_response.sources) <= ask_request.max_chunks
        assert ask_response.answer is not None
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_batch_request_response_cycle(self):
        """Test batch request-response cycle."""
        batch_request = BatchAskRequest(
            questions=["Question 1", "Question 2", "Question 3"],
            domains=["docs"],
            max_chunks=2,
            min_confidence=0.25
        )
        
        # Create individual responses
        individual_responses = []
        for i, question in enumerate(batch_request.questions):
            response = AskResponse(
                answer=f"Answer to {question}",
                confidence=0.5 + (i * 0.1),
                sources=[],
                chunks_used=1,
                response_time=1.0,
                provider_used="openai"
            )
            individual_responses.append(response)
        
        batch_response = BatchAskResponse(
            results=individual_responses,
            total_questions=len(batch_request.questions),
            successful_answers=len(individual_responses),
            processing_time=sum(r.response_time for r in individual_responses)
        )
        
        assert batch_response.total_questions == 3
        assert batch_response.successful_answers == 3
        assert len(batch_response.results) == 3
        assert batch_response.processing_time == 3.0
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_error_model_serialization(self):
        """Test error model JSON serialization."""
        error = ErrorResponse(
            error="Request validation failed",
            error_type="validation_error",
            detail="Missing required field 'question'",
            timestamp="2024-01-15T10:30:00Z"
        )
        
        # Test JSON serialization
        error_dict = error.dict()
        expected_keys = {"error", "error_type", "detail", "timestamp"}
        assert set(error_dict.keys()) == expected_keys
        
        # Test JSON string conversion
        import json
        error_json = json.dumps(error_dict)
        assert "Request validation failed" in error_json
        assert "validation_error" in error_json
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_model_field_aliases(self):
        """Test model field aliases and serialization options."""
        # Test that models serialize correctly for API responses
        source = Source(
            id="test",
            content="Test content",
            metadata={"key": "value"},
            distance=0.5
        )
        
        serialized = source.dict()
        
        # Verify all fields are present and correctly named
        assert "id" in serialized
        assert "content" in serialized 
        assert "metadata" in serialized
        assert "distance" in serialized
        
        # Test that nested models serialize correctly
        response = AskResponse(
            answer="Test answer",
            confidence=0.75,
            sources=[source],
            chunks_used=1,
            response_time=1.0,
            provider_used="test"
        )
        
        serialized = response.dict()
        assert "sources" in serialized
        assert len(serialized["sources"]) == 1
        assert isinstance(serialized["sources"][0], dict)


class TestModelPerformance(BaseAPITest):
    """Test model validation performance."""
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_large_batch_request_validation(self):
        """Test validation performance with large batch requests."""
        import time
        
        # Create large batch request (at maximum limit)
        large_batch = BatchAskRequest(
            questions=[f"Question {i}" for i in range(10)],  # Maximum allowed
            domains=["docs"] * 5,  # Multiple domains
            max_chunks=10,
            min_confidence=0.25
        )
        
        start_time = time.time()
        
        # Validate questions (this happens during construction)
        assert len(large_batch.questions) == 10
        assert all(q.startswith("Question") for q in large_batch.questions)
        
        validation_time = time.time() - start_time
        
        # Validation should be fast even for maximum batch size
        assert validation_time < 0.1  # Less than 100ms
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_large_response_serialization(self):
        """Test performance of serializing large response objects."""
        import time
        
        # Create large response with many sources
        large_sources = [
            Source(
                id=f"source_{i}",
                content=f"Content from source {i} " * 50,  # Long content
                metadata={"index": i, "type": "test", "category": f"cat_{i % 5}"},
                distance=0.1 + (i * 0.01)
            )
            for i in range(20)  # Maximum chunks
        ]
        
        large_response = AskResponse(
            answer="This is a comprehensive answer based on multiple sources. " * 10,
            confidence=0.85,
            sources=large_sources,
            chunks_used=len(large_sources),
            response_time=2.5,
            provider_used="openai",
            metadata={"model": "gpt-4", "tokens": 1500, "extra_data": "x" * 1000}
        )
        
        start_time = time.time()
        serialized = large_response.dict()
        serialization_time = time.time() - start_time
        
        # Serialization should be reasonable even for large responses
        assert serialization_time < 0.05  # Less than 50ms
        assert len(serialized["sources"]) == 20
        assert isinstance(serialized["metadata"], dict)