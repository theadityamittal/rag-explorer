import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
from src.api.app import app


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_healthz(self):
        """Test health check endpoint."""
        client = TestClient(app)
        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestReindexEndpoint:
    """Test the reindex endpoint."""

    @patch("src.api.app.ingest_folder")
    def test_reindex_success(self, mock_ingest):
        """Test successful reindexing."""
        mock_ingest.return_value = 42

        client = TestClient(app)
        response = client.post("/reindex")

        assert response.status_code == 200
        assert response.json() == {"chunks_indexed": 42}
        mock_ingest.assert_called_once()

    @patch("src.api.app.ingest_folder")
    def test_reindex_connection_error(self, mock_ingest):
        """Test reindexing with connection error."""
        mock_ingest.side_effect = ConnectionError("DB connection failed")

        client = TestClient(app)
        response = client.post("/reindex")

        assert response.status_code == 503
        assert "Database connection failed" in response.json()["detail"]

    @patch("src.api.app.ingest_folder")
    def test_reindex_file_not_found(self, mock_ingest):
        """Test reindexing with missing docs folder."""
        mock_ingest.side_effect = FileNotFoundError("Docs folder not found")

        client = TestClient(app)
        response = client.post("/reindex")

        assert response.status_code == 404
        assert "Documentation folder not found" in response.json()["detail"]

    @patch("src.api.app.ingest_folder")
    def test_reindex_general_error(self, mock_ingest):
        """Test reindexing with general error."""
        mock_ingest.side_effect = Exception("General error")

        client = TestClient(app)
        response = client.post("/reindex")

        assert response.status_code == 500
        assert "Indexing failed" in response.json()["detail"]


class TestSearchEndpoint:
    """Test the search endpoint."""

    @patch("src.api.app.retrieve")
    def test_search_success(self, mock_retrieve):
        """Test successful search."""
        mock_retrieve.return_value = [
            {
                "text": "Sample document content for testing search functionality",
                "meta": {"path": "test.md", "chunk_id": 0},
                "distance": 0.3,
            },
            {
                "text": "Another document chunk",
                "meta": {"path": "test2.md", "chunk_id": 1},
                "distance": 0.5,
            },
        ]

        client = TestClient(app)
        response = client.post("/search", json={"query": "test query", "k": 2})

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert len(data["results"]) == 2
        assert data["results"][0]["path"] == "test.md"
        assert data["results"][0]["distance"] == 0.3

    @patch("src.api.app.retrieve")
    def test_search_connection_error(self, mock_retrieve):
        """Test search with connection error."""
        mock_retrieve.side_effect = ConnectionError("DB connection failed")

        client = TestClient(app)
        response = client.post("/search", json={"query": "test query"})

        assert response.status_code == 503
        assert "Database connection failed" in response.json()["detail"]

    def test_search_invalid_request(self):
        """Test search with invalid request data."""
        client = TestClient(app)

        # Empty query
        response = client.post("/search", json={"query": ""})
        assert response.status_code == 422

        # Missing query
        response = client.post("/search", json={"k": 5})
        assert response.status_code == 422

        # Invalid k value
        response = client.post("/search", json={"query": "test", "k": 0})
        assert response.status_code == 422


class TestAskEndpoint:
    """Test the ask endpoint."""

    @patch("src.api.app.answer_question")
    def test_ask_success(self, mock_answer):
        """Test successful question answering."""
        mock_answer.return_value = {
            "answer": "This is the answer to your question.",
            "citations": [{"rank": 1, "path": "test.md", "preview": "Sample content"}],
            "confidence": 0.8,
        }

        client = TestClient(app)
        response = client.post("/ask", json={"question": "How do I test something?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is the answer to your question."
        assert len(data["citations"]) == 1
        assert data["confidence"] == 0.8

    @patch("src.api.app.answer_question")
    def test_ask_with_domains(self, mock_answer):
        """Test asking with domain filtering."""
        mock_answer.return_value = {
            "answer": "Domain-specific answer.",
            "citations": [],
            "confidence": 0.6,
        }

        client = TestClient(app)
        response = client.post(
            "/ask",
            json={
                "question": "Domain-specific question?",
                "domains": ["docs.example.com"],
            },
        )

        assert response.status_code == 200
        mock_answer.assert_called_once_with(
            "Domain-specific question?", domains=["docs.example.com"]
        )

    @patch("src.api.app.answer_question")
    def test_ask_refusal(self, mock_answer):
        """Test question refusal due to low confidence."""
        mock_answer.return_value = {
            "answer": "I don't have enough information in the docs to answer that.",
            "citations": [],
            "confidence": 0.1,
        }

        client = TestClient(app)
        response = client.post("/ask", json={"question": "Unknown topic?"})

        assert response.status_code == 200
        data = response.json()
        assert "don't have enough information" in data["answer"]
        assert data["confidence"] == 0.1

    def test_ask_invalid_request(self):
        """Test ask with invalid request data."""
        client = TestClient(app)

        # Empty question
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422

        # Question too long
        long_question = "x" * 1001
        response = client.post("/ask", json={"question": long_question})
        assert response.status_code == 422

        # Too many domains
        many_domains = [f"domain{i}.com" for i in range(11)]
        response = client.post(
            "/ask", json={"question": "test", "domains": many_domains}
        )
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Test the metrics endpoint."""

    def test_metrics_endpoint(self):
        """Test metrics endpoint returns expected structure."""
        client = TestClient(app)
        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "ask" in data
        assert "search" in data
        assert "version" in data

        # Check metrics structure
        assert "count" in data["ask"]
        assert "p50_ms" in data["ask"]
        assert "p95_ms" in data["ask"]


class TestLLMPingEndpoint:
    """Test the LLM ping endpoint."""

    @patch("src.api.app.llm_echo")
    def test_llm_ping_success(self, mock_llm):
        """Test successful LLM ping."""
        mock_llm.return_value = "Yeah Yeah! I'm awake!"

        client = TestClient(app)
        response = client.get("/llm_ping")

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "ollama"
        assert data["reply"] == "Yeah Yeah! I'm awake!"

    @patch("src.api.app.llm_echo")
    def test_llm_ping_failure(self, mock_llm):
        """Test LLM ping failure."""
        mock_llm.side_effect = Exception("LLM service down")

        client = TestClient(app)
        response = client.get("/llm_ping")

        assert response.status_code == 503
        assert "LLM service unavailable" in response.json()["detail"]


class TestBatchAskEndpoint:
    """Test the batch ask endpoint."""

    @patch("src.api.app.batch_ask")
    def test_batch_ask_success(self, mock_batch):
        """Test successful batch asking."""
        mock_batch.return_value = [
            {"question": "Q1", "answer": "A1", "confidence": 0.8},
            {"question": "Q2", "answer": "A2", "confidence": 0.6},
        ]

        client = TestClient(app)
        response = client.post(
            "/batch_ask", json={"questions": ["How to install?", "How to configure?"]}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2

    def test_batch_ask_validation(self):
        """Test batch ask request validation."""
        client = TestClient(app)

        # Empty questions list
        response = client.post("/batch_ask", json={"questions": []})
        assert response.status_code == 422

        # Too many questions
        many_questions = [f"Question {i}?" for i in range(11)]
        response = client.post("/batch_ask", json={"questions": many_questions})
        assert response.status_code == 422

        # Empty question in list
        response = client.post("/batch_ask", json={"questions": ["Valid question", ""]})
        assert response.status_code == 422


class TestCrawlEndpoints:
    """Test web crawling endpoints."""

    @patch("src.api.app.index_urls")
    def test_crawl_success(self, mock_index):
        """Test successful URL crawling."""
        mock_index.return_value = {"indexed": 2, "skipped": 0}

        client = TestClient(app)
        response = client.post(
            "/crawl",
            json={
                "urls": ["https://example.com/doc1", "https://example.com/doc2"],
                "force": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        mock_index.assert_called_once_with(
            ["https://example.com/doc1", "https://example.com/doc2"], force=False
        )

    @patch("src.api.app.crawl_urls")
    def test_crawl_depth_success(self, mock_crawl):
        """Test successful depth crawling."""
        mock_crawl.return_value = {"pages_crawled": 5, "pages_indexed": 4}

        client = TestClient(app)
        response = client.post(
            "/crawl_depth",
            json={
                "seeds": ["https://example.com"],
                "depth": 2,
                "max_pages": 10,
                "same_domain": True,
                "force": False,
            },
        )

        assert response.status_code == 200
        mock_crawl.assert_called_once()

    def test_crawl_validation(self):
        """Test crawl request validation."""
        client = TestClient(app)

        # Invalid URL
        response = client.post("/crawl", json={"urls": ["not-a-url"]})
        assert response.status_code == 422

        # Too many URLs
        many_urls = [f"https://example{i}.com" for i in range(51)]
        response = client.post("/crawl", json={"urls": many_urls})
        assert response.status_code == 422

    @patch("src.api.app.crawl_urls")
    def test_crawl_default_success(self, mock_crawl):
        """Test default crawling."""
        mock_crawl.return_value = {"pages_crawled": 3}

        client = TestClient(app)
        response = client.post("/crawl_default")

        assert response.status_code == 200
        mock_crawl.assert_called_once()
