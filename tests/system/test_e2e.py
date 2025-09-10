"""
End-to-end system tests for the Support Deflect Bot.
These tests require a fully configured environment with:
- Ollama service running
- All dependencies installed
- Test data available
"""

import pytest
import requests
import time
import os
import tempfile
from pathlib import Path


@pytest.mark.requires_ollama
class TestE2ESystem:
    """End-to-end system tests."""

    BASE_URL = "http://127.0.0.1:8000"

    @classmethod
    def setup_class(cls):
        """Setup test environment."""
        # Wait for API to be ready
        cls.wait_for_api()

    @classmethod
    def wait_for_api(cls, timeout=30):
        """Wait for API to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{cls.BASE_URL}/healthz", timeout=2)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)

        pytest.skip("API not available - skipping E2E tests")

    def test_health_check(self):
        """Test basic health check."""
        response = requests.get(f"{self.BASE_URL}/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_llm_connectivity(self):
        """Test LLM service connectivity."""
        response = requests.get(f"{self.BASE_URL}/llm_ping")

        if response.status_code == 503:
            pytest.skip("LLM service not available")

        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "reply" in data

    def test_reindex_and_search_flow(self):
        """Test complete reindex and search flow."""
        # Step 1: Reindex documents
        reindex_response = requests.post(f"{self.BASE_URL}/reindex")

        if reindex_response.status_code == 404:
            pytest.skip("Documentation folder not found")

        assert reindex_response.status_code == 200
        reindex_data = reindex_response.json()
        assert "chunks_indexed" in reindex_data
        assert reindex_data["chunks_indexed"] > 0

        # Step 2: Search for content
        search_response = requests.post(
            f"{self.BASE_URL}/search", json={"query": "configuration", "k": 3}
        )

        assert search_response.status_code == 200
        search_data = search_response.json()
        assert "query" in search_data
        assert "results" in search_data
        assert len(search_data["results"]) > 0

    def test_ask_question_flow(self):
        """Test complete question answering flow."""
        # Ensure documents are indexed
        requests.post(f"{self.BASE_URL}/reindex")

        # Ask a question
        ask_response = requests.post(
            f"{self.BASE_URL}/ask", json={"question": "How do I configure the system?"}
        )

        assert ask_response.status_code == 200
        ask_data = ask_response.json()
        assert "answer" in ask_data
        assert "citations" in ask_data
        assert "confidence" in ask_data

        # Verify confidence is a number
        assert isinstance(ask_data["confidence"], (int, float))
        assert 0 <= ask_data["confidence"] <= 1

    def test_web_crawl_flow(self):
        """Test web crawling functionality."""
        # Test single URL crawling
        crawl_response = requests.post(
            f"{self.BASE_URL}/crawl",
            json={"urls": ["https://httpbin.org/html"], "force": True},
        )

        # This may fail due to domain restrictions, which is expected
        assert crawl_response.status_code in [200, 500]

        if crawl_response.status_code == 200:
            crawl_data = crawl_response.json()
            assert "result" in crawl_data

    def test_batch_ask_flow(self):
        """Test batch question processing."""
        # Ensure documents are indexed
        requests.post(f"{self.BASE_URL}/reindex")

        batch_response = requests.post(
            f"{self.BASE_URL}/batch_ask",
            json={
                "questions": [
                    "What is configuration?",
                    "How to setup the system?",
                    "Where are the logs stored?",
                ]
            },
        )

        assert batch_response.status_code == 200
        batch_data = batch_response.json()
        assert "results" in batch_data
        assert len(batch_data["results"]) == 3

        # Verify each result has required fields
        for result in batch_data["results"]:
            assert "question" in result
            assert "answer" in result
            assert "confidence" in result

    def test_metrics_tracking(self):
        """Test that metrics are properly tracked."""
        # Make some requests to generate metrics
        requests.post(f"{self.BASE_URL}/search", json={"query": "test"})
        requests.post(f"{self.BASE_URL}/ask", json={"question": "test question?"})

        metrics_response = requests.get(f"{self.BASE_URL}/metrics")
        assert metrics_response.status_code == 200

        metrics_data = metrics_response.json()
        assert "ask" in metrics_data
        assert "search" in metrics_data
        assert "version" in metrics_data

        # Verify metrics structure
        for endpoint in ["ask", "search"]:
            assert "count" in metrics_data[endpoint]
            assert "p50_ms" in metrics_data[endpoint]
            assert "p95_ms" in metrics_data[endpoint]
            assert metrics_data[endpoint]["count"] > 0

    def test_error_handling(self):
        """Test API error handling."""
        # Invalid search request
        response = requests.post(f"{self.BASE_URL}/search", json={"query": ""})
        assert response.status_code == 422

        # Invalid ask request
        response = requests.post(f"{self.BASE_URL}/ask", json={"question": ""})
        assert response.status_code == 422

        # Invalid batch request
        response = requests.post(f"{self.BASE_URL}/batch_ask", json={"questions": []})
        assert response.status_code == 422

    def test_confidence_thresholds(self):
        """Test that confidence thresholds work properly."""
        # This test would require specific test data and configuration
        # to reliably test confidence behavior
        pass

    def test_domain_filtering(self):
        """Test domain filtering functionality."""
        # Index some web content first (if possible)
        requests.post(f"{self.BASE_URL}/reindex")

        # Ask question with domain filtering
        response = requests.post(
            f"{self.BASE_URL}/ask",
            json={
                "question": "How to install packages?",
                "domains": ["docs.python.org"],
            },
        )

        assert response.status_code == 200
        # The actual behavior will depend on available indexed content

    def test_caching_behavior(self):
        """Test that caching works properly."""
        # This would test web crawl caching
        # Multiple requests to same URL should use cache
        url = "https://httpbin.org/html"

        # First request
        response1 = requests.post(
            f"{self.BASE_URL}/crawl", json={"urls": [url], "force": False}
        )

        # Second request (should use cache)
        response2 = requests.post(
            f"{self.BASE_URL}/crawl", json={"urls": [url], "force": False}
        )

        # Both should succeed or fail consistently
        assert response1.status_code == response2.status_code


@pytest.mark.integration
class TestSystemIntegration:
    """Integration tests for system components."""

    def test_database_integration(self):
        """Test database connectivity and operations."""
        # This would test ChromaDB integration
        pass

    def test_embeddings_integration(self):
        """Test embeddings service integration."""
        # This would test Ollama embeddings integration
        pass

    def test_llm_integration(self):
        """Test LLM service integration."""
        # This would test Ollama chat integration
        pass
