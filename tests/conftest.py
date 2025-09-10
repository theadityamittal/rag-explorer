import json
import os
import tempfile
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_docs():
    """Sample documentation content for testing."""
    return {
        "test_doc1.md": """# Getting Started
        
This is a test document about getting started.
It contains information about setup and configuration.

## Installation
Run `pip install package-name` to install.

## Usage
Use `command --flag` to run the tool.""",
        "test_doc2.md": """# Advanced Configuration

This document covers advanced configuration options.

## Environment Variables
Set `API_KEY=your-key` in your environment.
Configure `DEBUG=true` for development mode.""",
    }


@pytest.fixture
def mock_ollama_response():
    """Mock response from Ollama API."""
    return {
        "embedding": [0.1] * 768,  # Mock 768-dimensional embedding
        "message": {"content": "This is a test response from the LLM."},
    }


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection."""
    mock_collection = Mock()
    mock_collection.add = Mock()
    mock_collection.query = Mock(
        return_value={
            "documents": [["Sample document content for testing"]],
            "metadatas": [[{"path": "test.md", "chunk_id": 0}]],
            "distances": [[0.5]],
        }
    )
    mock_collection.count = Mock(return_value=10)
    return mock_collection


@pytest.fixture
def mock_settings():
    """Mock application settings."""
    return {
        "OLLAMA_MODEL": "llama3.1",
        "OLLAMA_EMBED_MODEL": "nomic-embed-text",
        "CHROMA_DB_PATH": "./test_chroma_db",
        "CHROMA_COLLECTION": "test_knowledge_base",
        "ANSWER_MIN_CONF": 0.20,
        "MAX_CHUNKS": 5,
        "MAX_CHARS_PER_CHUNK": 800,
        "ALLOW_HOSTS": {"docs.python.org", "test.example.com"},
        "TRUSTED_DOMAINS": {"test.example.com"},
        "DOCS_FOLDER": "./test_docs",
        "CRAWL_CACHE_PATH": "./test_cache.json",
    }


@pytest.fixture
def sample_crawl_cache(temp_dir):
    """Sample crawl cache data for testing."""
    cache_data = {
        "https://test.example.com/doc1": {
            "etag": "test-etag-123",
            "last_modified": "Wed, 21 Oct 2015 07:28:00 GMT",
            "content_hash": "abc123def456",
            "timestamp": 1635724800,
            "status_code": 200,
        },
        "https://test.example.com/doc2": {
            "etag": "test-etag-456",
            "last_modified": "Thu, 22 Oct 2015 08:30:00 GMT",
            "content_hash": "def456ghi789",
            "timestamp": 1635811200,
            "status_code": 200,
        },
    }
    cache_path = os.path.join(temp_dir, "test_cache.json")
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)
    return cache_path


@pytest.fixture
def mock_requests_response():
    """Mock HTTP response from requests."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {
        "content-type": "text/html",
        "etag": "test-etag-789",
        "last-modified": "Fri, 23 Oct 2015 09:30:00 GMT",
    }
    mock_response.text = """
    <html>
        <head><title>Test Document</title></head>
        <body>
            <h1>Test Document</h1>
            <p>This is a test document for web crawling.</p>
            <a href="/doc2">Link to doc2</a>
        </body>
    </html>
    """
    return mock_response
