"""
Global pytest fixtures for Support Deflect Bot unified architecture testing.

This file provides shared fixtures for testing all components of the unified architecture
including engine services, providers, CLI, and API interfaces.
"""

import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Ensure src module is available for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import unified engine services for mocking
from support_deflect_bot.engine import (
    UnifiedRAGEngine,
    UnifiedDocumentProcessor,
    UnifiedQueryService,
    UnifiedEmbeddingService
)


# =============================================================================
# DIRECTORY AND FILESYSTEM FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_chroma_db():
    """Provide a temporary ChromaDB path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chroma_path = os.path.join(tmpdir, "test_chroma_db")
        yield chroma_path


# =============================================================================
# MOCK ENGINE SERVICES FIXTURES
# =============================================================================

@pytest.fixture
def mock_rag_engine():
    """Mock UnifiedRAGEngine for testing."""
    mock_engine = Mock(spec=UnifiedRAGEngine)
    mock_engine.query_documents = AsyncMock()
    mock_engine.query_documents.return_value = {
        "answer": "This is a test answer from the unified RAG engine.",
        "confidence": 0.85,
        "sources": ["test_document.md"],
        "chunks": ["Test chunk 1", "Test chunk 2"]
    }
    mock_engine.add_documents = AsyncMock()
    mock_engine.add_documents.return_value = {"status": "success", "count": 5}
    mock_engine.get_stats = Mock()
    mock_engine.get_stats.return_value = {"document_count": 10, "chunk_count": 50}
    return mock_engine


@pytest.fixture
def mock_document_processor():
    """Mock UnifiedDocumentProcessor for testing."""
    mock_processor = Mock(spec=UnifiedDocumentProcessor)
    mock_processor.process_documents = AsyncMock()
    mock_processor.process_documents.return_value = [
        {
            "content": "Test document content",
            "metadata": {"source": "test.md", "type": "markdown"},
            "chunks": ["Chunk 1", "Chunk 2"]
        }
    ]
    mock_processor.process_web_content = AsyncMock()
    mock_processor.process_web_content.return_value = [
        {
            "content": "Web content",
            "metadata": {"url": "https://example.com", "title": "Test Page"}
        }
    ]
    return mock_processor


@pytest.fixture
def mock_query_service():
    """Mock UnifiedQueryService for testing."""
    mock_service = Mock(spec=UnifiedQueryService)
    mock_service.process_query = AsyncMock()
    mock_service.process_query.return_value = {
        "processed_query": "test query",
        "intent": "information_request",
        "confidence": 0.9
    }
    mock_service.validate_query = Mock()
    mock_service.validate_query.return_value = True
    return mock_service


@pytest.fixture
def mock_embedding_service():
    """Mock UnifiedEmbeddingService for testing."""
    mock_service = Mock(spec=UnifiedEmbeddingService)
    # Mock embedding generation to return consistent test vectors
    mock_service.generate_embeddings = AsyncMock()
    mock_service.generate_embeddings.return_value = [
        [0.1] * 384,  # Standard test embedding dimension
        [0.2] * 384
    ]
    mock_service.similarity_search = AsyncMock()
    mock_service.similarity_search.return_value = [
        {"content": "Similar content 1", "score": 0.95},
        {"content": "Similar content 2", "score": 0.88}
    ]
    return mock_service


# =============================================================================
# MOCK PROVIDER FIXTURES
# =============================================================================

@pytest.fixture
def mock_openai_provider():
    """Mock OpenAI provider for testing."""
    mock_provider = Mock()
    mock_provider.name = "openai"
    mock_provider.is_available = Mock(return_value=True)
    mock_provider.generate_response = AsyncMock()
    mock_provider.generate_response.return_value = {
        "content": "OpenAI test response",
        "model": "gpt-4o-mini",
        "usage": {"input_tokens": 100, "output_tokens": 50}
    }
    mock_provider.generate_embeddings = AsyncMock()
    mock_provider.generate_embeddings.return_value = [[0.1] * 1536]  # OpenAI embedding dim
    return mock_provider


@pytest.fixture
def mock_groq_provider():
    """Mock Groq provider for testing."""
    mock_provider = Mock()
    mock_provider.name = "groq"
    mock_provider.is_available = Mock(return_value=True)
    mock_provider.generate_response = AsyncMock()
    mock_provider.generate_response.return_value = {
        "content": "Groq test response",
        "model": "llama-3.1-8b-instant",
        "usage": {"input_tokens": 100, "output_tokens": 50}
    }
    return mock_provider


@pytest.fixture
def mock_ollama_provider():
    """Mock Ollama provider for testing."""
    mock_provider = Mock()
    mock_provider.name = "ollama"
    mock_provider.is_available = Mock(return_value=True)
    mock_provider.generate_response = AsyncMock()
    mock_provider.generate_response.return_value = {
        "content": "Ollama test response",
        "model": "llama3.1",
        "usage": {"input_tokens": 100, "output_tokens": 50}
    }
    mock_provider.generate_embeddings = AsyncMock()
    mock_provider.generate_embeddings.return_value = [[0.1] * 768]  # Ollama embedding dim
    return mock_provider


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_documents():
    """Sample documentation content for testing."""
    return {
        "getting_started.md": """# Getting Started with Support Deflect Bot

This guide will help you set up and use the Support Deflect Bot.

## Installation

1. Clone the repository
2. Install dependencies: `pip install -e .`
3. Configure your API keys

## Basic Usage

Use the CLI command `deflect-bot ask` to start asking questions.
""",
        "configuration.md": """# Configuration Guide

## Environment Variables

Set these environment variables for optimal performance:

- `OPENAI_API_KEY`: Your OpenAI API key
- `CHROMA_DB_PATH`: Path to ChromaDB storage
- `MAX_CHUNKS`: Maximum chunks to retrieve (default: 5)

## Provider Settings

The bot supports multiple AI providers with automatic fallback.
""",
        "api_reference.md": """# API Reference

## Endpoints

### POST /ask
Ask a question to the bot.

**Request Body:**
```json
{
    "question": "How do I install the bot?",
    "max_chunks": 5
}
```

**Response:**
```json
{
    "answer": "Installation instructions...",
    "confidence": 0.85,
    "sources": ["getting_started.md"]
}
```
"""
    }


@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors for testing."""
    return {
        "query_embedding": [0.1, 0.2, 0.3] * 128,  # 384-dim test embedding
        "document_embeddings": [
            [0.2, 0.3, 0.4] * 128,
            [0.3, 0.4, 0.5] * 128,
            [0.4, 0.5, 0.6] * 128
        ]
    }


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    return [
        {
            "content": "This is a test document chunk about installation.",
            "metadata": {
                "source": "getting_started.md",
                "chunk_id": "chunk_1",
                "section": "Installation"
            },
            "embedding": [0.1] * 384
        },
        {
            "content": "This chunk explains configuration options.",
            "metadata": {
                "source": "configuration.md", 
                "chunk_id": "chunk_2",
                "section": "Environment Variables"
            },
            "embedding": [0.2] * 384
        }
    ]


@pytest.fixture
def sample_api_responses():
    """Sample API responses from providers for testing."""
    return {
        "openai_response": {
            "choices": [{
                "message": {
                    "content": "This is a test response from OpenAI GPT-4",
                    "role": "assistant"
                }
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            },
            "model": "gpt-4o-mini"
        },
        "groq_response": {
            "choices": [{
                "message": {
                    "content": "This is a test response from Groq",
                    "role": "assistant"
                }
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            },
            "model": "llama-3.1-8b-instant"
        }
    }


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def test_settings():
    """Test settings configuration."""
    return {
        "APP_NAME": "Support Deflect Bot Test",
        "CHROMA_DB_PATH": "./test_chroma_db",
        "ANSWER_MIN_CONF": 0.25,
        "MAX_CHUNKS": 5,
        "MAX_CHARS_PER_CHUNK": 800,
        "MONTHLY_BUDGET_USD": 10.0,
        "DEFAULT_PROVIDER_STRATEGY": "cost_optimized"
    }


@pytest.fixture
def mock_settings(test_settings):
    """Mock settings module with test configuration."""
    with patch.multiple(
        'support_deflect_bot.utils.settings',
        **test_settings
    ) as mock:
        yield mock


# =============================================================================
# HTTP CLIENT FIXTURES (for API testing)
# =============================================================================

@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for testing HTTP requests."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_response.text = "Success"
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.get = AsyncMock(return_value=mock_response)
    return mock_client


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically clean up any test files after each test."""
    yield
    # Cleanup logic could go here if needed
    # For now, we rely on temp directories for isolation


# =============================================================================
# INTEGRATION TEST FIXTURES
# =============================================================================

@pytest.fixture
def integration_setup():
    """Setup for integration tests that need multiple components."""
    return {
        "temp_db_path": tempfile.mkdtemp(),
        "test_documents": ["test1.md", "test2.md"],
        "mock_api_keys": {
            "OPENAI_API_KEY": "test_key_openai",
            "GROQ_API_KEY": "test_key_groq"
        }
    }


# =============================================================================
# CLI TESTING FIXTURES
# =============================================================================

@pytest.fixture
def cli_runner():
    """Click CLI test runner for testing CLI commands."""
    from click.testing import CliRunner
    return CliRunner()


@pytest.fixture
def mock_cli_output():
    """Mock CLI output formatting for testing."""
    mock_console = Mock()
    mock_console.print = Mock()
    return mock_console


# =============================================================================
# MARKS AND CONFIGURATION
# =============================================================================

# Define test markers for better organization
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for isolated components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interaction"  
    )
    config.addinivalue_line(
        "markers", "system: System-level end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "requires_api: Tests requiring real API access (guarded)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow-running tests (>10 seconds)"
    )