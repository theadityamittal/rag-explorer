# Test Suite Documentation

This directory contains comprehensive tests for the Support Deflect Bot application.

## Test Structure

```
tests/
├── conftest.py              # Global test fixtures and configuration
├── pytest.ini              # Pytest configuration (in root directory)
├── test_runner.py           # Custom test runner script
├── unit/                    # Unit tests for individual components
│   ├── test_chunker.py      # Text chunking functionality
│   ├── test_embeddings.py   # Embedding generation and processing
│   ├── test_rag.py          # RAG pipeline components
│   └── test_store.py        # ChromaDB storage operations
├── integration/             # Integration tests for API endpoints
│   └── test_api.py          # FastAPI endpoint testing
├── system/                  # End-to-end system tests
│   └── test_e2e.py          # Complete workflow testing
└── fixtures/                # Test data and fixtures
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual functions and classes in isolation
- **Dependencies**: Minimal - use mocks for external services
- **Coverage**: Core business logic, utility functions, data processing
- **Speed**: Fast (< 1 second per test)

### Integration Tests (`tests/integration/`)
- **Purpose**: Test API endpoints and component interactions
- **Dependencies**: FastAPI TestClient, mocked external services
- **Coverage**: HTTP endpoints, request/response validation, error handling
- **Speed**: Medium (1-5 seconds per test)

### System Tests (`tests/system/`)
- **Purpose**: Test complete end-to-end workflows
- **Dependencies**: Full application stack, Ollama service, real databases
- **Coverage**: Complete user workflows, performance, system behavior
- **Speed**: Slow (5-30 seconds per test)

## Running Tests

### Prerequisites
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov httpx

# For system tests, ensure Ollama is running
ollama pull llama3.1
ollama pull nomic-embed-text
```

### Quick Commands
```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only  
pytest tests/integration/ -v

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_chunker.py -v

# Run specific test function
pytest tests/unit/test_rag.py::TestRagUtilities::test_stem_simple -v
```

### Using the Test Runner
```bash
# Quick tests (no external dependencies)
python tests/test_runner.py --quick

# Unit tests only
python tests/test_runner.py --unit

# Integration tests only
python tests/test_runner.py --integration

# Tests requiring Ollama
python tests/test_runner.py --ollama

# All tests (default)
python tests/test_runner.py --all
```

## Test Markers

Tests are organized with pytest markers:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests  
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.requires_ollama`: Tests needing Ollama service

### Running by Markers
```bash
# Skip slow tests
pytest -m "not slow"

# Run only unit tests
pytest -m "unit"

# Skip tests requiring Ollama
pytest -m "not requires_ollama"
```

## Test Configuration

### pytest.ini
Located in project root, configures:
- Test discovery patterns
- Default options and markers
- Warning filters
- Output formatting

### conftest.py
Provides shared fixtures:
- `temp_dir`: Temporary directory for test files
- `sample_docs`: Mock documentation content
- `mock_ollama_response`: Mocked Ollama API responses
- `mock_chroma_collection`: Mocked ChromaDB collection
- `sample_crawl_cache`: Test web crawl cache data

## Test Data and Fixtures

### Sample Documents
```python
sample_docs = {
    "getting_started.md": "# Getting Started\nInstall with pip install...",
    "advanced.md": "# Advanced Configuration\nSet API_KEY=..."
}
```

### Mock Responses
- Ollama embeddings: 768-dimensional float vectors
- ChromaDB queries: Document chunks with metadata
- HTTP responses: HTML content with proper headers

## Writing New Tests

### Unit Test Example
```python
import pytest
from src.module import function_to_test

class TestFunctionToTest:
    def test_basic_functionality(self):
        result = function_to_test("input")
        assert result == "expected_output"
    
    def test_error_handling(self):
        with pytest.raises(ValueError):
            function_to_test("invalid_input")
```

### Integration Test Example
```python
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api.app import app

def test_api_endpoint():
    client = TestClient(app)
    with patch('src.api.app.external_service') as mock_service:
        mock_service.return_value = "mocked_result"
        response = client.post("/endpoint", json={"data": "test"})
        assert response.status_code == 200
```

### System Test Example
```python
import requests

@pytest.mark.requires_ollama
def test_complete_workflow():
    # Index documents
    reindex_response = requests.post("http://127.0.0.1:8000/reindex")
    assert reindex_response.status_code == 200
    
    # Ask question
    ask_response = requests.post("http://127.0.0.1:8000/ask", 
                                json={"question": "How to install?"})
    assert ask_response.status_code == 200
```

## Test Best Practices

### 1. Test Isolation
- Each test should be independent
- Use fixtures for setup/teardown
- Mock external dependencies
- Clean up resources after tests

### 2. Test Naming
- Use descriptive test names: `test_function_behavior_condition`
- Group related tests in classes: `TestClassName`
- Use consistent naming patterns

### 3. Assertions
- One logical assertion per test
- Use specific assertion methods: `assert_called_once_with()`
- Include meaningful error messages

### 4. Mocking
- Mock external services (Ollama, ChromaDB, HTTP requests)
- Mock at the boundary of your system
- Use `patch` decorators for clean mocking

### 5. Test Data
- Keep test data minimal and focused
- Use fixtures for reusable test data
- Generate data programmatically when possible

## Continuous Integration

### GitHub Actions Example
```yaml
- name: Run Tests
  run: |
    pip install -r requirements.txt
    pip install pytest pytest-cov
    pytest tests/ --cov=src --cov-report=xml
```

### Pre-commit Hooks
```yaml
- repo: local
  hooks:
    - id: pytest-check
      name: pytest-check
      entry: pytest tests/unit/ --tb=short
      language: system
      pass_filenames: false
```

## Performance Testing

### Load Testing
Use system tests to verify performance:
```python
import time

def test_response_time():
    start = time.time()
    response = requests.post("/ask", json={"question": "test"})
    duration = time.time() - start
    assert duration < 2.0  # Should respond within 2 seconds
```

### Memory Testing
Monitor memory usage during long-running tests:
```python
import psutil
import os

def test_memory_usage():
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    # Perform operations...
    final_memory = process.memory_info().rss
    assert final_memory - initial_memory < 100_000_000  # 100MB limit
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **Ollama Connection**: Check if Ollama service is running
3. **Database Errors**: Verify ChromaDB path permissions
4. **Timeout Issues**: Increase timeout values for slow tests
5. **Mock Conflicts**: Check mock patch paths and decorators

### Debug Commands
```bash
# Verbose output with full tracebacks
pytest -v --tb=long

# Stop on first failure
pytest -x

# Run with Python debugger
pytest --pdb

# Print output (disable capture)
pytest -s
```

## Coverage Reports

Generate coverage reports to identify untested code:

```bash
# Terminal report
pytest --cov=src --cov-report=term-missing

# HTML report (creates htmlcov/ directory)
pytest --cov=src --cov-report=html

# XML report (for CI systems)
pytest --cov=src --cov-report=xml
```

Target coverage goals:
- Unit tests: 90%+ coverage
- Critical paths: 100% coverage
- Overall project: 80%+ coverage