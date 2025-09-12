# Test Suite - Unified Architecture

**Comprehensive test suite aligned with the Support Deflect Bot unified architecture.**

This test suite was completely rebuilt in 2025 to align with the new unified engine architecture, replacing the legacy test structure with modern testing practices and comprehensive coverage.

## 🏗️ Test Architecture

### **Test Structure**
```
tests/
├── conftest.py                    # Global fixtures for unified architecture
├── base.py                        # Base test classes and utilities
├── README.md                      # This documentation
│
├── unit/                          # Unit tests (isolated component testing)
│   ├── engine/                    # Test unified engine services
│   ├── providers/                 # Test provider ecosystem
│   ├── cli/                       # Test CLI interface
│   ├── api/                       # Test API interface
│   ├── config/                    # Test configuration system
│   ├── utils/                     # Test utility modules
│   └── data/                      # Test data processing
│
├── integration/                   # Integration tests (cross-component)
├── system/                        # System-level E2E tests
└── fixtures/                      # Shared test data and fixtures
    ├── sample_documents/
    ├── provider_responses/
    └── configuration/
```

## 🎯 Test Categories

### **Unit Tests** (`tests/unit/`)
- **Purpose**: Test individual components in complete isolation
- **Dependencies**: All external services mocked
- **Coverage Target**: 95%+ code coverage
- **Speed**: <1 second per test

**Key Areas:**
- **Engine Services**: `UnifiedRAGEngine`, `UnifiedQueryService`, `UnifiedDocumentProcessor`, `UnifiedEmbeddingService`
- **Providers**: All 8 AI provider implementations with fallback testing
- **CLI Commands**: All CLI functionality with rich output testing
- **API Endpoints**: All REST API endpoints with middleware testing

### **Integration Tests** (`tests/integration/`)
- **Purpose**: Test component interactions and workflows
- **Dependencies**: Real engine services, mocked external APIs
- **Coverage Target**: Critical user paths
- **Speed**: <10 seconds per test

**Key Areas:**
- CLI and API consistency (both using same engine services)
- Provider fallback chain testing
- End-to-end user workflows
- Cost optimization scenarios

### **System Tests** (`tests/system/`)
- **Purpose**: Complete end-to-end system validation
- **Dependencies**: Full system stack (with guarded real API calls)
- **Coverage Target**: Production readiness
- **Speed**: <60 seconds per test

**Key Areas:**
- Full deployment testing
- Performance benchmarking
- Architecture compliance validation

## 🛠️ Available Fixtures

### **Engine Service Fixtures**
- `mock_rag_engine`: Mock `UnifiedRAGEngine` with realistic responses
- `mock_document_processor`: Mock `UnifiedDocumentProcessor` 
- `mock_query_service`: Mock `UnifiedQueryService`
- `mock_embedding_service`: Mock `UnifiedEmbeddingService`

### **Provider Fixtures**
- `mock_openai_provider`: Mock OpenAI provider with GPT-4 responses
- `mock_groq_provider`: Mock Groq provider with Llama responses
- `mock_ollama_provider`: Mock Ollama provider with local model responses
- *(More providers available in `conftest.py`)*

### **Test Data Fixtures**
- `sample_documents`: Realistic documentation content
- `sample_embeddings`: Test embedding vectors
- `sample_chunks`: Document chunks with metadata
- `sample_api_responses`: Mock API responses from providers

### **Utility Fixtures**
- `temp_dir`: Temporary directory for file operations
- `temp_chroma_db`: Temporary ChromaDB instance
- `test_settings`: Test configuration settings
- `cli_runner`: Click CLI test runner

## 🚀 Running Tests

### **Quick Commands**
```bash
# Run all tests
pytest

# Run specific categories
pytest -m unit                    # Unit tests only
pytest -m integration             # Integration tests only
pytest -m "unit and engine"       # Engine unit tests only

# Run specific components
pytest tests/unit/engine/         # All engine tests
pytest tests/unit/providers/      # All provider tests
pytest tests/integration/         # All integration tests

# Run with coverage
pytest --cov=src/support_deflect_bot --cov-report=html

# Skip slow/expensive tests
pytest -m "not slow and not requires_api"
```

### **Test Markers**
```bash
# By component
-m unit                # Unit tests for isolated components
-m integration         # Integration tests for component interaction
-m system             # System-level end-to-end tests

# By requirement
-m slow               # Slow running tests (>10 seconds)
-m requires_api       # Tests requiring real API access (guarded)
-m requires_ollama    # Tests requiring Ollama service
-m cost_sensitive     # Tests that may incur API costs

# By architecture area  
-m engine             # Tests for unified engine services
-m providers          # Tests for AI provider implementations
-m cli                # Tests for CLI interface
-m api                # Tests for REST API interface
```

## 📝 Writing New Tests

### **Unit Test Example**
```python
import pytest
from tests.base import BaseEngineTest

class TestUnifiedRAGEngine(BaseEngineTest):
    """Test the UnifiedRAGEngine service."""
    
    @pytest.mark.unit
    @pytest.mark.engine
    async def test_query_documents_success(self, mock_rag_engine):
        """Test successful document querying."""
        # Arrange
        query = "How do I install the bot?"
        
        # Act
        result = await mock_rag_engine.query_documents(query)
        
        # Assert
        assert result["confidence"] > 0.8
        assert "install" in result["answer"].lower()
        mock_rag_engine.query_documents.assert_called_once_with(query)
```

### **Integration Test Example**
```python
import pytest
from tests.base import BaseIntegrationTest

class TestCLIAPIConsistency(BaseIntegrationTest):
    """Test CLI and API use same engine services."""
    
    @pytest.mark.integration
    async def test_cli_api_same_results(self, mock_rag_engine):
        """Test CLI and API return consistent results."""
        # Test implementation here
        pass
```

### **Provider Test Example**
```python
import pytest
from tests.base import BaseProviderTest

class TestOpenAIProvider(BaseProviderTest):
    """Test OpenAI provider implementation."""
    
    @pytest.mark.unit
    @pytest.mark.providers
    @pytest.mark.cost_sensitive
    async def test_generate_response(self, mock_openai_provider):
        """Test OpenAI response generation."""
        # Arrange
        query = "Test query"
        
        # Act
        response = await mock_openai_provider.generate_response(query)
        
        # Assert
        assert response["content"] is not None
        assert response["model"].startswith("gpt")
        self.assert_provider_response(mock_openai_provider, query, response["content"])
```

## 🔧 Base Classes

### **BaseEngineTest**
Use for testing unified engine services:
- `create_mock_provider()`: Create consistent provider mocks
- `assert_async_called_with()`: Assert async method calls

### **BaseProviderTest**
Use for testing AI provider implementations:
- `create_mock_response()`: Create realistic API responses
- `assert_provider_available()`: Test provider availability
- `assert_provider_response()`: Test provider responses

### **BaseAPITest**
Use for testing REST API endpoints:
- `assert_status_code()`: Assert HTTP status codes
- `assert_json_response()`: Assert JSON response structure

### **BaseCLITest**
Use for testing CLI commands:
- `assert_cli_success()`: Assert successful command execution
- `assert_cli_error()`: Assert command failures

## 📊 Test Utilities

### **TestUtils**
Static utility methods:
- `create_test_document()`: Generate test documents
- `create_test_embedding()`: Generate test embeddings
- `create_test_chunk()`: Generate document chunks

### **MockFactory**
Consistent mock creation:
- `create_engine_service_mock()`: Mock any engine service
- `create_provider_mock()`: Mock any AI provider

### **TestDataBuilder**
Fluent test data creation:
```python
test_data = (TestDataBuilder()
    .with_documents(3)
    .with_chunks(5)
    .with_embeddings(5)
    .build())
```

## 🔍 Best Practices

### **1. Test Isolation**
- Each test should be completely independent
- Use fixtures for setup/teardown
- Mock all external dependencies (APIs, databases)

### **2. Realistic Mocks**
- Use provided fixtures that simulate real behavior
- Mock at system boundaries, not internal methods
- Test both success and failure scenarios

### **3. Clear Test Names**
- Use descriptive names: `test_method_scenario_expected_outcome`
- Group related tests in classes: `TestUnifiedRAGEngine`
- Use markers to categorize tests

### **4. Coverage Goals**
- **Unit Tests**: 95%+ coverage of all components
- **Integration Tests**: 100% coverage of critical user paths
- **System Tests**: 100% coverage of deployment scenarios

### **5. Performance Testing**
- Mark slow tests with `@pytest.mark.slow`
- Monitor test execution time
- Use `@pytest.mark.cost_sensitive` for tests that may incur API costs

## 🛡️ Cost Control

### **API Cost Management**
- All provider tests use mocks by default
- Real API tests are marked with `@pytest.mark.requires_api`
- Real API tests only run in CI with budget controls
- Use `@pytest.mark.cost_sensitive` for cost-aware testing

### **Running Cost-Sensitive Tests**
```bash
# Skip all cost-sensitive tests (default)
pytest -m "not cost_sensitive"

# Run cost-sensitive tests (CI only)
pytest -m cost_sensitive

# Run with API budget guard
MONTHLY_BUDGET_USD=2.0 pytest -m requires_api
```

## 📈 Coverage Reports

Generate comprehensive coverage reports:

```bash
# Terminal report with missing lines
pytest --cov=src --cov-report=term-missing

# HTML report (creates htmlcov/ directory)  
pytest --cov=src --cov-report=html

# XML report for CI systems
pytest --cov=src --cov-report=xml
```

## 🔧 Debugging Tests

### **Common Debug Commands**
```bash
# Verbose output with full tracebacks
pytest -vv --tb=long

# Stop on first failure
pytest -x

# Run with Python debugger
pytest --pdb

# Print all output (disable capture)
pytest -s

# Run specific failing test
pytest tests/unit/engine/test_rag_engine.py::TestUnifiedRAGEngine::test_specific_method -v
```

## 🏁 Migration from Legacy Tests

This test suite completely replaces the previous test structure. Key improvements:

### **Architecture Alignment**
- ✅ Tests mirror unified engine architecture exactly
- ✅ Separate testing for CLI and API interfaces using shared services
- ✅ Comprehensive provider ecosystem testing

### **Modern Testing Practices**
- ✅ Comprehensive fixture system for consistent mocking
- ✅ Base classes for common testing patterns  
- ✅ Clear test categorization with markers
- ✅ Cost-aware testing with budget controls

### **Enhanced Coverage**
- ✅ 4 unified engine services fully tested
- ✅ 8 AI provider implementations tested
- ✅ Complete CLI and API interface coverage
- ✅ Integration and system-level testing

---

*This test suite was rebuilt in 2025 to align with the unified architecture. All tests follow modern pytest practices with comprehensive mocking and realistic fixtures.*