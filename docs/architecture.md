# Support Deflect Bot - Architecture Design Document

## Table of Contents
1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Directory Structure](#directory-structure)
4. [Core Architecture Components](#core-architecture-components)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [Key Functions Reference](#key-functions-reference)
7. [Configuration System](#configuration-system)
8. [Testing Architecture](#testing-architecture)
9. [Development Guidelines](#development-guidelines)

---

## Introduction

This document provides a comprehensive technical overview of the Support Deflect Bot architecture, designed for developers who want to understand, contribute to, or debug the codebase. We'll explain complex concepts in simple terms and provide clear examples.

### What This Bot Does (In Simple Terms)

Think of this bot as a smart librarian that:
1. **Reads your documentation** (like a human would read books)
2. **Remembers everything** by creating a searchable index (like a card catalog)
3. **Answers questions** by finding relevant information in the index
4. **Admits when it doesn't know** instead of making things up

### Key Technologies Explained

- **RAG (Retrieval-Augmented Generation)**: A fancy way of saying "look up information first, then answer"
- **Embeddings**: Converting text into numbers so computers can understand similarity
- **Vector Database**: A special database that can find similar text quickly
- **LLM (Large Language Model)**: The AI that generates human-like responses

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SUPPORT DEFLECT BOT                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐                    ┌─────────────────────┐  │
│  │   CLI       │                    │    REST API         │  │
│  │ Interface   │                    │   (Optional)        │  │
│  └─────────────┘                    └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   CORE PROCESSING ENGINE                    │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Document   │  │     RAG      │  │  Multi-Provider │   │
│  │  Processing  │  │   Engine     │  │   LLM System    │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      DATA LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   ChromaDB   │  │  Embeddings  │  │   Configuration │   │
│  │ Vector Store │  │   Service    │  │    Manager      │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
User Input → CLI/API → RAG Engine → Vector Search → LLM Provider → Response
     ↑                                    ↓
     └── Configuration Manager ← Confidence Check ←─────────────┘
```

---

## Directory Structure

### Source Code Organization (`src/`)

```
src/
├── data/                        # Data processing and storage
│   ├── chunker.py               # Text chunking algorithms
│   ├── embeddings.py            # Embedding generation
│   ├── ingest.py                # Document ingestion pipeline
│   ├── store.py                 # ChromaDB vector store operations
│   └── web_ingest.py            # Web crawling and indexing
└── support_deflect_bot/         # Main application package
    ├── api/                     # REST API interface
    │   ├── dependencies/        # API dependency injection
    │   ├── endpoints/           # API route handlers
    │   ├── middleware/          # Request/response middleware
    │   ├── models/              # Request/response models
    │   └── app.py               # FastAPI application
    ├── cli/                     # Command-line interface
    │   ├── commands/            # CLI command implementations
    │   ├── main.py              # CLI entry point
    │   ├── ask_session.py       # Interactive Q&A session
    │   └── output.py            # Terminal output formatting
    ├── config/                  # Configuration system
    │   ├── manager.py           # Configuration management
    │   └── schema.py            # Configuration validation
    ├── core/                    # Core business logic
    │   └── providers/           # Multi-provider LLM system
    │       ├── base.py          # Provider base classes
    │       ├── config.py        # Provider configuration
    │       ├── strategies.py    # Selection strategies
    │       └── implementations/ # Individual provider implementations
    ├── engine/                  # Unified RAG engine
    │   ├── document_processor.py # Document processing pipeline
    │   ├── embedding_service.py # Embedding service abstraction
    │   ├── query_service.py     # Query processing service
    │   └── rag_engine.py        # Main RAG orchestration
    └── utils/                   # Utility modules
        ├── settings.py          # Application settings
        ├── metrics.py           # Performance monitoring
        └── batch.py             # Batch processing utilities
```

### Test Organization (`tests/`)

```
tests/
├── unit/                        # Component-level tests
├── integration/                 # API integration tests
├── providers/                   # Multi-provider testing
├── system/                      # End-to-end testing
└── conftest.py                  # Shared test fixtures
```

---

## Core Architecture Components

### 1. Document Processing Pipeline

#### File: `src/data/ingest.py`
**Purpose**: Convert documents into searchable chunks

```python
# Simplified flow:
def ingest_folder(folder_path):
    """
    1. Find all text files in folder
    2. Read and clean content
    3. Split into chunks
    4. Generate embeddings
    5. Store in vector database
    """
```

#### File: `src/data/chunker.py`
**Purpose**: Split documents into optimal sizes

**Why chunking matters**: 
- LLMs have input limits
- Smaller chunks = more precise retrieval
- Overlap ensures context isn't lost

```python
class TextChunker:
    def chunk_text(self, text, max_chars=800):
        """
        Splits text while preserving:
        - Sentence boundaries
        - Paragraph structure
        - Code blocks (if any)
        """
```

### 2. RAG (Retrieval-Augmented Generation) Engine

#### File: `src/support_deflect_bot/engine/rag_engine.py`
**Purpose**: The brain of the system - unified RAG orchestration

```python
class UnifiedRAGEngine:
    """
    The main RAG pipeline:
    
    1. Search for relevant chunks using embeddings
    2. Calculate confidence score
    3. If confidence < threshold: refuse to answer
    4. If confidence >= threshold: generate answer
    5. Return answer with citations
    6. Collect metrics and performance data
    """
```

#### Confidence Calculation
**Purpose**: Prevent hallucinations by measuring answer reliability

```python
def calculate_confidence(hits, question):
    """
    Combines multiple metrics:
    - Semantic similarity (from vector search): Primary factor
    - Keyword overlap (exact word matches): Secondary factor
    - Provider confidence scores: Tertiary factor
    
    Returns: 0.0 (no confidence) to 1.0 (high confidence)
    """
```

### 3. Multi-Provider LLM System

#### File: `src/support_deflect_bot/core/providers/base.py`
**Purpose**: Abstract interface for all LLM providers

```python
class BaseLLMProvider:
    """
    Every LLM provider must implement:
    - generate_response(): Get answer from LLM
    - calculate_cost(): Track usage costs
    - is_available(): Check if provider is working
    """
```

#### Provider Selection Strategy
**File**: `src/support_deflect_bot/core/providers/strategies.py`

```python
class ProviderStrategy:
    """
    Chooses best provider based on:
    - Cost optimization (cheapest first)
    - Performance (fastest first)
    - Local-first (privacy-focused)
    - Availability (fallback chains)
    """
```

### 4. Vector Database Operations

#### File: `src/data/store.py` (Lines 12-91)
**Purpose**: Manage ChromaDB for similarity search

```python
def query_by_embedding(query_embedding, k=5):
    """
    1. Take a question's embedding (list of numbers)
    2. Find k most similar document chunks
    3. Return chunks with similarity scores
    """
```

**Why ChromaDB**: 
- Optimized for similarity search
- Handles embedding operations efficiently
- Persistent storage with good performance

### 5. Configuration Management

#### File: `src/support_deflect_bot/config/manager.py`
**Purpose**: Centralized configuration handling

```python
class ConfigManager:
    """
    Manages:
    - Environment variables
    - Default settings
    - Provider API keys
    - Runtime configuration
    """
```

---

## Data Flow Diagrams

### 1. Document Indexing Flow

```
Documents → Read Files → Chunk Text → Generate Embeddings → Store in ChromaDB
    ↓
Local Files/Web Pages
    ↓
Text Extraction (BeautifulSoup for web)
    ↓
Intelligent Chunking (preserve context)
    ↓
Embedding Generation (via selected provider)
    ↓
Vector Storage with Metadata
```

### 2. Question Answering Flow

```
User Question
    ↓
Generate Question Embedding
    ↓
Vector Search in ChromaDB (find similar chunks)
    ↓
Calculate Confidence Score
    ↓
┌─────────────────────┐    ┌──────────────────────┐
│ Confidence < 0.20?  │───→│ Return "I don't      │
│                     │    │ have enough info"    │
└─────────────────────┘    └──────────────────────┘
    ↓ No
Context Formation (combine relevant chunks)
    ↓
LLM Provider Selection (cost/performance optimization)
    ↓
Generate Response with Citations
    ↓
Return Answer to User
```

### 3. Provider Selection Flow

```
User Query
    ↓
Check Available Providers
    ↓
Apply Selection Strategy:
┌─────────────┬─────────────┬─────────────┐
│ Cost        │ Performance │ Local       │
│ Optimized   │ Optimized   │ Only        │
└─────────────┴─────────────┴─────────────┘
    ↓
Primary Provider Available?
    ↓ No
Fallback Chain:
API Providers → Ollama (Local) → Error
```

---

## Key Functions Reference

### Core RAG Functions (`src/core/rag.py`)

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `answer_question()` | Main entry point for Q&A | Question string, optional domains | Answer dict with confidence |
| `_confidence()` | Calculate answer reliability | Document hits, question | Float 0.0-1.0 |
| `_keyword_overlap()` | Count matching words | Question, document text | Integer count |
| `_format_context()` | Prepare context for LLM | Document hits | Formatted string |

### Document Processing (`src/data/`)

| Function | File | Purpose |
|----------|------|---------|
| `ingest_folder()` | `ingest.py` | Process local documents |
| `chunk_text()` | `chunker.py` | Split text optimally |
| `generate_embeddings()` | `embeddings.py` | Convert text to vectors |
| `upsert_chunks()` | `store.py` | Save to database |

### CLI Commands (`src/support_deflect_bot/cli/main.py`)

| Command | Function | Purpose |
|---------|----------|---------|
| `deflect-bot index` | `index()` | Index local documentation |
| `deflect-bot ask` | `ask()` | Start interactive Q&A session |
| `deflect-bot search` | `search()` | Search indexed documents |
| `deflect-bot crawl` | `crawl()` | Crawl and index web pages |
| `deflect-bot status` | `status()` | Check system health |
| `deflect-bot ping` | `ping()` | Test LLM connectivity |
| `deflect-bot config` | `config()` | Display configuration |
| `deflect-bot metrics` | `metrics()` | Show performance metrics |

### Provider Management

| Function | File | Purpose |
|----------|------|---------|
| `select_provider()` | `strategies.py` | Choose best LLM provider |
| `calculate_cost()` | `base.py` | Track usage costs |
| `create_fallback_chain()` | `config.py` | Setup provider fallbacks |

---

## Configuration System

### Environment Variables

```bash
# Core Settings
ANSWER_MIN_CONF=0.20           # Confidence threshold for answers
MAX_CHUNKS=5                   # Max document chunks to retrieve
CHROMA_DB_PATH=./chroma_db     # Vector database location

# LLM Provider API Keys
OPENAI_API_KEY=your_key        # GPT models
GOOGLE_API_KEY=your_key        # Gemini models
ANTHROPIC_API_KEY=your_key     # Claude models
GROQ_API_KEY=your_key          # Fast inference
```

### Configuration Hierarchy

1. **Environment Variables** (highest priority)
2. **`.env` file** in project root
3. **Default values** in `settings.py`

### Settings File: `src/support_deflect_bot/utils/settings.py`

```python
# Key configuration constants
ANSWER_MIN_CONF = float(os.getenv("ANSWER_MIN_CONF", "0.20"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "5"))
MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "800"))

# Provider settings
DEFAULT_PROVIDER_STRATEGY = os.getenv("DEFAULT_PROVIDER_STRATEGY", "cost_optimized")
```

---

## Testing Architecture

### Test Categories

#### 1. Unit Tests (`tests/unit/`)
**Purpose**: Test individual components in isolation

```python
# Example: test_rag.py
def test_confidence_calculation():
    """Test confidence scoring algorithm"""
    
def test_keyword_overlap():
    """Test keyword matching logic"""
```

#### 2. Integration Tests (`tests/integration/`)
**Purpose**: Test component interactions with real APIs

```python
# Example: test_api.py
def test_real_gemini_api():
    """Test with actual Google Gemini API"""
```

#### 3. Provider Tests (`tests/providers/`)
**Purpose**: Validate multi-provider ecosystem

```python
# Example: test_provider_ecosystem.py
def test_provider_fallback_chain():
    """Test automatic provider switching"""
```

#### 4. System Tests (`tests/system/`)
**Purpose**: End-to-end workflow validation

```python
# Example: test_e2e.py  
def test_complete_indexing_and_qa_workflow():
    """Test full pipeline: index → ask → answer"""
```

### Test Fixtures (`tests/conftest.py`)

```python
@pytest.fixture
def sample_docs():
    """Provides test documents for indexing"""

@pytest.fixture  
def mock_chroma_collection():
    """Mocks ChromaDB for unit tests"""
```

### CI/CD Pipeline

The system uses GitHub Actions with 8 parallel job types:

1. **Code Quality**: Formatting, linting, type checking
2. **Package Build**: Wheel creation and CLI testing
3. **Unit Tests**: Multi-platform testing (Python 3.9-3.12)
4. **Integration Tests**: Real API testing (cost-controlled)
5. **Provider Tests**: Multi-provider validation
6. **System Tests**: E2E workflows
7. **Security Scan**: Vulnerability scanning
8. **PyPI Installation**: Distribution testing

---

## Development Guidelines

### Adding a New LLM Provider

1. **Create implementation file**:
   ```python
   # src/support_deflect_bot/core/providers/implementations/your_provider.py
   from ..base import BaseLLMProvider
   
   class YourProvider(BaseLLMProvider):
       def generate_response(self, prompt):
           # Implementation here
   ```

2. **Register in provider config**:
   ```python
   # src/support_deflect_bot/core/providers/config.py
   AVAILABLE_PROVIDERS["your_provider"] = YourProvider
   ```

3. **Add tests**:
   ```python
   # tests/providers/test_your_provider.py
   def test_your_provider_integration():
       # Test implementation
   ```

### Debugging Common Issues

#### 1. **"I don't have enough information" responses**
- **Cause**: Confidence score below threshold
- **Debug**: Check confidence calculation in `src/support_deflect_bot/engine/rag_engine.py`
- **Fix**: Adjust `ANSWER_MIN_CONF` or improve document indexing

#### 2. **ChromaDB connection errors**
- **Cause**: Database path or permissions issues
- **Debug**: Check `get_client()` in `src/data/store.py:12-19`
- **Fix**: Verify `CHROMA_DB_PATH` and folder permissions

#### 3. **Provider API failures**
- **Cause**: Invalid API keys or rate limits
- **Debug**: Check provider implementations in `src/support_deflect_bot/core/providers/implementations/`
- **Fix**: Verify API keys and check provider status

### Code Style Guidelines

- **Formatting**: Use Black (88-character line length)
- **Imports**: Use isort for consistent import ordering
- **Type Hints**: Add gradually (MyPy enabled but not strict)
- **Documentation**: Use docstrings for public functions
- **Testing**: Aim for >80% coverage on new code

### Performance Optimization Tips

1. **Embedding Caching**: Cache embeddings to avoid recomputation
2. **Batch Processing**: Process multiple documents together
3. **Lazy Loading**: Load providers only when needed
4. **Connection Pooling**: Reuse database connections

---

## Contributing to the Project

### Setup Development Environment

```bash
# 1. Clone and install in development mode
git clone https://github.com/theadityamittal/support-deflect-bot.git
cd support-deflect-bot
pip install -e ".[dev]"

# 2. Run tests to verify setup
pytest tests/unit/ -v

# 3. Check code quality
black --check src tests
isort --check-only src tests
flake8 src tests
```

### Understanding the Codebase

1. **Start with**: `src/support_deflect_bot/cli/main.py` to understand user interface
2. **Core logic**: `src/support_deflect_bot/engine/rag_engine.py` for the main RAG implementation
3. **Data flow**: `src/data/store.py` for database operations
4. **Configuration**: `src/support_deflect_bot/utils/settings.py` for all settings

### Making Changes

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Write tests first**: Add tests for new functionality
3. **Implement changes**: Follow existing patterns
4. **Run full test suite**: `pytest` before submitting
5. **Check code quality**: All formatters and linters must pass

This architecture document should help you understand how the Support Deflect Bot works internally and guide you in contributing effectively to the project!