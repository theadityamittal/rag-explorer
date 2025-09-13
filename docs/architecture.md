# Support Deflect Bot - Unified Dual Architecture Design

## Table of Contents
1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Architecture Layers](#architecture-layers)
4. [Directory Structure](#directory-structure)
5. [Core Engine Components](#core-engine-components)
6. [Interface Implementations](#interface-implementations)
7. [Data Flow Diagrams](#data-flow-diagrams)
8. [Deployment Models](#deployment-models)
9. [Configuration System](#configuration-system)
10. [Development Guidelines](#development-guidelines)

---

## Introduction

This document provides a comprehensive technical overview of the Support Deflect Bot's unified dual-architecture design. The system supports both CLI package distribution and API service deployment while maximizing code reuse through a shared engine layer.

### What This Bot Does (Enhanced)

Think of this bot as a smart librarian that:
1. **Reads your documentation** (local files and web content)
2. **Remembers everything** by creating a searchable vector index
3. **Answers questions** using advanced RAG (Retrieval-Augmented Generation)
4. **Admits uncertainty** instead of hallucinating responses
5. **Works everywhere** - as a CLI tool or web service

### Key Architectural Principles

- **Single Source of Truth**: 95% shared core functionality between CLI and API
- **Interface Separation**: Clean separation between user interfaces and business logic
- **Zero Breaking Changes**: 100% backward compatibility maintained
- **Deployment Flexibility**: Support for both pip package and containerized service
- **Provider Agnostic**: Multi-provider LLM system with intelligent fallbacks

### Key Technologies

- **RAG (Retrieval-Augmented Generation)**: Context-aware question answering
- **Vector Embeddings**: Semantic similarity search using ChromaDB
- **Multi-Provider LLM**: Unified interface for Google Gemini, OpenAI, Ollama, and more
- **FastAPI**: Modern async web framework for API deployment
- **Click**: Elegant command-line interface framework

---

## System Overview

### Unified Dual-Architecture Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────┐       ┌─────────────────────────────────┐    │
│  │      CLI INTERFACE        │       │        API INTERFACE            │    │
│  │                           │       │                                 │    │
│  │  • Command Parsing        │       │  • HTTP Request Handling        │    │
│  │  • Interactive Sessions   │       │  • JSON Schema Validation       │    │
│  │  • Terminal Formatting    │       │  • Authentication & Rate Limit  │    │
│  │  • Progress Indicators    │       │  • CORS & Security Headers      │    │
│  │  • Configuration Loading  │       │  • API Documentation (Swagger)  │    │
│  │  • Error Display          │       │  • Health Checks & Metrics      │    │
│  └───────────────────────────┘       └─────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                       SHARED BUSINESS LOGIC LAYER                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │   RAG ENGINE    │  │ DOCUMENT        │  │    QUERY PROCESSING         │  │
│  │                 │  │ PROCESSOR       │  │                             │  │
│  │ • Question      │  │                 │  │ • Query Preprocessing       │  │
│  │   Answering     │  │ • Local File    │  │ • Embedding Generation      │  │
│  │ • Confidence    │  │   Processing    │  │ • Vector Similarity Search  │  │
│  │   Scoring       │  │ • Web Content   │  │ • Result Ranking & Filter   │  │
│  │ • Context       │  │   Crawling      │  │ • Keyword Overlap Analysis  │  │
│  │   Assembly      │  │ • Text Chunking │  │ • Domain-based Filtering    │  │
│  │ • Citation      │  │ • Metadata      │  │ • Performance Optimization  │  │
│  │   Generation    │  │   Extraction    │  │ • Caching & Memoization     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                        PROVIDER ABSTRACTION LAYER                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ LLM PROVIDERS   │  │ EMBEDDING       │  │    PROVIDER MANAGEMENT      │  │
│  │                 │  │ PROVIDERS       │  │                             │  │
│  │ • Google Gemini │  │                 │  │ • Strategy Selection        │  │
│  │ • OpenAI GPT    │  │ • Google        │  │ • Health Monitoring         │  │
│  │ • Anthropic     │  │   Embedding     │  │ • Automatic Failover        │  │
│  │ • Groq          │  │ • OpenAI        │  │ • Cost Tracking             │  │
│  │ • Mistral       │  │   Embedding     │  │ • Rate Limit Management     │  │
│  │ • Ollama Local  │  │ • Ollama Local  │  │ • Performance Monitoring    │  │
│  │ • Claude Code   │  │   Embedding     │  │ • Configuration Validation  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                          DATA PERSISTENCE LAYER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ VECTOR DATABASE │  │ CONFIGURATION   │  │      CACHING LAYER          │  │
│  │                 │  │ STORAGE         │  │                             │  │
│  │ • ChromaDB      │  │                 │  │ • Query Result Caching      │  │
│  │   Collections   │  │ • Environment   │  │ • Embedding Caching         │  │
│  │ • Embedding     │  │   Variables     │  │ • Provider Response Cache   │  │
│  │   Vectors       │  │ • User Settings │  │ • Document Metadata Cache   │  │
│  │ • Document      │  │ • API Keys      │  │ • Performance Metrics       │  │
│  │   Metadata      │  │ • Default       │  │ • Error Rate Tracking       │  │
│  │ • Similarity    │  │   Configurations│  │ • Usage Statistics          │  │
│  │   Indices       │  │ • Validation    │  │ • Health Check Results      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
┌─────────────────────┐    ┌─────────────────────┐
│    User Request     │    │    User Request     │
│   (CLI Command)     │    │   (HTTP Request)    │
└──────────┬──────────┘    └──────────┬──────────┘
           │                          │
           ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐
│   CLI Handler       │    │   API Handler       │
│  • Parse Args       │    │  • Validate JSON    │
│  • Load Config      │    │  • Auth Check       │
│  • Format Output    │    │  • Rate Limiting    │
└──────────┬──────────┘    └──────────┬──────────┘
           │                          │
           └─────────┬────────────────┘
                     ▼
         ┌─────────────────────┐
         │   SHARED ENGINE     │
         │  • RAG Engine       │
         │  • Doc Processor    │
         │  • Query Service    │
         │  • Embedding Svc    │
         └──────────┬──────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
┌─────────────┐ ┌─────────┐ ┌─────────────┐
│ Provider    │ │Vector   │ │Configuration│
│ System      │ │Database │ │Management   │
│• Strategy   │ │• Chroma │ │• Settings   │
│• Fallback   │ │• Search │ │• Validation │
│• Health     │ │• Store  │ │• Defaults   │
└─────────────┘ └─────────┘ └─────────────┘
        │           │           │
        └───────────┼───────────┘
                    ▼
         ┌─────────────────────┐
         │    Response         │
         │  • Answer/Results   │
         │  • Confidence       │
         │  • Citations        │
         │  • Metadata         │
         └──────────┬──────────┘
                    │
           ┌────────┴────────┐
           ▼                 ▼
┌─────────────────────┐ ┌─────────────────────┐
│   CLI Response      │ │   API Response      │
│  • Terminal Output  │ │  • JSON Response    │
│  • Rich Formatting  │ │  • HTTP Headers     │
│  • Exit Codes       │ │  • Status Codes     │
└─────────────────────┘ └─────────────────────┘
```

---

## Architecture Layers

### 1. User Interface Layer

**Purpose**: Handle user interactions and present results

#### CLI Interface (`src/support_deflect_bot/cli/`)
- **Command Parsing**: Convert CLI commands to engine calls
- **Interactive Sessions**: Persistent Q&A conversations  
- **Terminal Formatting**: Rich output with colors and progress bars
- **Configuration Management**: Environment and settings handling

#### API Interface (`src/support_deflect_bot/api/`)
- **HTTP Request Handling**: RESTful endpoint implementation
- **JSON Schema Validation**: Request/response validation
- **Authentication**: API key and session management
- **Documentation**: Auto-generated OpenAPI/Swagger docs

### 2. Shared Business Logic Layer

**Purpose**: Core functionality used by both interfaces

#### RAG Engine (`src/support_deflect_bot/engine/rag_engine.py`)
- **Question Answering**: Main RAG pipeline implementation
- **Confidence Scoring**: Reliability measurement to prevent hallucinations
- **Context Assembly**: Intelligent combination of retrieved documents
- **Citation Generation**: Source attribution for transparency

#### Document Processor (`src/support_deflect_bot/engine/document_processor.py`)
- **Local File Processing**: Markdown, text, and structured document support
- **Web Content Crawling**: Intelligent web scraping with respect robots.txt
- **Text Chunking**: Optimal segmentation for embedding generation
- **Metadata Extraction**: File metadata and content structure analysis

#### Query Service (`src/support_deflect_bot/engine/query_service.py`)
- **Query Preprocessing**: Query optimization and normalization
- **Vector Similarity Search**: High-performance semantic search
- **Result Ranking**: Multi-factor result scoring and filtering
- **Performance Optimization**: Caching and memoization strategies

#### Embedding Service (`src/support_deflect_bot/engine/embedding_service.py`)
- **Multi-Provider Embedding**: Support for multiple embedding models
- **Batch Processing**: Efficient bulk embedding generation
- **Vector Dimension Management**: Consistent embedding dimensions
- **Embedding Caching**: Persistent embedding storage and reuse

### 3. Provider Abstraction Layer

**Purpose**: Unified interface for external services

#### Multi-Provider LLM System (`src/support_deflect_bot/core/providers/`)
- **Google Gemini**: Primary provider for cost-effective performance
- **Ollama**: Local inference for privacy and offline operation
- **OpenAI/Anthropic/Groq/Mistral**: Fallback providers for reliability
- **Strategy Selection**: Intelligent provider selection based on cost/performance
- **Health Monitoring**: Automatic provider availability checking
- **Fallback Chains**: Seamless provider switching on failures

### 4. Data Persistence Layer

**Purpose**: Data storage and retrieval

#### Vector Database (ChromaDB)
- **Document Collections**: Organized storage of embedded documents
- **Similarity Search**: Fast vector similarity operations
- **Metadata Storage**: Document metadata and indexing information
- **Persistent Storage**: Reliable data persistence across sessions

---

## Directory Structure

### Unified Package Structure (`src/support_deflect_bot/`)

```
src/support_deflect_bot/
├── __init__.py                           # Package initialization with version
├── engine/                               # 🆕 SHARED BUSINESS LOGIC LAYER
│   ├── __init__.py                       # Engine exports and initialization
│   ├── rag_engine.py                     # Main RAG pipeline (replaces src/core/rag.py)
│   ├── document_processor.py             # Document processing (enhances src/data/ingest.py)
│   ├── embedding_service.py              # Embedding generation (enhances src/data/embeddings.py)
│   └── query_service.py                  # Query processing (replaces src/core/retrieve.py)
├── cli/                                  # ✏️ ENHANCED CLI INTERFACE
│   ├── __init__.py                       # CLI exports
│   ├── main.py                           # Updated to use shared engine
│   ├── ask_session.py                    # Interactive Q&A sessions
│   ├── configure.py                      # Configuration management
│   ├── output.py                         # Terminal output formatting
│   └── commands/                         # 🆕 Modular command structure
│       ├── __init__.py
│       ├── ask_commands.py              # Question answering commands
│       ├── search_commands.py           # Search and retrieval commands
│       ├── index_commands.py            # Document indexing commands
│       ├── crawl_commands.py            # Web crawling commands
│       └── admin_commands.py            # Admin and health commands
├── api/                                  # 🆕 COMPREHENSIVE API INTERFACE
│   ├── __init__.py                       # API package exports
│   ├── app.py                           # FastAPI application with shared engine
│   ├── models/                          # Request/Response models
│   │   ├── __init__.py                  # Model exports
│   │   ├── requests.py                  # All request schemas
│   │   ├── responses.py                 # All response schemas
│   │   └── validators.py               # Custom validation logic
│   ├── endpoints/                       # Modular endpoint structure
│   │   ├── __init__.py                  # Endpoint exports
│   │   ├── query.py                     # /ask and /search endpoints
│   │   ├── indexing.py                  # /reindex and /crawl endpoints
│   │   ├── health.py                    # /healthz and /metrics endpoints
│   │   ├── admin.py                     # Administrative endpoints
│   │   └── batch.py                     # Batch processing endpoints
│   ├── middleware/                      # API middleware
│   │   ├── __init__.py
│   │   ├── cors.py                      # CORS configuration
│   │   ├── rate_limiting.py            # Rate limiting implementation
│   │   ├── authentication.py           # Authentication handling
│   │   ├── error_handling.py           # Global error handling
│   │   └── logging.py                  # Request/response logging
│   └── dependencies/                    # FastAPI dependencies
│       ├── __init__.py
│       ├── engine.py                   # Engine dependency injection
│       ├── validation.py              # Request validation
│       └── security.py                # Security dependencies
├── core/                                # ✅ EXISTING provider system
│   └── providers/                       # Multi-provider LLM system
│       ├── __init__.py                  # Provider exports
│       ├── base.py                      # Provider base classes
│       ├── config.py                    # Provider configuration
│       ├── strategies.py               # Selection strategies
│       └── implementations/             # Individual providers
│           ├── google_gemini.py         # Google Gemini (primary)
│           ├── ollama_provider.py       # Ollama local (fallback)
│           ├── openai_provider.py       # OpenAI fallback
│           ├── anthropic_provider.py    # Anthropic Claude fallback
│           ├── groq_provider.py         # Groq fallback
│           └── mistral_provider.py      # Mistral fallback
├── config/                              # Enhanced configuration
│   ├── __init__.py                      # Config exports
│   ├── manager.py                       # Enhanced configuration management
│   └── schema.py                        # Enhanced validation schemas
└── utils/                               # Enhanced utilities
    ├── __init__.py                      # Utils exports
    ├── settings.py                      # Enhanced with architecture settings
    ├── metrics.py                       # Enhanced performance monitoring
    ├── batch.py                         # Batch processing utilities
    ├── run_eval.py                      # Evaluation utilities
    ├── stderr_suppressor.py            # Output filtering
    └── warnings_suppressor.py          # Warning management

# 🗑️ PRESERVED SHARED MODULES (continue using from src/data/)
src/data/                               # KEEP - used by engine modules
├── store.py                            # ChromaDB operations
├── chunker.py                          # Text chunking algorithms
├── embeddings.py                       # Used by embedding_service.py
├── ingest.py                           # Used by document_processor.py
├── web_ingest.py                       # Used by document_processor.py
└── __init__.py                         # Data module exports
```

### Test Organization (`tests/`)

```
tests/
├── unit/                               # Component-level tests
│   ├── engine/                         # Engine component tests
│   ├── cli/                           # CLI interface tests
│   ├── api/                           # API interface tests
│   └── providers/                     # Provider system tests
├── integration/                        # Multi-component tests
│   ├── engine_integration/            # Engine service integration
│   ├── api_integration/               # API endpoint integration
│   └── provider_integration/          # Provider ecosystem tests
├── system/                            # End-to-end tests
│   ├── cli_e2e/                       # Full CLI workflows
│   ├── api_e2e/                       # Full API workflows
│   └── dual_deployment/               # Both CLI and API scenarios
└── conftest.py                        # Shared test fixtures
```

---

## Core Engine Components

### 1. Unified RAG Engine (`src/support_deflect_bot/engine/rag_engine.py`)

**Purpose**: Central RAG processing with confidence-based answering

```python
class UnifiedRAGEngine:
    """
    Main RAG engine used by both CLI and API interfaces
    """
    
    def __init__(self, config_manager, provider_system, query_service):
        self.config = config_manager
        self.providers = provider_system
        self.query_service = query_service
    
    async def answer_question(self, question: str, domains: Optional[List[str]] = None) -> Dict:
        """
        Main RAG pipeline:
        1. Generate question embedding
        2. Retrieve relevant documents
        3. Calculate confidence score
        4. Generate answer if confidence > threshold
        5. Return structured response with citations
        """
```

**Key Methods**:
- `answer_question()`: Main entry point for question answering
- `search_documents()`: Document retrieval without answer generation
- `calculate_confidence()`: Reliability scoring to prevent hallucinations
- `get_metrics()`: Performance and usage metrics
- `validate_providers()`: Health checking for provider system

### 2. Document Processor (`src/support_deflect_bot/engine/document_processor.py`)

**Purpose**: Unified document processing for local and web content

```python
class UnifiedDocumentProcessor:
    """
    Handles both local directory processing and web content crawling
    """
    
    def __init__(self, embedding_service, vector_store):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
    
    async def process_local_directory(self, directory_path: str) -> ProcessingResult:
        """
        Process local documents:
        1. Discover supported files
        2. Extract and clean content
        3. Chunk text optimally
        4. Generate embeddings
        5. Store in vector database
        """
    
    async def process_web_content(self, urls: List[str], crawl_config: CrawlConfig) -> ProcessingResult:
        """
        Process web content:
        1. Fetch web pages (respecting robots.txt)
        2. Extract clean text content
        3. Process similar to local files
        4. Handle dynamic content and JavaScript
        """
```

### 3. Query Service (`src/support_deflect_bot/engine/query_service.py`)

**Purpose**: Advanced query processing and document retrieval

```python
class UnifiedQueryService:
    """
    High-performance query processing with multiple ranking algorithms
    """
    
    def __init__(self, vector_store, embedding_service):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    async def retrieve_documents(self, query: str, filters: Optional[Dict] = None) -> List[Document]:
        """
        Multi-stage document retrieval:
        1. Query preprocessing and optimization
        2. Vector similarity search
        3. Keyword overlap analysis
        4. Domain-based filtering
        5. Result ranking and deduplication
        """
```

### 4. Embedding Service (`src/support_deflect_bot/engine/embedding_service.py`)

**Purpose**: Multi-provider embedding generation with caching

```python
class UnifiedEmbeddingService:
    """
    Handles embedding generation across multiple providers
    """
    
    def __init__(self, provider_system):
        self.provider_system = provider_system
        self.embedding_cache = {}
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings with caching:
        1. Check cache for existing embeddings
        2. Batch process new texts
        3. Use primary embedding provider
        4. Fallback to alternative providers
        5. Cache results for future use
        """
```

---

## Interface Implementations

### CLI Interface (`src/support_deflect_bot/cli/`)

**Core Philosophy**: Terminal-first experience with rich formatting

#### Key Commands
- `deflect-bot index <directory>`: Index local documentation
- `deflect-bot crawl <urls>`: Index web content
- `deflect-bot ask [question]`: Interactive or direct Q&A
- `deflect-bot search <query>`: Search without answer generation
- `deflect-bot ping`: Health check all providers
- `deflect-bot config`: Configuration management

#### Enhanced Features
- **Rich Terminal Output**: Colors, progress bars, and formatting
- **Interactive Sessions**: Persistent conversations with context
- **Configuration Wizard**: Guided setup for first-time users
- **Batch Operations**: Process multiple documents or queries
- **Debug Mode**: Detailed logging and performance metrics

### API Interface (`src/support_deflect_bot/api/`)

**Core Philosophy**: RESTful design with comprehensive OpenAPI documentation

#### Key Endpoints

**Query Operations**:
- `POST /ask`: Question answering with confidence scoring
- `POST /search`: Document search without answer generation
- `POST /batch/ask`: Batch question processing

**Indexing Operations**:
- `POST /reindex`: Index local directory
- `POST /crawl`: Crawl and index web content
- `GET /collections/{collection}/stats`: Collection statistics

**Health & Administration**:
- `GET /healthz`: System health check
- `GET /metrics`: Performance metrics
- `POST /admin/recompute-embeddings`: Recompute embeddings

#### Enhanced Features
- **Auto-generated Documentation**: OpenAPI/Swagger integration
- **Rate Limiting**: Configurable per-endpoint rate limits
- **Authentication**: API key and JWT token support
- **CORS Support**: Configurable cross-origin resource sharing
- **Request Validation**: Comprehensive input validation
- **Error Handling**: Structured error responses with details

---

## Data Flow Diagrams

### 1. Unified Document Indexing Flow

```
Document Sources (Local/Web)
    ↓
┌─────────────────────────────┐
│   Document Processor        │
│   • File Discovery          │
│   • Content Extraction      │
│   • Metadata Analysis       │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│   Text Chunking             │
│   • Semantic Boundaries     │
│   • Overlap Management      │
│   • Size Optimization       │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│   Embedding Service         │
│   • Provider Selection      │
│   • Batch Processing        │
│   • Cache Management        │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│   Vector Database           │
│   • ChromaDB Storage        │
│   • Metadata Indexing       │
│   • Collection Management   │
└─────────────────────────────┘
```

### 2. Unified Question Answering Flow

```
User Question (CLI/API)
    ↓
┌─────────────────────────────┐
│   Query Preprocessing       │
│   • Query Optimization      │
│   • Intent Detection        │
│   • Filter Application      │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│   Query Service             │
│   • Vector Search           │
│   • Keyword Matching        │
│   • Result Ranking          │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│   RAG Engine                │
│   • Confidence Calculation  │
│   • Context Assembly        │
│   • Citation Generation     │
└─────────────┬───────────────┘
              ↓
    ┌─────────────────┐
    │ Confidence      │
    │ Check           │
    └─────┬───────────┘
          │
    ┌─────▼──────┐         ┌──────────────┐
    │ >= 0.20?   │────NO──→│ "I don't     │
    │            │         │ have enough  │
    └─────┬──────┘         │ information" │
          │YES             └──────────────┘
          ▼
┌─────────────────────────────┐
│   LLM Provider System       │
│   • Provider Selection      │
│   • Fallback Management     │
│   • Response Generation     │
└─────────────┬───────────────┘
              ▼
    Response with Citations
```

### 3. Provider Selection and Fallback Flow

```
Request for LLM Service
    ↓
┌─────────────────────────────┐
│   Provider Strategy         │
│   • Cost Optimization       │
│   • Performance Priority    │
│   • Local-first Option      │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│   Primary Provider Check    │
│   • Google Gemini (default) │
│   • Health Validation       │
│   • Rate Limit Check        │
└─────────────┬───────────────┘
              ▼
    ┌─────────────────┐
    │ Available &     │
    │ Within Limits?  │
    └─────┬───────────┘
          │
    ┌─────▼──────┐         ┌──────────────────┐
    │    YES     │────────→│ Use Primary      │
    │            │         │ Provider         │
    └────────────┘         └──────────────────┘
          │NO
          ▼
┌─────────────────────────────┐
│   Fallback Chain            │
│   1. OpenAI GPT             │
│   2. Groq (Fast)            │
│   3. Ollama (Local)         │
│   4. Anthropic Claude       │
│   5. Mistral                │
└─────────────┬───────────────┘
              ▼
    Provider Response or Error
```

---

## Deployment Models

### 1. CLI Package Deployment (pip install)

**Use Case**: Developer tools, local documentation, offline usage

```bash
# Installation
pip install support-deflect-bot

# Usage
deflect-bot index ./docs
deflect-bot ask "How do I configure authentication?"
```

**Advantages**:
- **Local Processing**: No network dependency for core operations
- **Privacy**: Documents never leave the local system
- **Speed**: No API latency for document search
- **Offline Support**: Works with Ollama for complete offline operation

### 2. API Service Deployment (Docker/Kubernetes)

**Use Case**: Team documentation, web integrations, scalable deployments

```yaml
# docker-compose.yml
version: '3.8'
services:
  support-deflect-bot:
    image: support-deflect-bot:latest
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - DEPLOYMENT_MODE=api
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./docs:/app/docs
```

**Advantages**:
- **Scalability**: Horizontal scaling with load balancers
- **Integration**: RESTful API for web applications
- **Centralization**: Shared knowledge base for teams
- **Monitoring**: Comprehensive health checks and metrics

### 3. Hybrid Deployment

**Use Case**: Development teams needing both local and shared access

- **Local CLI**: Individual developer productivity
- **Shared API**: Team knowledge sharing and integration
- **Same Configuration**: Identical behavior across deployments
- **Synchronized Data**: Optional shared vector database

---

## Configuration System

### Environment Variables (Both CLI and API)

```bash
# Core RAG Configuration
ANSWER_MIN_CONF=0.20                    # Confidence threshold
MAX_CHUNKS=5                            # Max retrieval chunks
MAX_CHARS_PER_CHUNK=800                 # Chunk size limit
CHROMA_DB_PATH=./chroma_db               # Vector database location

# Primary LLM Provider (Gemini recommended)
GOOGLE_API_KEY=your_gemini_key
PRIMARY_LLM_PROVIDER=google_gemini_paid

# Fallback Providers
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
OLLAMA_HOST=http://localhost:11434      # Local Ollama
ANTHROPIC_API_KEY=your_claude_key
MISTRAL_API_KEY=your_mistral_key

# Provider Strategy
PROVIDER_STRATEGY=cost_optimized        # cost_optimized | performance | local_only
FALLBACK_LLM_PROVIDERS=openai,groq,ollama

# Deployment Mode
DEPLOYMENT_MODE=auto                    # auto | cli | api

# Web Crawling Configuration
CRAWL_DEPTH=1                           # Crawl depth limit
CRAWL_MAX_PAGES=40                      # Max pages per crawl
CRAWL_SAME_DOMAIN=true                  # Respect domain boundaries
DEFAULT_SEEDS=https://docs.python.org/3/faq/

# API-Specific Configuration (when DEPLOYMENT_MODE=api)
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your_api_key                    # Optional API authentication
CORS_ORIGINS=*                          # CORS configuration
RATE_LIMIT_PER_MINUTE=60               # Rate limiting
```

### Configuration Hierarchy

1. **Environment Variables** (highest priority)
2. **`.env` file** in working directory
3. **System `.env` file** in package directory
4. **Default values** in `src/support_deflect_bot/utils/settings.py`

### Deployment Mode Detection

```python
# Automatic deployment mode detection
def get_deployment_mode():
    """
    Determines deployment mode:
    1. Explicit DEPLOYMENT_MODE environment variable
    2. CLI usage detection (when imported via CLI entry point)
    3. API usage detection (when FastAPI app is imported)
    4. Default to CLI mode
    """
```

---

## Development Guidelines

### Adding New Features

#### 1. Engine Layer Features

**For shared functionality (used by both CLI and API):**

1. **Add to appropriate engine module**:
   ```python
   # src/support_deflect_bot/engine/rag_engine.py
   def new_rag_feature(self, parameters):
       """Implement new RAG functionality"""
   ```

2. **Update both interfaces**:
   ```python
   # CLI: src/support_deflect_bot/cli/commands/
   # API: src/support_deflect_bot/api/endpoints/
   ```

3. **Add comprehensive tests**:
   ```python
   # tests/unit/engine/test_rag_engine.py
   # tests/integration/engine_integration/
   ```

#### 2. Interface-Specific Features

**For CLI-only features:**
- Add to `src/support_deflect_bot/cli/commands/`
- Follow Click framework conventions
- Include help text and examples

**For API-only features:**
- Add to `src/support_deflect_bot/api/endpoints/`
- Include OpenAPI documentation
- Add request/response models

### Testing Strategy

#### 1. Unit Tests (`tests/unit/`)
- **Engine Components**: Test business logic in isolation
- **Interface Components**: Test UI logic separately
- **Provider System**: Mock external API calls

#### 2. Integration Tests (`tests/integration/`)
- **Engine Integration**: Test component interactions
- **Provider Integration**: Test with real APIs (rate-limited)
- **Database Integration**: Test with real ChromaDB

#### 3. System Tests (`tests/system/`)
- **CLI End-to-End**: Full workflows via CLI
- **API End-to-End**: Full workflows via HTTP requests
- **Dual Deployment**: Test both modes simultaneously

### Code Quality Standards

```bash
# Code formatting
black src tests
isort src tests

# Type checking
mypy src/support_deflect_bot/engine/
mypy src/support_deflect_bot/cli/
mypy src/support_deflect_bot/api/

# Linting
flake8 src tests

# Testing
pytest tests/unit/ -v --cov=src/support_deflect_bot/
pytest tests/integration/ -v --maxfail=3
pytest tests/system/ -v --maxfail=1
```

### Performance Optimization Guidelines

#### 1. Engine Layer Optimizations
- **Embedding Caching**: Cache embeddings to avoid recomputation
- **Query Optimization**: Use efficient vector search parameters
- **Batch Processing**: Process multiple items together when possible
- **Memory Management**: Clean up large objects after processing

#### 2. Interface Layer Optimizations
- **CLI**: Use streaming output for large results
- **API**: Implement response compression and caching headers
- **Both**: Lazy loading of large dependencies

### Contributing to the Project

#### Setup Development Environment

```bash
# 1. Clone and install in development mode
git clone https://github.com/theadityamittal/support-deflect-bot.git
cd support-deflect-bot
pip install -e ".[dev]"

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 3. Run tests to verify setup
pytest tests/unit/ -v

# 4. Test both CLI and API modes
deflect-bot --help                    # CLI mode
python -m support_deflect_bot.api.app  # API mode (http://localhost:8000/docs)
```

#### Development Workflow

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Understand architecture**: Start with engine layer for shared functionality
3. **Write tests first**: Add tests before implementation
4. **Implement changes**: Follow existing patterns and conventions
5. **Test both interfaces**: Ensure CLI and API work correctly
6. **Run quality checks**: All formatters, linters, and tests must pass
7. **Update documentation**: Include API docs and CLI help text

#### Architecture Decision Guidelines

- **Shared vs Interface-Specific**: Put business logic in engine layer
- **Provider Selection**: Use existing provider system for external APIs  
- **Configuration**: Use environment variables with sensible defaults
- **Error Handling**: Provide clear error messages for both CLI and API
- **Performance**: Consider both memory usage and response time
- **Backward Compatibility**: Maintain existing CLI command signatures

This unified architecture document provides comprehensive guidance for understanding, using, and contributing to the Support Deflect Bot's dual-architecture design!