# Personal RAG Explorer: Simplification Plan

**Date:** January 2025  
**Purpose:** Transform RAG Explorer into a simple, personal RAG exploration tool  
**Target:** Pip-installable CLI package for local RAG experimentation

---

## 🎯 **Simplification Goals**

### **What to Keep:**
1. **ChromaDB** - Perfect for local experimentation
2. **Ollama integration** - Privacy + cost-free local models
3. **3 API providers** - OpenAI, Anthropic, Google Gemini
4. **Core RAG pipeline** - Document ingestion, chunking, embedding, retrieval, generation
5. **CLI interface** - Simple commands for indexing and querying
6. **Confidence scoring** - Unique feature worth preserving

### **What to Remove:**
1. **Entire API implementation** - All FastAPI, middleware, endpoints
2. **Enterprise patterns** - Circuit breakers, retry policies, resilience
3. **Complex configuration** - Multiple config sources, validation schemas
4. **Unused providers** - Groq, Mistral, Claude Code, etc.
5. **Advanced features** - Metrics collection, health checks, monitoring
6. **Web crawling** - Keep it simple, local files only

---

## 📋 **Implementation Roadmap**

### **Phase 1: Remove API Implementation (Day 1)**
- [ ] Delete entire `src/rag_explorer/api/` directory
- [ ] Remove FastAPI dependencies from pyproject.toml
- [ ] Remove API-related imports and references
- [ ] Clean up unused middleware and models

### **Phase 2: Simplify Provider System (Day 1-2)**
- [ ] Keep only 4 providers: Ollama, OpenAI, Anthropic, Google
- [ ] Remove provider registry complexity
- [ ] Simplify provider selection to basic fallback
- [ ] Remove provider strategies and complex configurations

### **Phase 3: Remove Enterprise Patterns (Day 2)**
- [ ] Remove circuit breakers from core/resilience.py
- [ ] Remove retry policies and complex error handling
- [ ] Simplify RAG engine by removing resilience patterns
- [ ] Keep basic error handling only

### **Phase 4: Simplify Configuration (Day 2-3)**
- [ ] Replace complex config system with simple .env loading
- [ ] Remove config/manager.py and config/schema.py
- [ ] Create simple settings.py with environment variables
- [ ] Update .env.example with minimal configuration

### **Phase 5: Streamline CLI (Day 3)**
- [ ] Simplify CLI commands to: index, ask, status
- [ ] Remove admin, metrics, ping, configure commands
- [ ] Simplify command implementations
- [ ] Remove complex CLI features

### **Phase 6: Clean Dependencies (Day 3-4)**
- [ ] Update pyproject.toml with minimal dependencies
- [ ] Remove unused imports throughout codebase
- [ ] Clean up requirements.txt
- [ ] Test package installation

### **Phase 7: Update Documentation (Day 4)**
- [ ] Rewrite README.md for personal use case
- [ ] Create simple setup instructions
- [ ] Document the 4 supported providers
- [ ] Add basic usage examples

### **Phase 8: Testing & Validation (Day 4-5)**
- [ ] Test pip installation process
- [ ] Verify all 4 providers work
- [ ] Test basic RAG pipeline
- [ ] Ensure ChromaDB integration works

---

## 🏗️ **Simplified Architecture**

### **New Directory Structure:**
```
src/rag_explorer/
├── __init__.py
├── _version.py
├── cli/
│   ├── __init__.py
│   └── main.py              # Simple CLI with 3 commands
├── core/
│   ├── __init__.py
│   ├── providers.py         # 4 providers only
│   └── rag_engine.py        # Simplified RAG pipeline
├── data/
│   ├── __init__.py
│   ├── chunker.py           # Keep existing simple chunker
│   ├── ingest.py            # Local file ingestion only
│   └── store.py             # ChromaDB integration
└── utils/
    ├── __init__.py
    └── settings.py          # Simple .env loading
```

### **Simplified Dependencies:**
```toml
dependencies = [
    "click>=8.1.0",           # CLI interface
    "rich>=13.0.0",           # Pretty output
    "chromadb==0.5.5",        # Vector database
    "python-dotenv==1.0.1",   # Environment variables
    "beautifulsoup4>=4.12.3", # HTML parsing
    "lxml>=5.2.2",            # XML parsing
    "openai>=1.0.0",          # OpenAI API
    "anthropic>=0.25.0",      # Claude API
    "google-generativeai>=0.3.0", # Gemini API
    "ollama>=0.3.0",          # Local Ollama
]
```

---

## 🔧 **Simplified Configuration**

### **New .env.example:**
```bash
# Primary provider (ollama, openai, anthropic, google)
PRIMARY_LLM_PROVIDER=ollama
PRIMARY_EMBEDDING_PROVIDER=ollama

# API Keys (optional - leave empty to use only Ollama)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

# Ollama settings (for local use)
OLLAMA_HOST=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# RAG settings
CHUNK_SIZE=800
CHUNK_OVERLAP=100
MIN_CONFIDENCE=0.25
MAX_CHUNKS=5

# Data directory
DOCS_FOLDER=./docs
CHROMA_DB_PATH=./chroma_db
```

### **Simple Settings Loading:**
```python
# utils/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# Provider settings
PRIMARY_LLM_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER", "ollama")
PRIMARY_EMBEDDING_PROVIDER = os.getenv("PRIMARY_EMBEDDING_PROVIDER", "ollama")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ollama settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# RAG settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.25"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "5"))

# Paths
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "./docs")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
```

---

## 🎮 **Simplified CLI Interface**

### **Three Simple Commands:**
```bash
# Index documents
rag-explorer index ./my-docs

# Ask questions
rag-explorer ask "How do I configure X?"

# Check status
rag-explorer status
```

### **CLI Implementation:**
```python
# cli/main.py
import click
from rich.console import Console
from ..core.rag_engine import SimpleRAGEngine

console = Console()
engine = SimpleRAGEngine()

@click.group()
def cli():
    """RAG Explorer - Simple document Q&A tool"""
    pass

@cli.command()
@click.argument('docs_path')
def index(docs_path):
    """Index documents from a directory"""
    console.print(f"[blue]Indexing documents from {docs_path}...[/blue]")
    result = engine.index_documents(docs_path)
    console.print(f"[green]Indexed {result['count']} documents[/green]")

@cli.command()
@click.argument('question')
def ask(question):
    """Ask a question about your documents"""
    console.print(f"[blue]Question:[/blue] {question}")
    result = engine.answer_question(question)
    
    console.print(f"[green]Answer:[/green] {result['answer']}")
    console.print(f"[dim]Confidence: {result['confidence']:.2f}[/dim]")

@cli.command()
def status():
    """Show system status"""
    status = engine.get_status()
    console.print(f"[blue]Provider:[/blue] {status['provider']}")
    console.print(f"[blue]Documents:[/blue] {status['doc_count']}")
    console.print(f"[blue]Database:[/blue] {status['db_status']}")
```

---

## 🧠 **Simplified RAG Engine**

### **Core Features Only:**
```python
# core/rag_engine.py
class SimpleRAGEngine:
    def __init__(self):
        self.providers = self._init_providers()
        self.store = ChromaDBStore()
    
    def _init_providers(self):
        """Initialize the 4 supported providers"""
        providers = {}
        
        # Always try to initialize Ollama (local)
        try:
            providers['ollama'] = OllamaProvider()
        except:
            pass
            
        # Initialize API providers if keys are available
        if OPENAI_API_KEY:
            providers['openai'] = OpenAIProvider()
        if ANTHROPIC_API_KEY:
            providers['anthropic'] = AnthropicProvider()
        if GOOGLE_API_KEY:
            providers['google'] = GoogleProvider()
            
        return providers
    
    def index_documents(self, docs_path):
        """Simple document indexing"""
        # Load documents
        documents = self._load_documents(docs_path)
        
        # Chunk documents
        chunks = self._chunk_documents(documents)
        
        # Generate embeddings
        embeddings = self._generate_embeddings(chunks)
        
        # Store in ChromaDB
        self.store.add_documents(chunks, embeddings)
        
        return {"count": len(chunks)}
    
    def answer_question(self, question):
        """Simple question answering"""
        # Generate query embedding
        query_embedding = self._generate_query_embedding(question)
        
        # Retrieve relevant chunks
        chunks = self.store.search(query_embedding, k=MAX_CHUNKS)
        
        # Calculate confidence
        confidence = self._calculate_confidence(chunks, question)
        
        # Generate answer if confidence is high enough
        if confidence >= MIN_CONFIDENCE:
            context = self._format_context(chunks)
            answer = self._generate_answer(question, context)
        else:
            answer = "I don't have enough information to answer that question."
        
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": [chunk["source"] for chunk in chunks[:3]]
        }
    
    def _get_provider(self, provider_type):
        """Get provider with simple fallback"""
        if provider_type == "llm":
            preferred = PRIMARY_LLM_PROVIDER
        else:
            preferred = PRIMARY_EMBEDDING_PROVIDER
            
        # Try preferred provider first
        if preferred in self.providers:
            return self.providers[preferred]
            
        # Fallback to any available provider
        for provider in self.providers.values():
            if provider.supports(provider_type):
                return provider
                
        raise Exception(f"No {provider_type} provider available")
```

---

## 📦 **Updated Package Configuration**

### **New pyproject.toml:**
```toml
[project]
name = "rag-explorer"
dynamic = ["version"]
authors = [{name = "Your Name", email = "your.email@example.com"}]
description = "Simple personal RAG exploration tool"
readme = "README.md"
requires-python = ">=3.9"
license = "MIT"

dependencies = [
    "click>=8.1.0",
    "rich>=13.0.0",
    "chromadb==0.5.5",
    "python-dotenv==1.0.1",
    "beautifulsoup4>=4.12.3",
    "lxml>=5.2.2",
    "openai>=1.0.0",
    "anthropic>=0.25.0",
    "google-generativeai>=0.3.0",
    "ollama>=0.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
]

[project.scripts]
rag-explorer = "rag_explorer.cli.main:cli"
```

---

## 🚀 **Installation & Usage**

### **Installation:**
```bash
# Install from PyPI (after publishing)
pip install rag-explorer

# Or install locally for development
git clone <your-repo>
cd rag-explorer
pip install -e .
```

### **Quick Start:**
```bash
# 1. Set up Ollama (for local use)
ollama pull llama3.1
ollama pull nomic-embed-text

# 2. Index your documents
rag-explorer index ./my-documents

# 3. Ask questions
rag-explorer ask "How do I configure authentication?"

# 4. Check status
rag-explorer status
```

### **With API Providers:**
```bash
# Set up API keys
export OPENAI_API_KEY="your-key"
export PRIMARY_LLM_PROVIDER="openai"

# Use as normal
rag-explorer ask "How do I deploy to production?"
```

---

## ✅ **Success Criteria**

### **Functional Requirements:**
- [ ] Pip installable package
- [ ] Works with Ollama (local, no API keys needed)
- [ ] Works with OpenAI, Anthropic, Google (when API keys provided)
- [ ] Simple 3-command CLI interface
- [ ] ChromaDB for local vector storage
- [ ] Basic RAG pipeline (index, retrieve, generate)
- [ ] Confidence-based answer filtering

### **Non-Functional Requirements:**
- [ ] <10 dependencies in total
- [ ] <5 minute setup time
- [ ] <100 lines of configuration
- [ ] Works offline with Ollama
- [ ] Clear error messages
- [ ] Simple documentation

### **Quality Gates:**
- [ ] All tests pass
- [ ] Package installs cleanly
- [ ] CLI commands work as expected
- [ ] Documentation is clear and complete
- [ ] No enterprise complexity remains

---

## 🎯 **Final Architecture**

This simplified version will be:
- **~80% smaller** codebase
- **~90% fewer dependencies**
- **100% focused** on personal RAG exploration
- **Zero enterprise complexity**
- **Maximum learning value**

Perfect for understanding RAG fundamentals and experimenting with different providers and configurations!

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Status:** Ready for Implementation
