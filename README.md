# RAG Explorer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Simple personal RAG exploration tool** - Transform your documents into an intelligent Q&A system with multiple AI provider support.

## 🎯 Project Description

RAG Explorer is a Retrieval-Augmented Generation (RAG) tool designed for personal document analysis and Q&A. It combines document indexing, vector search, and multiple AI providers to create an intelligent knowledge base from your documents.

### ✨ Key Features

- **Multiple AI Providers**: Support for OpenAI GPT, Google Gemini, Anthropic Claude, and local Ollama models
- **Smart Document Processing**: Automatic chunking and embedding of various document formats
- **Confidence Scoring**: 4-factor algorithm (similarity, count, keywords, content length) for result quality
- **Rich CLI Interface**: Beautiful terminal interface with progress indicators and formatted output
- **Local & Cloud Options**: Use local Ollama models for privacy or cloud APIs for performance
- **Vector Search**: ChromaDB-powered semantic search with configurable parameters
- **Website Crawling**: Index content directly from websites
- **Interactive Configuration**: Built-in configuration management

### 🎯 Use Cases

- **Personal Knowledge Base**: Index your notes, documents, and research materials
- **Document Analysis**: Ask questions about large document collections
- **Research Assistant**: Quick retrieval of relevant information from your data
- **Privacy-Focused**: Option to run entirely locally with Ollama

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- Git

### Quick Install

```bash
# Clone the repository
git clone https://github.com/theadityamittal/rag-explorer.git
cd rag-explorer

# Install in development mode
pip install -e .

# Verify installation
rag-explorer --help
```

### Recommended: Virtual Environment

```bash
# Create virtual environment
python -m venv rag-env
source rag-env/bin/activate  # On Windows: rag-env\Scripts\activate

# Install
pip install -e .
```

## ⚙️ Configuration

### 1. Environment Setup

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

### 2. Complete .env Configuration

```bash
# Primary provider (choose your preferred AI provider)
PRIMARY_LLM_PROVIDER=ollama             # ollama, openai, anthropic, google
PRIMARY_EMBEDDING_PROVIDER=ollama       # ollama, openai, google

# Ollama settings (for local use)
OLLAMA_HOST=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# API Keys (required for cloud providers)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_google_gemini_api_key_here

# Model settings
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
ANTHROPIC_LLM_MODEL=claude-3-7-sonnet-20250219
GEMINI_LLM_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# RAG settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
MIN_CONFIDENCE=0.25
MAX_CHUNKS=5

# Paths
DOCS_FOLDER=./docs
CHROMA_DB_PATH=./chroma_db

# Crawl settings (optional)
CRAWL_SOURCES=""
CRAWL_DEPTH=2
CRAWL_MAX_PAGES=50
```

### 3. Provider-Specific Setup

#### 🤖 Ollama (Local, Privacy-Focused)

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.1
ollama pull nomic-embed-text

# Verify Ollama is running
ollama list
```

#### 🔑 OpenAI

1. Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Add to your `.env` file:
   ```bash
   OPENAI_API_KEY=sk-...
   PRIMARY_LLM_PROVIDER=openai
   PRIMARY_EMBEDDING_PROVIDER=openai
   ```

#### 🧠 Google Gemini

1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Add to your `.env` file:
   ```bash
   GEMINI_API_KEY=AI...
   PRIMARY_LLM_PROVIDER=google
   PRIMARY_EMBEDDING_PROVIDER=google
   ```

#### 🎭 Anthropic Claude

1. Get API key from [Anthropic Console](https://console.anthropic.com/)
2. Add to your `.env` file:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-...
   PRIMARY_LLM_PROVIDER=anthropic
   # Note: Anthropic doesn't provide embeddings, use openai or google for embeddings
   PRIMARY_EMBEDDING_PROVIDER=openai
   ```

### 4. Interactive Configuration

Use the built-in configuration tool:

```bash
rag-explorer configure
```

## 📖 Usage

### Quick Start

1. **Check system status**:
   ```bash
   rag-explorer status
   ```

2. **Test provider connectivity**:
   ```bash
   rag-explorer ping
   ```

3. **Index your documents**:
   ```bash
   rag-explorer index ./your-documents
   ```

4. **Ask questions**:
   ```bash
   rag-explorer ask "What is the main topic of these documents?"
   ```

### Core Commands

#### 📂 Document Indexing
```bash
# Index documents from a directory
rag-explorer index ./docs

# The system will automatically:
# - Process text files (.txt, .md)
# - Extract text from PDFs
# - Chunk documents optimally
# - Generate embeddings
# - Store in vector database
```

#### ❓ Question Answering
```bash
# Ask questions about your indexed documents
rag-explorer ask "What are the key findings?"
rag-explorer ask "Summarize the methodology"

# The system provides:
# - Relevant document excerpts
# - Confidence scores
# - Source references
```

#### 🔍 Document Search
```bash
# Search for specific topics
rag-explorer search "machine learning"
rag-explorer search "project timeline" --count 10

# Returns ranked results with:
# - Similarity scores
# - Source documents
# - Relevant snippets
```

#### 🌐 Website Crawling
```bash
# Add websites to crawl sources in .env
CRAWL_SOURCES=https://example.com,https://blog.example.com

# Crawl and index websites
rag-explorer crawl
```

#### 📊 System Management
```bash
# Check system status and configuration
rag-explorer status

# Test AI provider connectivity
rag-explorer ping

# View storage metrics and database info
rag-explorer metrics

# Interactive configuration editor
rag-explorer configure
```

### Example Workflow

```bash
# 1. Set up your environment
cp .env.example .env
# Edit .env with your API keys

# 2. Test connectivity
rag-explorer ping

# 3. Index your documents
rag-explorer index ./my-research-papers

# 4. Check what was indexed
rag-explorer metrics

# 5. Start asking questions
rag-explorer ask "What are the main conclusions?"
rag-explorer ask "What methodology was used?"

# 6. Search for specific topics
rag-explorer search "neural networks"
```

## 🏗️ Architecture

### High-Level Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │    │   AI Providers  │    │  Vector Store   │
│  (.txt, .pdf)   │────│ OpenAI/Gemini   │────│   ChromaDB      │
│                 │    │ Claude/Ollama   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  RAG Explorer   │
                    │  CLI Interface  │
                    └─────────────────┘
```

### Core Components

#### 🧠 UnifiedRAGEngine
- **Purpose**: Central orchestrator for the RAG pipeline
- **Functions**: Question answering, document search, confidence calculation
- **Features**: 4-factor confidence scoring, configurable result filtering

#### 🔌 Provider System
- **OpenAI Provider**: GPT models + text-embedding-3
- **Google Provider**: Gemini models + embedding-001
- **Anthropic Provider**: Claude models (LLM only)
- **Ollama Provider**: Local models (llama3.1 + nomic-embed-text)

#### 📄 Document Processing Pipeline
- **Text Extraction**: Support for .txt, .md, .pdf files
- **Smart Chunking**: Configurable chunk size with overlap
- **Embedding Generation**: Provider-specific embedding creation
- **Metadata Handling**: Source tracking and document organization

#### 🗄️ Vector Database (ChromaDB)
- **Storage**: Persistent local vector storage
- **Search**: Semantic similarity search
- **Management**: Collection management and indexing

#### 🖥️ CLI Interface (Click + Rich)
- **Commands**: 8 core commands for all operations
- **UI**: Rich formatting with progress bars and tables
- **Configuration**: Interactive environment management

### Confidence Scoring Algorithm

The system uses a 4-factor confidence score to rank results:

1. **Similarity Score** (40%): Vector similarity between query and document
2. **Result Count** (25%): Number of relevant chunks found
3. **Keyword Overlap** (20%): Direct keyword matches
4. **Content Length** (15%): Comprehensive content indicator

```python
confidence = (
    similarity_score * 0.4 +
    count_factor * 0.25 +
    keyword_factor * 0.20 +
    length_factor * 0.15
)
```

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.9+**: Primary language
- **Click**: CLI framework for command-line interface
- **Rich**: Terminal formatting and progress indicators
- **ChromaDB**: Vector database for embeddings storage
- **python-dotenv**: Environment variable management

### AI Providers
- **OpenAI**: `openai>=1.0.0` - GPT models and text embeddings
- **Google Gemini**: `google-generativeai>=0.3.0` - Gemini models
- **Anthropic**: `anthropic>=0.25.0` - Claude models
- **Ollama**: `ollama>=0.3.0` - Local model serving

### Document Processing
- **BeautifulSoup4**: HTML parsing for web crawling
- **lxml**: XML/HTML processing backend
- **Built-in**: Text extraction and chunking

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking

## 📁 Supported File Types

### Currently Supported
- **Text Files**: `.txt`, `.md`
- **PDF Documents**: `.pdf` (text extraction)

### Future Roadmap
- Microsoft Office documents (`.docx`, `.pptx`)
- Web pages (direct URL indexing)
- Jupyter notebooks (`.ipynb`)
- Code files (`.py`, `.js`, etc.)

## 🔧 Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIMARY_LLM_PROVIDER` | `ollama` | Main LLM provider (ollama, openai, anthropic, google) |
| `PRIMARY_EMBEDDING_PROVIDER` | `ollama` | Embedding provider (ollama, openai, google) |
| `OPENAI_API_KEY` | `""` | OpenAI API key |
| `ANTHROPIC_API_KEY` | `""` | Anthropic API key |
| `GEMINI_API_KEY` | `""` | Google Gemini API key |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_LLM_MODEL` | `llama3.1` | Ollama LLM model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHUNK_SIZE` | `1000` | Document chunk size in characters |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `MIN_CONFIDENCE` | `0.25` | Minimum confidence threshold |
| `MAX_CHUNKS` | `5` | Maximum chunks to retrieve |
| `DOCS_FOLDER` | `./docs` | Default documents directory |
| `CHROMA_DB_PATH` | `./chroma_db` | Vector database storage path |
| `CRAWL_SOURCES` | `""` | Comma-separated URLs to crawl |
| `CRAWL_DEPTH` | `2` | Maximum crawl depth |
| `CRAWL_MAX_PAGES` | `50` | Maximum pages to crawl |

## 🔧 Troubleshooting

### Common Issues

#### Provider Connection Errors
```bash
# Test provider connectivity
rag-explorer ping

# Check configuration
rag-explorer status

# Common fixes:
# 1. Verify API keys in .env file
# 2. Check internet connection
# 3. Validate provider URLs (for Ollama)
```

#### No Results Found
```bash
# Check if documents are indexed
rag-explorer metrics

# Re-index documents if needed
rag-explorer index ./docs

# Adjust confidence threshold
# In .env: MIN_CONFIDENCE=0.1
```

#### Ollama Issues
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# Pull required models
ollama pull llama3.1
ollama pull nomic-embed-text
```

#### Database Issues
```bash
# Check database status
rag-explorer status

# Reset database if corrupted
rm -rf ./chroma_db
rag-explorer index ./docs  # Re-index
```

### Debug Mode

Set environment variable for detailed logging:
```bash
export RAG_DEBUG=true
rag-explorer ask "your question"
```

## 🔄 Development

### Project Structure
```
rag-explorer/
├── src/rag_explorer/
│   ├── cli/              # CLI commands and interface
│   ├── core/             # Core RAG logic and providers
│   ├── engine/           # Unified RAG engine
│   └── utils/            # Utilities and settings
├── tests/                # Test suite
├── docs/                 # Example documents
└── pyproject.toml        # Project configuration
```

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src/rag_explorer
```

### Code Formatting
```bash
# Format code
black src/

# Sort imports
isort src/

# Type checking
mypy src/
```

## 📄 License & Credits

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Author
**Aditya Mittal** - [theadityamittal@gmail.com](mailto:theadityamittal@gmail.com)

### Acknowledgments
- OpenAI for GPT models and embeddings API
- Google for Gemini models and embedding services
- Anthropic for Claude models
- Ollama community for local model serving
- ChromaDB team for the vector database
- Click and Rich libraries for CLI interface

---

**Made with ❤️ for personal knowledge exploration**

For issues and feature requests, please visit: [GitHub Issues](https://github.com/theadityamittal/rag-explorer/issues)