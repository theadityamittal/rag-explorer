# RAG Explorer

**🔍 Simple personal RAG exploration tool**

A streamlined CLI tool for experimenting with Retrieval-Augmented Generation (RAG) using your own documents. Perfect for learning how RAG works and comparing different AI providers.

## ✨ Features

- **4 AI Providers**: Ollama (local), OpenAI, Anthropic, Google Gemini
- **Local-First**: Works completely offline with Ollama
- **Simple Setup**: Just 3 commands to get started
- **Confidence Scoring**: Know when the AI is uncertain
- **Document Citations**: See exactly where answers come from
- **ChromaDB Storage**: Efficient local vector database

## 🚀 Quick Start

### 1. Install
```bash
# Clone and install
git clone https://github.com/theadityamittal/rag-explorer.git
cd rag-explorer
pip install -e .
```

### 2. Set up Ollama (Local AI)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3.1            # For text generation
ollama pull nomic-embed-text    # For embeddings
```

### 3. Use it!
```bash
# Index your documents
rag-explorer index ./docs

# Ask questions
rag-explorer ask "How do I configure authentication?"

# Check status
rag-explorer status
```

## 🔧 Configuration

Create a `.env` file (optional):

```bash
# Primary provider (ollama, openai, anthropic, google)
PRIMARY_LLM_PROVIDER=ollama
PRIMARY_EMBEDDING_PROVIDER=ollama

# API Keys (optional - leave empty to use only Ollama)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=

# Ollama settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# RAG settings
CHUNK_SIZE=800
CHUNK_OVERLAP=100
MIN_CONFIDENCE=0.25
MAX_CHUNKS=5

# Paths
DOCS_FOLDER=./docs
CHROMA_DB_PATH=./chroma_db
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Index markdown files
rag-explorer index ./my-docs

# Ask a question
rag-explorer ask "What is the API rate limit?"

# Check what's indexed
rag-explorer status
```

### With API Providers
```bash
# Set up OpenAI
export OPENAI_API_KEY="your-key-here"
export PRIMARY_LLM_PROVIDER="openai"

# Now uses OpenAI for generation
rag-explorer ask "Explain the deployment process"
```

### Advanced Options
```bash
# Custom chunk size
rag-explorer index ./docs --chunk-size 1000 --chunk-overlap 200

# Custom confidence threshold
rag-explorer ask "How do I deploy?" --confidence 0.5

# More context chunks
rag-explorer ask "What are the requirements?" --max-chunks 10
```

## 🧠 How It Works

1. **Document Indexing**: Splits documents into chunks and generates embeddings
2. **Question Processing**: Converts your question into an embedding
3. **Retrieval**: Finds the most similar document chunks
4. **Confidence Scoring**: Calculates how confident the system is
5. **Generation**: Uses an LLM to generate an answer from the context
6. **Citation**: Shows which documents were used

## 🔄 Provider Comparison

| Provider | Type | Cost | Speed | Quality |
|----------|------|------|-------|---------|
| Ollama | Local | Free | Medium | Good |
| OpenAI | API | Low | Fast | Excellent |
| Anthropic | API | Medium | Fast | Excellent |
| Google | API | Low | Fast | Very Good |

## 📁 Supported File Types

- `.md` - Markdown files
- `.txt` - Plain text files
- `.rst` - reStructuredText files

## 🛠️ Development

```bash
# Install with dev dependencies
pip install -e .[dev]

# Run tests
pytest

# Format code
black src/
isort src/
```

## 🎓 Learning RAG

This tool is perfect for understanding:

- How document chunking affects retrieval quality
- Differences between embedding models
- How LLM providers handle the same context
- The importance of confidence scoring in RAG systems
- Local vs. cloud AI trade-offs

## 🤔 Troubleshooting

### "No providers available"
- Make sure Ollama is running: `ollama serve`
- Or set up an API key: `export OPENAI_API_KEY="your-key"`

### "No documents found"
- Check your document directory exists
- Ensure files have supported extensions (.md, .txt, .rst)

### "ChromaDB connection failed"
- The database will be created automatically
- Check write permissions in the current directory

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Built for learning and experimentation with RAG systems. Perfect for developers, researchers, and AI enthusiasts who want to understand how retrieval-augmented generation works under the hood.

---

**Happy exploring! 🚀**
