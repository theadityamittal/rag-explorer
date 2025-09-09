# Installation Guide

## Prerequisites
- **Python 3.11+**
- **Ollama**: [Install from ollama.com](https://ollama.com/)
- **4GB+ RAM**: For running local LLM models

## Step-by-Step Installation

### 1. Install Ollama and models
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models (this may take a few minutes)
ollama pull llama3.1        # ~4GB - for text generation
ollama pull nomic-embed-text # ~274MB - for embeddings

# Verify installation
ollama list
```

### 2. Setup Python environment
```bash
git clone <your-repo-url>
cd support-deflect-bot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install CLI application
pip install -e .

# Verify CLI installation
deflect-bot --help
```

### 3. Configure environment (optional)
```bash
cp .env.example .env
# Edit .env to customize settings (see Configuration section)
```

### 4. Add your documentation
```bash
# Add your .md and .txt files to the docs/ folder
cp your-docs/*.md docs/
```

### 5. Index your documentation
```bash
# Index local files from ./docs folder  
deflect-bot index

# Output: ✅ Successfully indexed N chunks from ./docs
```

### 6. Verify setup
```bash
# Check system health
deflect-bot status

# Check LLM connectivity  
deflect-bot ping

# Test asking a question
deflect-bot ask
# ❓ You: How do I configure the system?
# 🤖 Bot: [Intelligent answer with citations]
# ❓ You: end
```

## Troubleshooting Installation

### CLI Path Issues
If you get "deflect-bot: command not found":
```bash
# Add to PATH (adjust Python version as needed)
export PATH="$PATH:$HOME/Library/Python/3.11/bin"  # macOS
export PATH="$PATH:$HOME/.local/bin"               # Linux

# Or find your Python version:
python3 --version
```

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
systemctl restart ollama  # Linux
brew services restart ollama  # macOS

# Check model availability
ollama list
```

## Docker Installation

```bash
# Build image
docker build -t support-deflect-bot .

# Run with external Ollama (recommended)
docker run -d \
  -p 8000:8000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v $(pwd)/docs:/app/docs \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --name support-bot \
  support-deflect-bot

# Index your documents
curl -X POST http://127.0.0.1:8000/reindex
```