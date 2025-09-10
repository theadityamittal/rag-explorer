# Installation Guide

## Prerequisites
- **Python 3.9+** (3.11+ recommended)
- **API Keys**: At least one provider API key (OpenAI, Groq, Claude, etc.)
- **2GB+ RAM**: For vector database and document processing
- **Optional**: Ollama for local deployment (requires 4GB+ RAM)

## Installation Paths

Choose the installation path that best fits your needs:

### Path 1: Local Deployment (Recommended) 🔒
**Best for**: Most users, privacy-focused operation, full control
- **Setup time**: 5 minutes
- **Cost**: Free (hardware requirements: 4GB+ RAM)
- **Benefits**: Complete privacy, offline capability, no API costs, reliable default setup

### Path 2: API Providers Setup ⚡
**Best for**: Advanced users seeking multiple provider options
- **Setup time**: 10 minutes
- **Cost**: ~$0.15 per 1M tokens (GPT-4o-mini)
- **Benefits**: Enhanced performance, multiple fallback options, cloud-based reliability

### Path 3: Development Setup 🛠️
**Best for**: Contributors and developers
- **Setup time**: 15 minutes
- **Benefits**: Full development environment, testing capabilities, contribution-ready

---

## Path 1: Local Deployment (Recommended)

### 1. Install Ollama and models
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models (this may take 10-15 minutes)
ollama pull llama3.1        # ~4GB - Default LLM model
ollama pull nomic-embed-text # ~274MB - Default embedding model

# Verify installation
ollama list
```

### 2. Install support-deflect-bot
```bash
# Clone and install from source
git clone https://github.com/theadityamittal/support-deflect-bot.git
cd support-deflect-bot
pip install -e .
```

### 3. Start using locally
```bash
# Index your documentation
deflect-bot index ./docs

# Uses local Ollama automatically (no API keys needed)
deflect-bot ask
# ❓ You: How do I configure the system?
# 🤖 Bot: [Answer using local llama3.1 model]
```

---

## Path 2: API Providers Setup (Advanced)

### 1. Install support-deflect-bot
```bash
# Clone and install from source
git clone https://github.com/theadityamittal/support-deflect-bot.git
cd support-deflect-bot
pip install -e .
```

### 2. Set up your API keys
Choose one or more providers and get API keys:

#### OpenAI (Recommended - Most Cost-Effective)
```bash
# Get API key from: https://platform.openai.com/api-keys
export OPENAI_API_KEY="your_openai_api_key_here"

# Test connection (if openai package installed)
python -c "import openai; print('✅ OpenAI connection ready')"
```

#### Groq (Ultra-Fast, Free Tier)
```bash
# Get API key from: https://console.groq.com/keys
export GROQ_API_KEY="your_groq_api_key_here"
```

#### Claude API (Premium Quality)
```bash
# Get API key from: https://console.anthropic.com/
export ANTHROPIC_API_KEY="your_claude_api_key_here"
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env to add your API keys and customize settings
```

### 4. Start using with API providers
```bash
# Index your documentation
deflect-bot index ./docs

# Start Q&A with automatic provider selection
deflect-bot ask
# ❓ You: How do I configure the system?
# 🤖 Bot: [Answer using best available API provider]
```

---

## Path 3: Development Setup

### 1. Clone and install in development mode
```bash
# Clone repository
git clone https://github.com/theadityamittal/support-deflect-bot.git
cd support-deflect-bot

# Install with development dependencies
pip install -e .[dev]
```

### 2. Set up testing environment
```bash
# Install Ollama for local testing
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1

# Copy environment template
cp .env.example .env
# Edit .env to add API keys for testing (optional)
```

### 3. Run tests and development tools
```bash
# Run test suite
python -m pytest tests/

# Start development server with hot reload
uvicorn src.app:app --reload --port 8000
```

---

## Verification & Testing

### Check provider status
```bash
# See which providers are available
deflect-bot providers list

# Test provider connections
deflect-bot providers test

# Check system health
deflect-bot status
```

### Test the system
```bash
# Index your documentation
deflect-bot index

# Ask a test question
deflect-bot ask
# ❓ You: How do I configure the system?
# 🤖 Bot: [Intelligent answer with citations and provider info]
```

## Troubleshooting Installation

### CLI Installation Issues
If you get "deflect-bot: command not found":
```bash
# Add to PATH (adjust Python version as needed)
export PATH="$PATH:$HOME/Library/Python/3.11/bin"  # macOS
export PATH="$PATH:$HOME/.local/bin"               # Linux

# Or install in user directory:
pip install --user support-deflect-bot[api]
```

### API Provider Issues

#### OpenAI API Errors
```bash
# Test API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Common issues:
# - Invalid API key: Check https://platform.openai.com/api-keys
# - Rate limits: Wait or upgrade plan
# - Insufficient credits: Check billing at https://platform.openai.com/account/billing
```

#### Groq API Errors
```bash
# Test API key
curl -H "Authorization: Bearer $GROQ_API_KEY" \
  https://api.groq.com/openai/v1/models

# Common issues:
# - Free tier limits: 1.2M tokens/day
# - Rate limits: 30 requests/minute
```

#### Authentication Errors
```bash
# Check environment variables
echo "OpenAI: ${OPENAI_API_KEY:0:8}..."
echo "Groq: ${GROQ_API_KEY:0:8}..."
echo "Claude: ${ANTHROPIC_API_KEY:0:8}..."

# Make environment variables permanent
echo 'export OPENAI_API_KEY="your_key"' >> ~/.bashrc
source ~/.bashrc
```

### Provider Selection Issues
```bash
# Check which providers are detected
deflect-bot providers list

# Force a specific provider
export DEFAULT_PROVIDER_STRATEGY=speed_focused

# Debug provider selection
deflect-bot --debug ask
```

### Ollama Issues (Local Deployment)
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
systemctl restart ollama  # Linux
brew services restart ollama  # macOS

# Check model availability
ollama list
```

### Installation Dependencies
```bash
# If you get import errors, install missing providers:
pip install openai groq mistralai anthropic google-generativeai

# For development:
pip install -e .[dev]
```

## Docker Installation

### API Providers (Recommended)
```bash
# Build image
docker build -t support-deflect-bot .

# Run with API providers
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your_openai_key" \
  -e GROQ_API_KEY="your_groq_key" \
  -e ANTHROPIC_API_KEY="your_claude_key" \
  -e DEFAULT_PROVIDER_STRATEGY="cost_optimized" \
  -v $(pwd)/docs:/app/docs \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --name support-bot \
  support-deflect-bot

# Index your documents
curl -X POST http://127.0.0.1:8000/reindex
```

### Local Ollama Setup
```bash
# Run with external Ollama
docker run -d \
  -p 8000:8000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v $(pwd)/docs:/app/docs \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --name support-bot \
  support-deflect-bot
```

### Hybrid Docker Setup
```bash
# Run with both API providers and Ollama fallback
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your_openai_key" \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e DEFAULT_PROVIDER_STRATEGY="cost_optimized" \
  -v $(pwd)/docs:/app/docs \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --name support-bot \
  support-deflect-bot
```

---

## Next Steps

After installation, check out:
- **[Provider Setup Guide](providers.md)** - Detailed provider configuration and cost optimization
- **[Configuration Guide](configuration.md)** - Advanced settings and customization
- **[Usage Guide](usage.md)** - CLI commands and usage patterns