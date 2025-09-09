# Installation Guide

## Prerequisites
- **Python 3.9+** (3.11+ recommended)
- **API Keys**: At least one provider API key (OpenAI, Groq, Claude, etc.)
- **2GB+ RAM**: For vector database and document processing
- **Optional**: Ollama for local deployment (requires 4GB+ RAM)

## Installation Paths

Choose the installation path that best fits your needs:

### Path 1: API Providers (Recommended) 🚀
**Best for**: Most users, cost-effective operation, reliable performance
- **Setup time**: 3 minutes
- **Cost**: ~$0.15 per 1M tokens (GPT-4o-mini)
- **Benefits**: No local setup, automatic updates, multiple provider options

### Path 2: Local Deployment 🔒
**Best for**: Privacy-focused users, offline usage, full control
- **Setup time**: 10 minutes
- **Cost**: Free (hardware requirements)
- **Benefits**: Complete privacy, offline capability, no API costs

### Path 3: Hybrid Setup ⚡
**Best for**: Advanced users, maximum flexibility
- **Setup time**: 15 minutes
- **Benefits**: Best of both worlds, automatic fallback

---

## Path 1: API Providers Installation (Recommended)

### 1. Install with API providers
```bash
# Install directly from pip (coming soon)
pip install support-deflect-bot[api]

# OR install from source
git clone https://github.com/theadityamittal/support-deflect-bot.git
cd support-deflect-bot
pip install -e .[api]
```

### 2. Set up your API keys
Choose one or more providers and get API keys:

#### OpenAI (Recommended - Most Cost-Effective)
```bash
# Get API key from: https://platform.openai.com/api-keys
export OPENAI_API_KEY="your_openai_api_key_here"

# Test connection
python -c "import openai; print('✅ OpenAI connection ready')"
```

#### Groq (Ultra-Fast, Free Tier)
```bash
# Get API key from: https://console.groq.com/keys
export GROQ_API_KEY="your_groq_api_key_here"

# Test connection
python -c "import groq; print('✅ Groq connection ready')"
```

#### Claude API (Premium Quality)
```bash
# Get API key from: https://console.anthropic.com/
export ANTHROPIC_API_KEY="your_claude_api_key_here"

# Test connection
python -c "import anthropic; print('✅ Claude API connection ready')"
```

#### Google Gemini (Free Tier Available)
```bash
# Get API key from: https://aistudio.google.com/app/apikey
export GOOGLE_API_KEY="your_google_api_key_here"

# Test connection
python -c "import google.generativeai; print('✅ Google Gemini connection ready')"
```

#### Mistral (EU-Compliant)
```bash
# Get API key from: https://console.mistral.ai/
export MISTRAL_API_KEY="your_mistral_api_key_here"

# Test connection
python -c "from mistralai.client import MistralClient; print('✅ Mistral connection ready')"
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env to add your API keys and customize settings
```

### 4. Start using immediately
```bash
# Add your documentation to ./docs folder
cp your-docs/*.md docs/

# Index your documentation
deflect-bot index

# Start Q&A with automatic provider selection
deflect-bot ask
# ❓ You: How do I configure the system?
# 🤖 Bot: [Answer using best available provider]
```

---

## Path 2: Local Deployment (Privacy-Focused)

### 1. Install Ollama and models
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models (this may take 10-15 minutes)
ollama pull llama3.1        # ~4GB - for text generation
ollama pull nomic-embed-text # ~274MB - for embeddings

# Verify installation
ollama list
```

### 2. Install support-deflect-bot
```bash
# Install with all providers (including local)
pip install support-deflect-bot[all]

# OR from source
git clone https://github.com/theadityamittal/support-deflect-bot.git
cd support-deflect-bot
pip install -e .[all]
```

### 3. Start using locally
```bash
# Index your documentation
deflect-bot index ./docs

# Uses local Ollama automatically (no API keys needed)
deflect-bot ask
```

---

## Path 3: Hybrid Setup (Maximum Flexibility)

### 1. Install with all providers
```bash
pip install support-deflect-bot[all]
```

### 2. Set up both API keys AND local Ollama
```bash
# Set up API providers (follow Path 1, Step 2)
export OPENAI_API_KEY="your_key"
export GROQ_API_KEY="your_key"

# Install Ollama (follow Path 2, Step 1)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1
```

### 3. Configure fallback strategy
```bash
# Edit .env to set provider priority
DEFAULT_PROVIDER_STRATEGY=cost_optimized  # API providers first
# System automatically falls back to Ollama if API providers fail
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