# Configuration Guide

## Environment Variables

Create `.env` file or set environment variables:

```bash
# Application Settings
APP_NAME=Support Deflection Bot
APP_VERSION=0.1.0

# Ollama Configuration
OLLAMA_MODEL=llama3.1                    # LLM model for chat
OLLAMA_EMBED_MODEL=nomic-embed-text      # Embedding model
OLLAMA_HOST=http://localhost:11434       # Ollama service URL

# RAG Behavior
ANSWER_MIN_CONF=0.20                     # Confidence threshold (0.0-1.0)
MAX_CHUNKS=5                             # Max context chunks per query
MAX_CHARS_PER_CHUNK=800                  # Max characters per chunk

# Web Crawling
ALLOW_HOSTS=docs.python.org,pip.pypa.io  # Comma-separated allowed domains
TRUSTED_DOMAINS=help.sigmacomputing.com  # Domains that bypass robots.txt checks
CRAWL_DEPTH=1                            # Default crawl depth
CRAWL_MAX_PAGES=40                       # Max pages per crawl session
CRAWL_SAME_DOMAIN=true                   # Restrict to same domain

# Storage Paths
CHROMA_DB_PATH=./chroma_db               # Vector database location
DOCS_FOLDER=./docs                       # Local documentation folder
CRAWL_CACHE_PATH=./data/crawl_cache.json # Web crawl cache file
```

## Customization Examples

### Adjust Confidence Threshold
```bash
# Stricter (fewer answers, higher quality)
ANSWER_MIN_CONF=0.35

# More lenient (more answers, potentially lower quality)
ANSWER_MIN_CONF=0.15
```

### Configure for Different Domains
```bash
# Python documentation only
ALLOW_HOSTS=docs.python.org,packaging.python.org,pip.pypa.io

# JavaScript ecosystem
ALLOW_HOSTS=nodejs.org,npmjs.com,developer.mozilla.org

# Your company docs
ALLOW_HOSTS=docs.yourcompany.com,wiki.yourcompany.com
```

### Optimize for Different Use Cases
```bash
# Quick responses (less context)
MAX_CHUNKS=3
MAX_CHARS_PER_CHUNK=400

# Comprehensive answers (more context)
MAX_CHUNKS=8
MAX_CHARS_PER_CHUNK=1200
```

### Docker Configuration
```bash
# For Docker deployment with external Ollama
OLLAMA_HOST=http://host.docker.internal:11434  # macOS/Windows
OLLAMA_HOST=http://172.17.0.1:11434           # Linux
```

## Configuration Recipes

### Customer Support Bot
Perfect for customer support with company documentation.

```bash
ANSWER_MIN_CONF=0.30                     # High confidence for customer-facing
MAX_CHUNKS=5
ALLOW_HOSTS=docs.yourcompany.com,help.yourcompany.com
```

### Development Documentation Assistant
For developers working with multiple technology stacks.

```bash
ANSWER_MIN_CONF=0.20                     # Balanced for technical questions
MAX_CHUNKS=7                             # More context for complex topics
ALLOW_HOSTS=docs.python.org,nodejs.org,docs.docker.com,kubernetes.io
```

### Academic Research Assistant
For processing academic papers and research documentation.

```bash
ANSWER_MIN_CONF=0.35                     # Very strict for academic accuracy
MAX_CHUNKS=10                            # Comprehensive context
MAX_CHARS_PER_CHUNK=1000                 # Longer chunks for detailed content
```

## Checking Configuration

```bash
# View current configuration
deflect-bot config

# Shows all current settings in a formatted table
```