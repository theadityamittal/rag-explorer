# Usage Guide

## Quick Start (5 minutes)

```bash
# 1. Install Ollama and models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1
ollama pull nomic-embed-text

# 2. Clone and setup
git clone <repo-url>
cd support-deflect-bot
python -m venv .venv && source .venv/bin/activate
pip install -e .  # Install CLI

# 3. Index your docs and start asking questions
deflect-bot index                          # Index ./docs folder
deflect-bot ask                           # Start interactive Q&A
# ❓ You: How do I configure the system?
# 🤖 Bot: [Intelligent answer with citations]
# ❓ You: end
```

## CLI Commands

### 📚 **Document Management**

```bash
# Index local documentation
deflect-bot index
# 🔄 Indexing local documentation...
# ✅ Successfully indexed 129 chunks from ./docs

# Search through documents
deflect-bot search "installation process"
# Shows beautiful formatted table with results

# Search with options
deflect-bot search "docker" --limit 3 --output json
deflect-bot --quiet search "api"  # Just file paths
```

### 🤖 **Interactive Q&A Session**

```bash
# Start conversational mode
deflect-bot ask
# 🤖 Support Deflect Bot - Interactive Q&A Session
# Ask me anything about your documentation. Type 'end' to exit.
# 
# ❓ You: How do I configure the database?
# 🤖 Bot: The database can be configured by setting CHROMA_DB_PATH...
#      📚 Sources:
#        [1] ./docs/configuration.md:25 - Database configuration...
# 
# ❓ You: end
# 👋 Goodbye! Asked 1 questions in this session.

# Filter responses by domains
deflect-bot ask --domains "docs.python.org,github.com"
```

### 🕷️ **Web Crawling**

```bash
# Crawl specific URLs
deflect-bot crawl https://docs.python.org/3/library/os.html

# Deep crawl with options  
deflect-bot crawl https://example.com --depth 2 --max-pages 50

# Crawl with domain restrictions
deflect-bot crawl https://docs.myapi.com --depth 2 --same-domain

# Use configured default URLs
deflect-bot crawl --default

# Force refresh cached content
deflect-bot crawl https://example.com/docs --force
```

### ⚙️ **System Operations**

```bash
# Check system health
deflect-bot status
# 🏥 Checking system health...
# ✅ System status: OK

# Test LLM connection
deflect-bot ping  
# 🏓 Pinging LLM service...
# ✅ LLM responded: Yeah Yeah! I'm awake!

# Show performance metrics
deflect-bot metrics
# Beautiful table with response times and counts

# View configuration
deflect-bot config
# Shows all current settings in a formatted table
```

## Global Options

**Important**: Global options must come BEFORE commands!

```bash
# ✅ Correct syntax:
deflect-bot --quiet search "docker"     # Minimal output  
deflect-bot --verbose index            # Extra details
deflect-bot --help                     # Show help

# ❌ Wrong syntax (will error):
deflect-bot search "docker" --quiet    # Error: No such option
```

## Usage Patterns

### First-Time Setup
```bash
deflect-bot config    # Check configuration
deflect-bot index     # Index local docs
deflect-bot ping      # Test system
deflect-bot ask       # Start asking questions
```

### Daily Usage
```bash
deflect-bot search "how to deploy"           # Quick search
deflect-bot ask                              # Interactive session
deflect-bot crawl https://updated-docs.com --force  # Update docs
```