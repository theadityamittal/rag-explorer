# CLI Usage Guide - Support Deflect Bot

## Installation & Setup

1. **Install the CLI:**
   ```bash
   # From project directory
   pip install -e .
   
   # If you get "deflect-bot: command not found", add to PATH:
   # For Python 3.9:
   export PATH="$PATH:$HOME/Library/Python/3.9/bin"
   # For Python 3.11:
   export PATH="$PATH:$HOME/Library/Python/3.11/bin"
   # For Python 3.12:
   export PATH="$PATH:$HOME/Library/Python/3.12/bin"
   
   # Or find your Python version:
   python3 --version
   ```

2. **Verify Installation:**
   ```bash
   deflect-bot --help
   deflect-bot config  # Show current configuration
   ```

## Core Commands

### 📚 **Document Management**

#### Index Local Documentation
```bash
# Index all .md and .txt files from ./docs folder
deflect-bot index

# Example output:
# 🔄 Indexing local documentation...
# ✅ Successfully indexed 129 chunks from ./docs
```

#### Search Documents
```bash
# Basic search
deflect-bot search "installation process"

# Limit results
deflect-bot search "docker setup" --limit 3

# JSON output
deflect-bot search "configuration" --output json

# Quiet mode (just file paths)
deflect-bot --quiet search "api"
```

### 🤖 **Interactive Q&A Session**

#### Start Conversational Mode
```bash
# Start interactive session
deflect-bot ask

# Example session:
# 🤖 Support Deflect Bot - Interactive Q&A Session  
# Ask me anything about your documentation. Type 'end' to exit.
#
# ❓ You: What is docker?
# 🤖 Bot: Docker is used for deployment as evidenced by the Dockerfile 
#      and docker-compose.yml files. To create a Docker image from the 
#      current directory, run: `docker build -t myapp .`
#
# 📚 Sources:
#   [1] ./docs/deployment_infrastructure.md:0 - # Deployment & Infrastructure...
#   [2] ./docs/deployment_infrastructure.md:1 - WORKDIR /app...
#
# ❓ You: end
# 👋 Goodbye! Asked 1 questions in this session.
```

#### Advanced Ask Options
```bash
# Filter responses to specific domains
deflect-bot ask --domains "docs.python.org,github.com"

# Note: Currently the --confidence parameter is not implemented
# Confidence thresholds are controlled via environment variables
```

### 🕷️ **Web Crawling**

#### Simple URL Crawling
```bash
# Crawl specific URLs
deflect-bot crawl https://docs.python.org/3/library/os.html

# Multiple URLs
deflect-bot crawl https://example.com/docs https://example.com/api

# Force re-index (ignore cache)
deflect-bot crawl https://example.com/docs --force
```

#### Deep Crawling with Options
```bash
# Deep crawl with depth limit
deflect-bot crawl https://docs.python.org/3/ --depth 2 --max-pages 50

# Restrict to same domain
deflect-bot crawl https://example.com --depth 3 --same-domain --max-pages 100

# Comprehensive crawl
deflect-bot crawl https://docs.myapi.com --depth 2 --max-pages 75 --force
```

#### Use Default Seeds
```bash
# Crawl configured default URLs
deflect-bot crawl --default

# Show what will be crawled (check config first)
deflect-bot config  # Look for CRAWL_* settings
```

### ⚙️ **System Operations**

#### Health & Status
```bash
# Basic system health
deflect-bot status

# Test LLM connection
deflect-bot ping

# Show performance metrics  
deflect-bot metrics

# JSON format for metrics
deflect-bot metrics --output json
```

#### Configuration
```bash
# Show current configuration
deflect-bot config

# Set configuration (future feature)
# deflect-bot config --set ANSWER_MIN_CONF=0.30
```

## Global Options

**IMPORTANT**: Global options must come BEFORE the command, not after!

```bash
# ✅ CORRECT syntax:
deflect-bot --verbose search "docker"     # ✅ Global option before command
deflect-bot --quiet index               # ✅ Global option before command

# ❌ WRONG syntax:
deflect-bot search "docker" --verbose    # ❌ Will show "No such option" error
deflect-bot index --quiet               # ❌ Will show "No such option" error

# Show version
deflect-bot --version
```

### Available Global Options:
- `--verbose, -v`: Show extra details and progress messages
- `--quiet, -q`: Minimal output (great for scripting)
- `--version`: Show version information
- `--help`: Show help information

## Usage Patterns

### 1. **First-Time Setup**
```bash
# 1. Check configuration
deflect-bot config

# 2. Index local docs
deflect-bot index

# 3. Test the system
deflect-bot ping
deflect-bot search "test query"

# 4. Start asking questions
deflect-bot ask
```

### 2. **Daily Usage**
```bash
# Quick search
deflect-bot search "how to deploy"

# Interactive session for multiple questions
deflect-bot ask

# Update documentation
deflect-bot crawl https://updated-docs.com --force
deflect-bot index  # Re-index local changes
```

### 3. **Automation & Scripting**
```bash
#!/bin/bash
# Daily update script

# Update from web sources
deflect-bot crawl --default --force

# Re-index local docs
deflect-bot index

# Check system health
deflect-bot status --quiet && echo "System OK" || echo "System issues"
```

### 4. **Troubleshooting**
```bash
# Check system status
deflect-bot ping          # Test LLM connection
deflect-bot status        # Overall health
deflect-bot metrics       # Performance stats

# Verify document indexing
deflect-bot search "test" --limit 1
deflect-bot config        # Check settings

# Re-index everything
deflect-bot crawl --default --force
deflect-bot index
```

## Output Examples

### Search Results (Table Format)
```
                    Search Results for: 'docker setup'
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Rank ┃ Source                      ┃ Distance ┃ Preview                    ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1    │ ./docs/deployment_infrastr… │ 0.319    │ # Deployment & Infrastructure │
│ 2    │ ./docs/database_operations… │ 0.469    │ # Database Operations       │
└──────┴───────────────────────────┴──────────┴──────────────────────────────┘
```

### Interactive Q&A Session
```
🤖 Support Deflect Bot - Interactive Q&A Session
Ask me anything about your documentation. Type 'end' to exit.

❓ You: What is docker?
🤖 Bot: Docker is used for deployment as evidenced by the Dockerfile 
and docker-compose.yml files. To create a Docker image from the current 
directory, run: `docker build -t myapp .`

📚 Sources:
  [1] ./docs/deployment_infrastructure.md:0 - # Deployment & Infrastructure...
  [2] ./docs/deployment_infrastructure.md:1 - WORKDIR /app...

❓ You: end
👋 Goodbye! Asked 1 questions in this session.
```

### Configuration Display
```
                 Configuration                  
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Setting            ┃ Value                 ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ APP_NAME            │ Support Deflection Bot │
│ ANSWER_MIN_CONF     │ 0.25                   │
│ MAX_CHUNKS          │ 5                      │
│ CRAWL_DEPTH         │ 3                      │
└─────────────────────┴────────────────────────┘
```

## Tips & Best Practices

1. **Start with indexing:** Always run `deflect-bot index` after adding new documentation
2. **Use interactive mode:** The `ask` command is perfect for exploring your docs conversationally  
3. **Regular updates:** Use `--force` flag to refresh cached web content
4. **Check metrics:** Monitor performance with `deflect-bot metrics`
5. **Quiet scripts:** Use `--quiet` for automation and scripting scenarios
6. **Global options syntax:** Remember global options go BEFORE commands (`--quiet search` not `search --quiet`)

## Common Issues & Troubleshooting

### Warning Messages (Safe to Ignore)
You may see these warnings during normal operation:
```
urllib3 NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+
Failed to send telemetry event: capture() takes 1 positional argument
```
These are harmless warnings from dependencies and don't affect functionality.

### Command Syntax Errors
```bash
# Error: "No such option: --quiet"
# Fix: Move global options before the command
deflect-bot --quiet search "query"  # ✅ Correct
```

### Empty Search Results  
```bash
# If search returns no results:
deflect-bot index              # Re-index documents
deflect-bot search "broader terms"  # Try broader search terms
```

### LLM Connection Issues
```bash
# If ping fails, check Ollama is running:
ollama list                    # Should show installed models
ollama pull llama3.1          # Re-download if needed
```

## Migration from API

If you were previously using the FastAPI version:

| Old API Endpoint | New CLI Command |
|-----------------|-----------------|
| `POST /ask` | `deflect-bot ask` |
| `POST /search` | `deflect-bot search "query"` |
| `POST /reindex` | `deflect-bot index` |
| `POST /crawl` | `deflect-bot crawl url1 url2` |
| `POST /crawl_depth` | `deflect-bot crawl --depth 2 url` |
| `GET /metrics` | `deflect-bot metrics` |
| `GET /healthz` | `deflect-bot status` |
| `GET /llm_ping` | `deflect-bot ping` |

The CLI provides the same functionality with a much better user experience!