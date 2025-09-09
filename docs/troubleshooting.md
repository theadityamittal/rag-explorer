# Troubleshooting Guide

## Common Issues

### Installation Issues

#### "deflect-bot: command not found"
The CLI wasn't installed correctly or isn't in your PATH.

**Solutions:**
```bash
# Reinstall the CLI
pip install -e .

# Add to PATH (adjust Python version as needed)
export PATH="$PATH:$HOME/.local/bin"               # Linux
export PATH="$PATH:$HOME/Library/Python/3.11/bin"  # macOS

# Find your Python version
python3 --version
```

#### "No module named 'deflect_bot'"
Virtual environment not activated or package not installed.

**Solutions:**
```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install in development mode
pip install -e .
```

### Ollama Connection Issues

#### "Connection refused" or "LLM ping failed"
Ollama service is not running or not accessible.

**Check Ollama status:**
```bash
# Test connection directly
curl http://localhost:11434/api/tags

# Check if models are installed
ollama list

# Start Ollama if not running
systemctl start ollama  # Linux
brew services start ollama  # macOS
```

#### "Model not found" errors
Required models aren't installed.

**Install models:**
```bash
ollama pull llama3.1        # Text generation
ollama pull nomic-embed-text # Embeddings

# Verify installation
ollama list
```

### Search and Indexing Issues

#### Empty search results
No documents indexed or search terms too specific.

**Solutions:**
```bash
# Re-index local documents
deflect-bot index

# Try broader search terms
deflect-bot search "configuration"  # Instead of "config file path"

# Check if documents exist
ls -la docs/

# Verify indexing worked
deflect-bot search "test" --limit 1
```

#### "I don't have enough information" responses
Confidence threshold too high or insufficient documentation.

**Solutions:**
```bash
# Lower confidence threshold
export ANSWER_MIN_CONF=0.15

# Add more comprehensive documentation
cp additional-docs/*.md docs/
deflect-bot index

# Check search results for your query
deflect-bot search "your question terms"
```

### Performance Issues

#### Slow responses (>10 seconds)
System under-resourced or configuration needs optimization.

**Solutions:**
```bash
# Reduce context size
export MAX_CHUNKS=3
export MAX_CHARS_PER_CHUNK=400

# Check system resources
htop | grep python
htop | grep ollama

# Monitor response times
deflect-bot metrics
```

#### High memory usage
Large document sets or high context limits.

**Solutions:**
```bash
# Optimize chunking
export MAX_CHUNKS=5
export MAX_CHARS_PER_CHUNK=600

# Clear vector database and re-index
rm -rf ./chroma_db
deflect-bot index

# Check database size
du -sh ./chroma_db
```

### Web Crawling Issues

#### "Domain not allowed" errors
Target domain not in ALLOW_HOSTS configuration.

**Solutions:**
```bash
# Check current allowed hosts
deflect-bot config | grep ALLOW_HOSTS

# Add domain to .env file
echo "ALLOW_HOSTS=docs.python.org,your-domain.com" >> .env

# Or set temporarily
export ALLOW_HOSTS="docs.python.org,your-domain.com"
```

#### Crawling fails or returns no content
Network issues, robots.txt restrictions, or authentication required.

**Solutions:**
```bash
# Force crawl (bypass cache)
deflect-bot crawl https://example.com --force

# Check robots.txt
curl https://example.com/robots.txt

# Add to trusted domains (bypasses robots.txt)
export TRUSTED_DOMAINS="your-domain.com"

# Test single page first
deflect-bot crawl https://example.com/single-page.html
```

### Configuration Issues

#### Settings not taking effect
Environment variables not loaded or syntax errors.

**Solutions:**
```bash
# Check current configuration
deflect-bot config

# Verify .env file syntax
cat .env

# Restart terminal/shell
source ~/.bashrc  # or ~/.zshrc

# Set variables directly
export ANSWER_MIN_CONF=0.25
```

#### Database corruption or inconsistent state
Vector database corrupted or out of sync.

**Solutions:**
```bash
# Reset vector database
rm -rf ./chroma_db

# Clear crawl cache
rm -f ./data/crawl_cache.json

# Re-index everything
deflect-bot index
deflect-bot crawl --default --force
```

## ✅ Known Resolved Issues

### Warnings and Noise (Fixed in Latest Version)
The following warnings have been automatically suppressed for cleaner CLI output:

- **urllib3 OpenSSL warnings**: `urllib3 v2 only supports OpenSSL 1.1.1+...`
- **ChromaDB telemetry errors**: `Failed to send telemetry event...`
- **Rich markup formatting**: Fixed malformed markup tags

These were harmless warnings that cluttered the output but didn't affect functionality. They are now automatically suppressed when using the CLI.

### Rich Formatting Issues (Fixed)
Fixed confidence score display and other Rich markup formatting issues that could cause parsing errors.

## Debug Mode

### Enable detailed logging
```bash
# Set log level to debug
export LOG_LEVEL=debug

# Run with verbose output
deflect-bot --verbose search "debug query"

# Check application logs
tail -f ~/.local/share/deflect-bot/app.log  # If logging to file
```

### Manual testing
```bash
# Test system components individually
deflect-bot ping      # LLM connection
deflect-bot status    # Overall health
deflect-bot config    # Configuration
deflect-bot metrics   # Performance stats

# Test search without asking
deflect-bot search "test query" --limit 3

# Test with minimal query
deflect-bot ask
# ❓ You: test
```

## System Health Checks

### Complete diagnostic script
```bash
#!/bin/bash
echo "=== System Diagnostic ==="

echo "1. Checking CLI installation..."
deflect-bot --version || echo "❌ CLI not installed"

echo "2. Checking Ollama connection..."
deflect-bot ping || echo "❌ Ollama connection failed"

echo "3. Checking configuration..."
deflect-bot config

echo "4. Checking document indexing..."
deflect-bot search "test" --limit 1 || echo "❌ No documents indexed"

echo "5. Checking system status..."
deflect-bot status

echo "6. Checking performance..."
deflect-bot metrics

echo "=== Diagnostic Complete ==="
```

## Performance Monitoring

### Response time monitoring
```bash
# Monitor API response times (if using API)
curl -s -w '%{time_total}s\n' http://127.0.0.1:8000/ask \
  -d '{"question": "test"}' > /dev/null

# Monitor CLI response times
time deflect-bot search "test query"
```

### Resource monitoring
```bash
# Monitor memory usage
ps aux | grep -E "(python|ollama)" | head -5

# Monitor disk usage
du -sh ./chroma_db ./data

# Monitor network (if using web crawling)
netstat -an | grep :11434
```

## Getting More Help

### Collect diagnostic information
When reporting issues, include:

```bash
# System info
uname -a
python3 --version
deflect-bot --version

# Configuration
deflect-bot config

# Recent logs (if available)
tail -50 ~/.local/share/deflect-bot/app.log

# Error reproduction
deflect-bot --verbose search "problematic query"
```

### Contact support
- **Issues**: [GitHub Issues](https://github.com/theadityamittal/support-deflect-bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/theadityamittal/support-deflect-bot/discussions)
- **Ask the bot**: Use the bot itself for troubleshooting questions!