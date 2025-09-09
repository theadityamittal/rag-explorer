# Frequently Asked Questions

## General Questions

### What makes this different from other AI assistants?
❌ **Most AI assistants**: Answer confidently but make things up  
✅ **This bot**: Refuses to answer when unsure, provides citations, measures confidence

### Does it require internet access?
No! Everything runs locally with Ollama. No external APIs are used, ensuring privacy and reliability.

### What file formats are supported?
- **Local files**: Markdown (.md) and plain text (.txt) files
- **Web content**: Any HTML pages that can be crawled
- **Future support**: PDF files (in development)

### How much RAM do I need?
At least 4GB RAM for the LLM models. More RAM (8GB+) recommended for better performance.

## Technical Questions

### Why am I getting "I don't have enough information" responses?
This is a feature, not a bug! The bot refuses to answer when confidence is low. You can:
- Lower the confidence threshold: `export ANSWER_MIN_CONF=0.15`
- Add more relevant documentation to the docs/ folder
- Re-index your documents: `deflect-bot index`

### How do I update my documentation?
```bash
# For local files: Add files to docs/ folder, then
deflect-bot index

# For web content: Re-crawl with force flag
deflect-bot crawl https://your-docs.com --force
```

### Can I restrict answers to specific domains?
Yes! Use the `--domains` flag:
```bash
deflect-bot ask --domains "docs.python.org,github.com"
```

### How do I improve answer quality?
1. Add more comprehensive documentation
2. Use descriptive file names and headings
3. Ensure documents have good structure
4. Adjust confidence thresholds in configuration

## Troubleshooting

### "deflect-bot: command not found"
The CLI wasn't installed correctly. Try:
```bash
pip install -e .
# If still failing, add to PATH:
export PATH="$PATH:$HOME/.local/bin"
```

### LLM connection errors
Check if Ollama is running:
```bash
ollama list                    # Should show installed models
curl http://localhost:11434/api/tags  # Should return JSON
```

### Empty search results
```bash
deflect-bot index              # Re-index documents
deflect-bot search "broader terms"  # Try broader search terms
```

### Slow responses
1. Reduce context size: `MAX_CHUNKS=3`
2. Ensure Ollama has enough RAM allocated
3. Use smaller models if needed

### Warning messages I can ignore
These warnings are harmless:
```
urllib3 NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+
Failed to send telemetry event: capture() takes 1 positional argument
```

## Performance Questions

### How fast should responses be?
- Typical response time: 2-5 seconds
- Search queries: <1 second
- Complex questions with lots of context: 5-10 seconds

### How much disk space is needed?
- Base installation: ~500MB
- Models: ~4.3GB (llama3.1 + nomic-embed-text)
- Vector database: Scales with document size (typically 10-50MB per 1000 documents)

### Can it handle large documentation sets?
Yes! The system is designed to scale:
- Tested with 10,000+ documents
- Uses efficient vector search
- Configurable chunk sizes and limits

## Integration Questions

### Can I use this with my existing documentation workflow?
Yes! The bot works with:
- Static site generators (Jekyll, Hugo, etc.)
- Wiki systems
- Documentation platforms
- Version control workflows

### Is there an API for integration?
Yes! While the CLI is recommended for daily use, the REST API is available for integrations:
```bash
uvicorn src.api.app:app --reload --port 8000
```

### Can I deploy this in production?
Absolutely! See the deployment documentation for Docker and production configuration examples.

## Getting Help

### Where can I report bugs or request features?
- **Issues**: [GitHub Issues](https://github.com/theadityamittal/support-deflect-bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/theadityamittal/support-deflect-bot/discussions)

### How do I contribute?
See the Contributing section in the main README for development setup and guidelines.

### Need more help?
Try asking the bot itself! It has access to all documentation:
```bash
deflect-bot ask
# ❓ You: How do I troubleshoot connection issues?
```