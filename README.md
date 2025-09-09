# Support Deflection Bot

**🤖 Intelligent CLI for document Q&A with confidence-based refusal**

Transform your documentation into a smart terminal assistant that answers questions accurately or refuses gracefully. Built for reliability over chattiness.

✨ **New**: Now with a beautiful CLI interface! Interactive conversations, rich terminal output, and intuitive commands.

## Quick Start (5 minutes)

```bash
# 1. Install Ollama and models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1
ollama pull nomic-embed-text

# 2. Clone and setup
git clone https://github.com/theadityamittal/support-deflect-bot.git
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

## What makes this different?

❌ **Most AI assistants**: Answer confidently but make things up  
✅ **This bot**: Refuses to answer when unsure, provides citations, measures confidence

### Core Behaviors
- **Grounded answers**: Only uses your provided documentation
- **Confident refusal**: Says "I don't have enough information" when evidence is weak
- **Citation tracking**: Shows exactly where answers come from
- **Confidence scoring**: Combines semantic similarity + keyword matching
- **Domain filtering**: Restrict answers to specific documentation sources

> **💡 Fresh Setup**: The bot starts with an empty knowledge base. You'll need to index your documentation (local files or web crawling) before asking questions. This ensures the bot only knows what you explicitly provide.

## 📖 Documentation

Comprehensive documentation is now organized in the `docs/` folder:

- **[Installation Guide](docs/installation.md)** - Step-by-step setup instructions
- **[Usage Guide](docs/usage.md)** - CLI commands and usage patterns  
- **[Configuration](docs/configuration.md)** - Environment variables and customization
- **[Features](docs/features.md)** - Complete feature overview
- **[FAQ](docs/faq.md)** - Frequently asked questions
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

## 🚀 Alternative: REST API

The original REST API is still available for integration scenarios:

```bash
# Start the API server (optional)
uvicorn src.api.app:app --reload --port 8000
```

The CLI is the recommended interface for daily use, while the API is perfect for integrations and automation.

---

## 🤔 Got Questions or Issues?

**Try asking our bot first!** It has access to all documentation and can help with most questions:

```bash
deflect-bot ask
# ❓ You: How do I troubleshoot connection issues?
# ❓ You: What configuration options are available?
# ❓ You: How do I optimize performance?
```

For bugs, feature requests, or if the bot can't help:
- **Issues**: [GitHub Issues](https://github.com/theadityamittal/support-deflect-bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/theadityamittal/support-deflect-bot/discussions)

---

## Developer

**Aditya Mittal**
- GitHub: [@theadityamittal](https://github.com/theadityamittal)
- Email: theadityamittal@gmail.com

---

## License

MIT License - see LICENSE file for details.

Built with ❤️ for reliable, grounded AI assistance.