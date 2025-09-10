# Support Deflection Bot

**🤖 Intelligent CLI for document Q&A with confidence-based refusal**

Transform your documentation into a smart terminal assistant that answers questions accurately or refuses gracefully. Built for reliability over chattiness.

✨ **New**: Multi-provider AI support with cost optimization! Choose from 8 different LLM providers with automatic fallback chains and budget control.

## Quick Start (5 minutes)

### Local Setup (Recommended)
```bash
# 1. Clone and install
git clone https://github.com/theadityamittal/support-deflect-bot.git
cd support-deflect-bot
pip install -e .

# 2. Install Ollama (default AI provider)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1            # Default LLM model
ollama pull nomic-embed-text    # Default embedding model

# 3. Start using immediately
deflect-bot index ./docs        # Index your documentation
deflect-bot ask                 # Start Q&A with Ollama
# ❓ You: How do I configure the system?
# 🤖 Bot: [Answer using local llama3.1 model]
```

### API Providers Setup (Advanced)
For enhanced performance and multiple provider options:

```bash
# 1. Install as above, then add API keys
export OPENAI_API_KEY="your_openai_key"        # GPT-4o-mini (cost-effective)
export GROQ_API_KEY="your_groq_key"           # Ultra-fast inference
export ANTHROPIC_API_KEY="your_claude_key"    # Claude API

# 2. System automatically uses API providers when available
deflect-bot ask  # Now uses API providers with Ollama fallback
```

### Development Setup
```bash
# Clone and install in development mode with all dependencies
git clone https://github.com/theadityamittal/support-deflect-bot.git
cd support-deflect-bot
pip install -e .[dev]
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

### 🚀 Multi-Provider Intelligence
- **8 AI Providers**: OpenAI, Groq, Mistral, Google Gemini, Claude API, Claude Code, Ollama
- **Cost Optimization**: Automatic selection of most cost-effective providers
- **Smart Fallbacks**: Seamless switching if primary provider is unavailable  
- **Budget Control**: Set monthly limits with real-time cost tracking
- **Regional Compliance**: GDPR-compliant providers for EU users
- **Subscription Leverage**: Use your existing Claude Pro and Google One AI Pro subscriptions

> **💰 Cost Example**: Default setup costs ~$0.15 per 1M input tokens using GPT-4o-mini, with free tiers available through Groq and Google.

## 📖 Documentation

Comprehensive documentation is now organized in the `docs/` folder:

- **[Installation Guide](docs/installation.md)** - Step-by-step setup instructions
- **[Usage Guide](docs/usage.md)** - CLI commands and usage patterns  
- **[Configuration](docs/configuration.md)** - Environment variables and customization
- **[Provider Setup](docs/providers.md)** - Multi-provider configuration and cost optimization
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
# ❓ You: How do I set up API providers?
# ❓ You: Which provider is most cost-effective?
# ❓ You: How do I optimize my budget settings?
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