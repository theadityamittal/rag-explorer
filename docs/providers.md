# Provider Setup & Configuration Guide

## Overview

Support Deflect Bot supports **8 different AI providers** with automatic failover and intelligent selection. The system automatically chooses the most cost-effective available provider while respecting regional compliance requirements.

### Provider Categories

- **🚀 Primary Providers**: OpenAI, Groq (recommended for most users)
- **🏆 Premium Providers**: Claude API, Google Gemini (higher quality, higher cost)
- **🌍 EU-Compliant**: Mistral, OpenAI, Claude API (GDPR compliant)
- **🔒 Local/Subscription**: Claude Code, Ollama (privacy-focused)

---

## Provider Comparison Table

| Provider | Cost (Input/Output) | Speed | GDPR | Free Tier | Best For |
|----------|--------------------|---------|---------|-----------|---------
| **OpenAI** | $0.15 / $0.60 | Fast | ✅ | No | Balanced cost/quality |
| **Groq** | $0.59 / $0.79 | Ultra-Fast | ❌ | ✅ 1.2M/day | Speed & free usage |
| **Mistral** | $0.25 / $0.25 | Fast | ✅ | No | EU compliance |
| **Google Gemini** | Free / $0.02 | Fast | Partial | ✅ 15 req/min | Free tier usage |
| **Claude API** | $0.25 / $1.25 | Medium | ✅ | No | Premium quality |
| **Claude Code** | Free* | Medium | ✅ | ✅ Pro sub | Existing subscription |
| **Ollama** | Free | Medium | ✅ | ✅ | Full privacy/offline |

*Requires Claude Pro subscription ($20/month)

**Cost per 1M tokens (USD)**

---

## Recommended Provider Strategies

### 💰 Cost-Optimized (Default)
**Best for**: Budget-conscious users, high-volume usage
- **Primary**: OpenAI GPT-4o-mini → Groq → Mistral
- **Monthly cost**: ~$5-10 for typical usage
- **Automatic**: GDPR filtering for EU users

### ⚡ Speed-Focused  
**Best for**: Real-time applications, chatbots
- **Primary**: Groq → OpenAI → Google Gemini
- **Response time**: <1 second average
- **Trade-off**: Higher cost for premium speed

### 🏆 Quality-First
**Best for**: Critical applications, complex reasoning
- **Primary**: Claude API → Google Gemini Pro → OpenAI GPT-4
- **Monthly cost**: $15-25
- **Best**: Reasoning, analysis, complex tasks

### 🔒 Privacy-Focused
**Best for**: Sensitive data, offline usage
- **Primary**: Ollama → Claude Code → Local only
- **Cost**: Hardware + subscriptions
- **Benefit**: Complete data control

---

## Provider Setup Guides

### OpenAI (Recommended)

**Why choose OpenAI?**
- Most cost-effective for quality ratio
- Globally compliant (including GDPR)
- Reliable infrastructure and support
- GPT-4o-mini provides excellent performance at low cost

**Setup:**
```bash
# 1. Get API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY="your_openai_api_key_here"

# 2. Test connection
python -c "
import openai
client = openai.OpenAI()
print('✅ OpenAI connected:', client.models.list().data[0].id)
"

# 3. Optional: Set custom model
export OPENAI_LLM_MODEL="gpt-4o-mini"  # Default, most cost-effective
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"  # Default
```

**Cost Management:**
```bash
# Monitor usage at: https://platform.openai.com/usage
# Set usage alerts at: https://platform.openai.com/account/billing/limits

# Typical costs (per 1M tokens):
# - GPT-4o-mini: $0.15 input, $0.60 output
# - GPT-4o: $2.50 input, $10.00 output
```

---

### Groq (Ultra-Fast)

**Why choose Groq?**
- Fastest inference speed (10-100x faster than competitors)
- Free tier: 1.2M tokens/day
- Excellent for real-time applications
- Llama 3.1 models with great performance

**Setup:**
```bash
# 1. Get API key from https://console.groq.com/keys
export GROQ_API_KEY="your_groq_api_key_here"

# 2. Test connection
python -c "
import groq
client = groq.Groq()
print('✅ Groq connected:', len(client.models.list().data), 'models')
"

# 3. Optional: Set custom model
export GROQ_MODEL="llama-3.1-8b-instant"  # Fastest
# export GROQ_MODEL="llama-3.1-70b-versatile"  # Higher quality
```

**Free Tier Limits:**
- **Daily**: 1.2M tokens
- **Rate**: 30 requests/minute  
- **Models**: All Llama 3.1 models included

---

### Claude API (Premium Quality)

**Why choose Claude API?**
- Excellent for reasoning and analysis
- GDPR compliant
- Large context window (200K tokens)
- Complements your Claude Pro subscription

**Setup:**
```bash
# 1. Get API key from https://console.anthropic.com/
export ANTHROPIC_API_KEY="your_claude_api_key_here"

# 2. Test connection
python -c "
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model='claude-3-haiku-20240307',
    max_tokens=10,
    messages=[{'role': 'user', 'content': 'Hi'}]
)
print('✅ Claude API connected:', response.content[0].text)
"

# 3. Optional: Set custom model
export CLAUDE_API_MODEL="claude-3-haiku-20240307"  # Most cost-effective
# export CLAUDE_API_MODEL="claude-3-sonnet-20240229"  # Balanced
# export CLAUDE_API_MODEL="claude-3-opus-20240229"  # Highest quality
```

---

### Google Gemini (Free Tier Available)

**Why choose Google Gemini?**
- Generous free tier (15 requests/minute)
- Large context window (1M tokens)
- Good performance for most tasks
- Leverages your Google One AI Pro subscription

**Setup:**
```bash
# 1. Get API key from https://aistudio.google.com/app/apikey
export GOOGLE_API_KEY="your_google_api_key_here"

# 2. Test connection
python -c "
import google.generativeai as genai
genai.configure(api_key='$GOOGLE_API_KEY')
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content('Hi')
print('✅ Google Gemini connected:', response.text[:50])
"

# 3. Optional: Set custom model
export GOOGLE_MODEL="gemini-1.5-flash"  # Free tier, fast
# export GOOGLE_MODEL="gemini-1.5-pro"  # Higher quality, paid
```

**Free Tier Limits:**
- **Rate**: 15 requests/minute
- **Daily**: 1,500 requests
- **Context**: Up to 1M tokens

---

### Mistral (EU-Compliant)

**Why choose Mistral?**
- EU-based and GDPR compliant
- Competitive pricing
- Good performance for European users
- Reduces latency for EU traffic

**Setup:**
```bash
# 1. Get API key from https://console.mistral.ai/
export MISTRAL_API_KEY="your_mistral_api_key_here"

# 2. Test connection
python -c "
from mistralai.client import MistralClient
client = MistralClient(api_key='$MISTRAL_API_KEY')
models = client.list_models()
print('✅ Mistral connected:', len(models.data), 'models')
"

# 3. Optional: Set custom model
export MISTRAL_MODEL="mistral-small-latest"  # Most cost-effective
# export MISTRAL_MODEL="mistral-medium-latest"  # Balanced
# export MISTRAL_MODEL="mistral-large-latest"  # Highest quality
```

---

### Claude Code (Pro Subscription)

**Why choose Claude Code?**
- Leverages your existing Claude Pro subscription ($20/month)
- No additional API costs
- Local subprocess execution
- Good fallback option

**Setup:**
```bash
# 1. Install Claude Code (if not installed)
# Download from: https://claude.ai/download

# 2. Verify installation
claude --version

# 3. Set custom path (if needed)
export CLAUDE_CODE_PATH="/path/to/claude"  # Default: "claude"

# 4. Test connection
deflect-bot providers test --provider claude_code
```

**Usage Notes:**
- Uses your Claude Pro message limit (~45 messages per 5 hours)
- Shared with your claude.ai web usage
- Subprocess execution (no streaming)

---

### Ollama (Local/Offline)

**Why choose Ollama?**
- Complete privacy and data control
- Offline operation
- No API costs after setup
- Full customization of models

**Setup:**
```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull models
ollama pull llama3.1        # ~4GB - for chat
ollama pull nomic-embed-text # ~274MB - for embeddings

# 3. Verify installation
ollama list

# 4. Optional: Set custom models
export OLLAMA_MODEL="llama3.1"
export OLLAMA_EMBED_MODEL="nomic-embed-text"
```

**Hardware Requirements:**
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB+ for models
- **CPU**: Modern multi-core processor

---

## Cost Optimization Strategies

### Budget-Friendly Setup ($5-10/month)
```bash
# Primary: Free and low-cost providers
export DEFAULT_PROVIDER_STRATEGY="cost_optimized"
export MONTHLY_BUDGET_USD=10.0

# Provider priority: Groq (free) → OpenAI (cheap) → others
# Expected usage: ~6M input + 1M output tokens/month
```

### Performance Setup ($15-25/month)
```bash
# Primary: Best performance providers
export DEFAULT_PROVIDER_STRATEGY="quality_first"
export MONTHLY_BUDGET_USD=25.0

# Provider priority: Claude API → Google Pro → OpenAI GPT-4
```

### Enterprise Setup ($50-100/month)
```bash
# Multiple providers with high limits
export DEFAULT_PROVIDER_STRATEGY="balanced"
export MONTHLY_BUDGET_USD=100.0

# All providers configured with business accounts
```

---

## Regional Compliance

### EU/GDPR Users
```bash
export REGIONAL_COMPLIANCE=true  # Automatically filters providers

# GDPR-Compliant providers:
# ✅ OpenAI, Mistral, Claude API, Google Gemini (partial)
# ❌ Groq (US-based, not GDPR compliant)
```

### US Users
```bash
# All providers available
# Groq recommended for speed + free tier
```

---

## Provider Selection Logic

The system automatically selects providers based on:

1. **Available API Keys**: Only providers with valid keys
2. **Regional Compliance**: GDPR filtering if enabled
3. **Strategy Preferences**: Cost vs speed vs quality
4. **Provider Health**: Availability and response time
5. **Budget Limits**: Respects monthly spending caps

**Selection Order Example (Cost-Optimized):**
```
1. Groq (free tier) → if available and under limits
2. OpenAI GPT-4o-mini → if API key present
3. Mistral Small → if EU user or API key present  
4. Claude API Haiku → if API key present
5. Google Gemini → if API key present
6. Claude Code → if installed and subscription active
7. Ollama → if installed and models available
```

---

## Troubleshooting Common Issues

### "No providers available"
```bash
# Check provider status
deflect-bot providers list

# Verify API keys
echo $OPENAI_API_KEY | head -c 10  # Should show key prefix

# Test individual providers
deflect-bot providers test
```

### Rate limiting errors
```bash
# Switch to different provider
export DEFAULT_PROVIDER_STRATEGY="balanced"

# Check rate limits
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/rate_limits
```

### High costs
```bash
# Enable cost tracking
export ENABLE_COST_TRACKING=true
export COST_ALERT_THRESHOLD=0.8  # Alert at 80%

# Use cheaper providers
export DEFAULT_PROVIDER_STRATEGY="cost_optimized"
```

### EU compliance issues
```bash
# Enable regional compliance
export REGIONAL_COMPLIANCE=true

# Check which providers are available
deflect-bot providers list --region EU
```

---

## Advanced Configuration

### Custom Provider Priority
```bash
# Override automatic selection
export PROVIDER_PRIORITY="groq,openai,mistral,claude_api"

# Force specific provider
export FORCE_PROVIDER="groq"
```

### Model Customization
```bash
# Use premium models
export OPENAI_LLM_MODEL="gpt-4o"
export CLAUDE_API_MODEL="claude-3-sonnet-20240229"
export GROQ_MODEL="llama-3.1-70b-versatile"
```

### Regional Optimization
```bash
# US East Coast
export PREFERRED_REGIONS="us-east-1,us-east-2"

# Europe
export PREFERRED_REGIONS="eu-west-1,eu-central-1"
```

---

## Next Steps

- **[Configuration Guide](configuration.md)** - Environment variables and settings
- **[Usage Guide](usage.md)** - CLI commands and usage patterns
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

For questions or issues with specific providers, check their official documentation:
- [OpenAI Documentation](https://platform.openai.com/docs)
- [Groq Documentation](https://console.groq.com/docs)
- [Anthropic Documentation](https://docs.anthropic.com/)
- [Google AI Documentation](https://ai.google.dev/)
- [Mistral Documentation](https://docs.mistral.ai/)