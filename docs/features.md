# Features Overview

## Core Behaviors
- **Grounded answers**: Only uses your provided documentation
- **Confident refusal**: Says "I don't have enough information" when evidence is weak
- **Citation tracking**: Shows exactly where answers come from
- **Confidence scoring**: Combines semantic similarity + keyword matching
- **Domain filtering**: Restrict answers to specific documentation sources

## 📚 **Document Processing**
- **Local files**: Markdown/TXT files in `./docs`
- **Web crawling**: Intelligent crawling with caching and robots.txt respect
- **Smart chunking**: Overlapping text chunks for better retrieval
- **Vector search**: Semantic similarity using local embeddings

## 🔍 **Question Answering**
- **Interactive CLI**: Conversational Q&A sessions with `deflect-bot ask`
- **RAG pipeline**: Retrieval-augmented generation with strict grounding
- **Confidence gating**: Configurable threshold for answer quality
- **Rich terminal output**: Beautiful tables, colors, and formatted responses
- **Dual interfaces**: Primary CLI + REST API for integrations
- **Performance metrics**: Response times and accuracy tracking

## 🌐 **Web Integration** 
- **HTTP caching**: ETag/Last-Modified support with content hashing
- **Domain whitelisting**: Security through allowed host lists
- **Force refresh**: Bypass caching when needed
- **Depth crawling**: Follow links with configurable depth limits

## 🛡️ **Reliability & Security**
- **No external APIs**: Everything runs locally with Ollama
- **Structured refusals**: Clear, consistent "don't know" responses  
- **Citation verification**: Track answer sources for validation
- **Error handling**: Graceful degradation and informative errors

## 🚀 **CLI Interface**

### Interactive Q&A
```bash
deflect-bot ask
# 🤖 Support Deflect Bot - Interactive Q&A Session
# Ask me anything about your documentation. Type 'end' to exit.
# 
# ❓ You: How do I configure the database?
# 🤖 Bot: [Intelligent answer with citations]
```

### Document Search
```bash
deflect-bot search "installation process"
# Beautiful formatted table with search results
```

### Web Crawling
```bash
deflect-bot crawl https://docs.python.org/3/ --depth 2 --max-pages 50
# Intelligent crawling with depth control
```

### System Management
```bash
deflect-bot status    # System health check
deflect-bot ping      # Test LLM connection
deflect-bot metrics   # Performance statistics
deflect-bot config    # View configuration
```

## 🔧 **Advanced Features**

### Confidence-Based Responses
The bot measures confidence using multiple signals:
- Semantic similarity scores
- Keyword matching
- Document relevance
- Context completeness

When confidence falls below the threshold, it gracefully refuses to answer.

### Citation Tracking
Every answer includes source citations:
```
📚 Sources:
  [1] ./docs/configuration.md:25 - Database configuration...
  [2] ./docs/setup.md:15 - Environment variables...
```

### Domain Filtering
Restrict answers to specific domains for focused responses:
```bash
deflect-bot ask --domains "docs.python.org,github.com"
```

### Intelligent Caching
- Web content cached with ETag/Last-Modified headers
- Content hashing for change detection
- Configurable cache expiration
- Force refresh capabilities

### Flexible Configuration
Extensive customization through environment variables:
- Confidence thresholds
- Context window sizes
- Crawling behavior
- Model selection
- Performance tuning

## 📊 **Performance Features**

### Metrics Tracking
```bash
deflect-bot metrics
# Shows response times, query counts, success rates
```

### Health Monitoring
```bash
deflect-bot status
# Comprehensive system health check
```

### Resource Management
- Configurable memory usage
- Efficient vector storage
- Optimized text chunking
- Background processing

## 🔌 **Integration Features**

### REST API
While the CLI is primary, a REST API is available for integrations:
```bash
uvicorn src.api.app:app --reload --port 8000
```

### Docker Support
```bash
docker build -t support-deflect-bot .
docker run -d -p 8000:8000 support-deflect-bot
```

### Automation Support
- Scriptable CLI interface
- Batch processing capabilities
- Silent/quiet modes
- JSON output formats

## 🎯 **Use Case Examples**

### Customer Support
- High confidence thresholds for customer-facing responses
- Domain restrictions to company documentation
- Citation tracking for answer verification

### Developer Documentation
- Multi-stack technology support
- Comprehensive context windows
- Integration with development workflows

### Academic Research
- Very strict confidence requirements
- Long-form document processing
- Detailed citation requirements

### API Documentation
- Structured response formatting
- Code example highlighting
- Version-specific documentation support

## 🔮 **Planned Features**

- **PDF Support**: Direct PDF document processing
- **Image Analysis**: Screenshots and diagram interpretation
- **Multi-language**: Documentation in multiple languages
- **Custom Models**: Support for specialized domain models
- **Collaboration**: Team sharing and collaboration features