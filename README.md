# Support Deflection Bot

Grounded answers to support questions using your local docs AND web documentation. Built with intelligent web crawling, semantic search, and strict confidence gating. If the answer isn't in the docs, it **refuses** rather than hallucinating.

## Why this exists

Most "docbots" answer confidently but make things up. This bot:

* **crawls and indexes** web documentation (with domain whitelisting and caching),
* retrieves the most relevant **chunks** from your docs (semantic vector search),
* asks a local LLM to answer **only from those chunks** (RAG),
* **refuses** with a clear sentence when evidence is weak,
* returns **citations**, **confidence scores**, and performance **metrics**,
* provides **domain filtering** for security and relevance.

---

## Features

* **RAG** over your Markdown/TXT docs + **web documentation**
* **Smart web crawling** with domain whitelisting and robots.txt respect
* **HTTP caching** with ETag/Last-Modified support and content hashing
* **Force re-indexing** to bypass 304 Not Modified responses
* **Domain filtering** for security and relevance (e.g., only Python docs)
* **Citations** (file + preview of the source chunk)
* **Refusal** when not grounded:
  `I don't have enough information in the docs to answer that.`
* **Confidence** score (semantic distance + keyword overlap with stemming)
* **Metrics** endpoint (counts, p50/p95 latency)
* **Batch eval** with a tiny gold set (accuracy for answers/refusals)
* **All local**: uses **Ollama** for chat + embeddings (no external API calls)

---

## Stack

* **API**: FastAPI (Python 3.11)
* **LLM**: Ollama (`llama3.1`)
* **Embeddings**: Ollama (`nomic-embed-text`)
* **Vector store**: Chroma (persistent)
* **Web crawler**: BeautifulSoup + requests with intelligent caching
* **Caching**: JSON-based crawl cache with ETag/Last-Modified/content hashing
* **Security**: Domain whitelisting and robots.txt compliance
* **Eval**: JSONL + batch scorer
* **Config**: `.env` + `src/settings.py`
* **Container**: Dockerfile provided

---

## Project layout

```
.
├─ src/
│  ├─ app.py           # FastAPI endpoints
│  ├─ rag.py           # retrieval + answer + refusal logic  
│  ├─ retrieve.py      # vector search
│  ├─ embeddings.py    # Ollama embeddings
│  ├─ llm_local.py     # Ollama chat wrapper
│  ├─ store.py         # Chroma wrapper (persistent)
│  ├─ chunker.py       # naive chunker with overlap
│  ├─ ingest.py        # crawl docs/ → chunks → embeddings → index
│  ├─ web_ingest.py    # web crawling with caching and indexing
│  ├─ metrics.py       # simple meters & summaries
│  ├─ batch.py         # batch ask for eval
│  ├─ run_eval.py      # lightweight evaluator
│  └─ settings.py      # environment-driven config
├─ docs/               # your documentation lives here
├─ data/
│  ├─ eval/            # tiny gold set for scoring
│  └─ crawl_cache.json # HTTP cache for web crawling (created at runtime)
├─ chroma_db/          # persistent vector DB (created at runtime)
├─ requirements.txt
├─ Dockerfile
├─ .env.example
└─ README.md
```

---

## Quickstart (local)

1. **Prereqs**

* Python 3.11+
* [Ollama](https://ollama.com/) installed and running:

  ```bash
  ollama pull llama3.1
  ollama pull nomic-embed-text
  ```

2. **Install & run**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (optional) copy .env.example to .env and tweak knobs
uvicorn src.app:app --reload
```

3. **Index docs & ask**

```bash
# index files in ./docs (md/txt)
curl -X POST http://127.0.0.1:8000/reindex

# ask something covered by your docs
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How do I enable debug mode?"}'
```

---

## Quickstart (web crawling)

**Index specific URLs:**
```bash
# Crawl specific documentation pages
curl -X POST http://127.0.0.1:8000/crawl \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://docs.python.org/3/library/venv.html"]}'

# Force re-index to bypass HTTP 304 caching
curl -X POST http://127.0.0.1:8000/crawl \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://docs.python.org/3/library/venv.html"], "force": true}'
```

**Crawl with depth (follows links):**
```bash
# BFS crawl starting from seed URLs
curl -X POST http://127.0.0.1:8000/crawl_depth \
  -H "Content-Type: application/json" \
  -d '{"seeds": ["https://docs.python.org/3/"], "depth": 2, "max_pages": 50}'

# Crawl default configured sites
curl -X POST http://127.0.0.1:8000/crawl_default
```

**Ask with domain filtering:**
```bash
# Only search within specific domains for security/relevance
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How do I make a virtual environment?", "domains":["docs.python.org"]}'
```

---

## Endpoints

**Core Operations:**
* `GET  /healthz` → liveness check
* `POST /reindex` → rebuilds the vector DB from `./docs`
* `POST /search` → raw retrieval preview (query → top chunks)
* `POST /ask` → RAG answer `{answer, citations[], confidence}` with optional domain filtering
* `POST /batch_ask` → bulk questions for eval
* `GET  /metrics` → `{ask: {count,p50_ms,p95_ms}, search: {...}}`

**Web Crawling:**
* `POST /crawl` → index specific URLs `{"urls": [...], "force": false}`
* `POST /crawl_depth` → BFS crawling `{"seeds": [...], "depth": 1, "max_pages": 30, "force": false}`
* `POST /crawl_default` → crawl preconfigured seed URLs
* `GET  /llm_ping` → test local LLM connectivity

---

## Configuration

Copy `.env.example` → `.env` and adjust:

```
APP_NAME=Support Deflection Bot
APP_VERSION=0.1.0

# Ollama (local)
OLLAMA_MODEL=llama3.1
OLLAMA_EMBED_MODEL=nomic-embed-text
# If app runs in Docker and Ollama stays on your host:
# OLLAMA_HOST=http://host.docker.internal:11434

# Chroma
CHROMA_DB_PATH=./chroma_db
CHROMA_COLLECTION=knowledge_base

# RAG knobs
ANSWER_MIN_CONF=0.20     # raise to be stricter (updated from 0.25)
MAX_CHUNKS=5
MAX_CHARS_PER_CHUNK=800

# Web crawling
ALLOW_HOSTS=docs.python.org,packaging.python.org,pip.pypa.io,virtualenv.pypa.io
DEFAULT_SEEDS=https://docs.python.org/3/faq/index.html,https://docs.python.org/3/library/venv.html
CRAWL_DEPTH=1
CRAWL_MAX_PAGES=40
CRAWL_SAME_DOMAIN=true
CRAWL_CACHE_PATH=./data/crawl_cache.json
CRAWL_USER_AGENT=SupportDeflectBot/0.1 (+https://example.local; contact: you@example.com)
DOCS_FOLDER=./docs
```

**Confidence gating:** If `confidence < ANSWER_MIN_CONF`, the bot refuses.
**Domain security:** Only URLs matching `ALLOW_HOSTS` are crawled.

---

## Evaluation

1. Gold set lives in `data/eval/gold.jsonl` (extend it as you like):

   * `"type":"answer"` + `must_include: ["needle1","needle2"]`
   * `"type":"refusal"` (expects the exact refusal sentence)

2. Run:

```bash
python -m src.run_eval
```

Outputs a summary and `data/eval/results_<timestamp>.csv` with per-question status.

---

## Monitoring

```bash
curl http://127.0.0.1:8000/metrics
# => {"ask":{"count":..,"p50_ms":..,"p95_ms":..}, "search": {...}, "version":"..."}
```

---

## Docker

Build:

```bash
docker build -t support-deflect-bot .
```

Run (macOS/Windows, Ollama on host):

```bash
docker run --rm -p 8000:8000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e ANSWER_MIN_CONF=0.3 \
  -v $(pwd)/docs:/app/docs \
  -v $(pwd)/chroma_db:/app/chroma_db \
  support-deflect-bot
```

Linux (host network differs):

```bash
docker run --rm -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v $(pwd)/docs:/app/docs \
  -v $(pwd)/chroma_db:/app/chroma_db \
  support-deflect-bot
```

---

## How it works (visual)

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Web Pages   │ →  │  Processing  │ →  │  Vector DB  │
│ Local Docs  │    │ (chunk+embed)│    │ (ChromaDB)  │
└─────────────┘    └──────────────┘    └─────────────┘
                                                ↓
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Answer    │ ←  │     LLM      │ ←  │   Search    │
│ + Citations │    │  (Generate)  │    │ (Retrieve)  │
└─────────────┘    └──────────────┘    └─────────────┘
```

**Step-by-step process:**

1. **Ingest:** Crawl web docs OR read local `./docs/*.md|*.txt` → chunk with overlap
2. **Embed:** `nomic-embed-text` via Ollama → 768-dimensional vectors → store in ChromaDB  
3. **Query:** User question → embedding → vector similarity search (cosine)
4. **Retrieve:** Top-k most similar chunks + domain filtering
5. **Confidence:** Calculate score from semantic similarity + keyword overlap
6. **Answer:** If confident enough, send context to LLM (`llama3.1`) with strict instructions
7. **Refuse:** If `confidence < ANSWER_MIN_CONF` or LLM indicates insufficient context
8. **Return:** Answer + citations + confidence score

---

## Tuning tips

* Raise `ANSWER_MIN_CONF` (e.g., `0.35`) to reduce risky answers.
* Adjust `MAX_CHUNKS` and `chunk_size/overlap` in `ingest.py` for your docs.
* Keep docs clean: headings and bullet lists retrieve better.
* Add an **"unanswerable"** section to test refusal behavior regularly.

---

## Troubleshooting

**Domain filtering returns no results:**
- Use `"force": true` when crawling to update host metadata in existing chunks
- Check that `ALLOW_HOSTS` includes your target domains  
- Verify chunks exist with `/search` endpoint before trying domain filtering

**Low confidence scores preventing answers:**  
- Lower `ANSWER_MIN_CONF` (try 0.15-0.20 instead of 0.25)
- Ensure your docs contain actual command examples, not just conceptual descriptions
- Use more specific questions that match your indexed content
- Check if stemming is working: "environment" vs "environments" should now match

**HTTP 304 preventing content updates:**
- Use `"force": true` parameter in `/crawl` requests to bypass caching
- Check crawl cache at `./data/crawl_cache.json` and delete if needed
- Verify `ETag` and `Last-Modified` headers are being handled correctly

**"Empty reply from server" or timeouts:**
- Check that Ollama is running: `ollama list`
- Test LLM connectivity: `curl http://127.0.0.1:8000/llm_ping`
- Reduce batch sizes if processing large documents
- Check logs for embedding or vector search errors

**Answers seem wrong or hallucinated:**
- Verify citations point to correct source chunks
- Increase `ANSWER_MIN_CONF` to be more conservative  
- Check that retrieved chunks actually contain relevant information
- Review system prompt in `src/rag.py` for instruction clarity

---

## Limitations & next steps

* No UI (yet) — add a small web front-end that hits `/ask`.
* No auth on endpoints — add an API key or gateway before exposing.
* Basic confidence heuristic — consider adding a reranker or entropy signals.
* Minimal logging — connect to Prometheus/Grafana if deploying.

---

## Advanced Usage

**Batch Operations:**
```bash
# Process multiple questions at once
curl -X POST http://127.0.0.1:8000/batch_ask \
  -H "Content-Type: application/json" \
  -d '{"questions": ["How do I enable debug mode?", "Where are logs stored?", "How do I reset configuration?"]}'
```

**Cache Management:**
```bash
# Clear crawl cache to force fresh downloads
rm ./data/crawl_cache.json

# Check cache contents (with jq for pretty formatting)
cat ./data/crawl_cache.json | jq

# Or without jq (raw JSON)
cat ./data/crawl_cache.json
```

**Custom Confidence Thresholds:**
```bash
# Run with higher confidence for production
ANSWER_MIN_CONF=0.35 uvicorn src.app:app --reload

# Or set in .env file for persistent changes
echo "ANSWER_MIN_CONF=0.30" >> .env
```

**Domain-Specific Knowledge Bases:**
```bash
# Python documentation only
curl -X POST http://127.0.0.1:8000/ask \
  -d '{"question":"How do I create a virtual environment?", "domains":["docs.python.org"]}'

# Package management docs only  
curl -X POST http://127.0.0.1:8000/ask \
  -d '{"question":"How do I install dependencies?", "domains":["packaging.python.org", "pip.pypa.io"]}'
```

**Performance Monitoring:**
```bash
# Real-time metrics (requires jq: brew install jq)
watch -n 2 'curl -s http://127.0.0.1:8000/metrics | jq'

# Check specific endpoint performance
curl -s http://127.0.0.1:8000/metrics | jq '.ask.p95_ms'

# Without jq
curl -s http://127.0.0.1:8000/metrics
```

---

## License

MIT

---

## Acknowledgements

* Ollama team (local LLM runtime)
* Chroma DB (vector store)

---

**Questions / improvements?**
Open an issue or tweak the gold set, thresholds, and chunking to fit your docs.
