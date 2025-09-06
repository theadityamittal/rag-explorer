# Support Deflection Bot

Grounded answers to support questions using your docs. If the answer isn’t in the docs, it **refuses** rather than hallucinating.

## Why this exists

Most “docbots” answer confidently but make things up. This bot:

* retrieves the most relevant **chunks** from your docs (vector search),
* asks a local LLM to answer **only from those chunks** (RAG),
* **refuses** with a clear sentence when evidence is weak,
* returns **citations**, **confidence**, and basic **metrics**.

---

## Features

* **RAG** over your Markdown/TXT docs
* **Citations** (file + preview of the source chunk)
* **Refusal** when not grounded:
  `I don’t have enough information in the docs to answer that.`
* **Confidence** score (semantic distance + keyword overlap)
* **Metrics** endpoint (counts, p50/p95 latency)
* **Batch eval** with a tiny gold set (accuracy for answers/refusals)
* **All local**: uses **Ollama** for chat + embeddings

---

## Stack

* **API**: FastAPI (Python 3.11)
* **LLM**: Ollama (`llama3.1`)
* **Embeddings**: Ollama (`nomic-embed-text`)
* **Vector store**: Chroma (persistent)
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
│  ├─ metrics.py       # simple meters & summaries
│  ├─ batch.py         # batch ask for eval
│  ├─ run_eval.py      # lightweight evaluator
│  └─ settings.py      # environment-driven config
├─ docs/               # your documentation lives here
├─ data/eval/          # tiny gold set for scoring
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

## Endpoints

* `GET  /healthz` → liveness check
* `POST /reindex` → rebuilds the vector DB from `./docs`
* `POST /search` → raw retrieval preview (query → top chunks)
* `POST /ask` → RAG answer `{answer, citations[], confidence}`
* `POST /batch_ask` → bulk questions for eval
* `GET  /metrics` → `{ask: {count,p50_ms,p95_ms}, search: {...}}`

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
ANSWER_MIN_CONF=0.25     # raise to be stricter
MAX_CHUNKS=5
MAX_CHARS_PER_CHUNK=800
```

**Confidence gating:** If `confidence < ANSWER_MIN_CONF`, the bot refuses.

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

## How it works (short)

1. **Ingest:** read `./docs/*.md|*.txt` → chunk (overlap)
2. **Embed:** `nomic-embed-text` via Ollama → vector DB (Chroma)
3. **Retrieve:** top-k chunks for a question
4. **Answer:** LLM (`llama3.1`) instructed to use **only** retrieved context
5. **Refuse:** below `ANSWER_MIN_CONF` or if LLM indicates insufficient context
6. **Return:** answer + top citations + confidence

---

## Tuning tips

* Raise `ANSWER_MIN_CONF` (e.g., `0.35`) to reduce risky answers.
* Adjust `MAX_CHUNKS` and `chunk_size/overlap` in `ingest.py` for your docs.
* Keep docs clean: headings and bullet lists retrieve better.
* Add an **“unanswerable”** section to test refusal behavior regularly.

---

## Limitations & next steps

* No UI (yet) — add a small web front-end that hits `/ask`.
* No auth on endpoints — add an API key or gateway before exposing.
* Basic confidence heuristic — consider adding a reranker or entropy signals.
* Minimal logging — connect to Prometheus/Grafana if deploying.

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
