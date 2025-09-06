# Support Deflection Bot

**What it does**  
Grounded answers to support questions using your docs. If the answer isn’t in the docs, it refuses.

**Stack**  
- FastAPI + Python 3.11  
- Ollama (local LLM: `llama3.1`, embeddings: `nomic-embed-text`)  
- Chroma (vector store)  
- Simple eval harness (JSONL + batch scorer)  

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1
ollama pull nomic-embed-text
uvicorn src.app:app --reload
# Build index
curl -X POST http://127.0.0.1:8000/reindex
# Ask
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question":"How do I enable debug mode?"}'
