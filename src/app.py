from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from src.llm_local import llm_chat, llm_echo
from src.ingest import ingest_folder
from src.retrieve import retrieve

app = FastAPI(title="Support Deflection Bot", version="0.0.3")

class AskRequest(BaseModel):
    question: str

class SearchRequest(BaseModel):
    query: str
    k: int = 5

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/reindex")
def reindex():
    n = ingest_folder("./docs")
    return {"chunks_indexed": n}

@app.post("/search")
def search(req: SearchRequest):
    hits = retrieve(req.query, k=req.k)
    # Return only essential fields
    return {
        "query": req.query,
        "results": [
            {
                "text": h["text"][:400],  # trim preview
                "path": h["meta"].get("path"),
                "chunk_id": h["meta"].get("chunk_id"),
                "distance": h.get("distance"),
            }
            for h in hits
        ],
    }

@app.post("/ask")
def ask(req: AskRequest):
    # We'll plug full RAG here in the next step
    return {
        "answer": "Retrieval not yet used here—coming next!",
        "citations": [],
        "confidence": 0.0
    }

@app.get("/llm_ping")
def llm_ping():
    text = llm_echo("PONG")
    return {"model": "ollama", "reply": text}
