import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from src.llm_local import llm_chat, llm_echo
from src.ingest import ingest_folder
from src.retrieve import retrieve
from src.rag import answer_question
from src.metrics import Meter

app = FastAPI(title="Support Deflection Bot", version="0.0.5")

# meters
ASK_METER = Meter()
SEARCH_METER = Meter()

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
    t0 = time.perf_counter()
    try:
        hits = retrieve(req.query, k=req.k)
        out = {
            "query": req.query,
            "results": [
                {
                    "text": h["text"][:400],
                    "path": h["meta"].get("path"),
                    "chunk_id": h["meta"].get("chunk_id"),
                    "distance": h.get("distance"),
                }
                for h in hits
            ],
        }
        return out
    finally:
        SEARCH_METER.observe(time.perf_counter() - t0)

@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.perf_counter()
    try:
        result = answer_question(req.question)
        return result
    finally:
        ASK_METER.observe(time.perf_counter() - t0)

@app.get("/metrics")
def metrics():
    return {
        "ask": ASK_METER.summary(),
        "search": SEARCH_METER.summary(),
        "version": app.version,
    }

@app.get("/llm_ping")
def llm_ping():
    text = llm_echo("PONG")
    return {"model": "ollama", "reply": text}
