from fastapi import FastAPI
from pydantic import BaseModel
from src.llm_local import llm_chat, llm_echo

app = FastAPI(title="Support Deflection Bot", version="0.0.2")

class AskRequest(BaseModel):
    question: str

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    # Placeholder: we’ll plug retrieval + LLM here later.
    return {
        "answer": "The bot is not wired up yet—coming soon!",
        "citations": [],
        "confidence": 0.0
    }

@app.get("/llm_ping")
def llm_ping():
    text = llm_echo("PONG")
    return {"model": "ollama", "reply": text}