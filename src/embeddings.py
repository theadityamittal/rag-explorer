import os
import ollama
from typing import List

EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Simple batch embedder calling Ollama per text.
    For small doc sets this is fine.
    """
    vectors = []
    for t in texts:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        vectors.append(resp["embedding"])
    return vectors

def embed_one(text: str) -> List[float]:
    return embed_texts([text])[0]
