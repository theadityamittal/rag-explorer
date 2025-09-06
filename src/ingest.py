import os
from typing import List, Dict
from src.chunker import chunk_text, build_docs_from_files
from src.embeddings import embed_texts
from src.store import get_collection, reset_collection

def read_docs_from_folder(folder: str = "./docs") -> Dict[str, str]:
    texts = {}
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".md", ".txt")):
                p = os.path.join(root, f)
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    texts[p] = fh.read()
    return texts

def ingest_folder(folder: str = "./docs", chunk_size: int = 900, overlap: int = 150) -> int:
    # Reset collection for a clean rebuild each time (simplest for the weekend)
    coll = reset_collection()

    raw = read_docs_from_folder(folder)
    docs = build_docs_from_files(raw)

    chunk_texts = []
    metadatas = []

    for path, text in docs.items():
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, ch in enumerate(chunks):
            chunk_texts.append(ch)
            metadatas.append({"path": path, "chunk_id": i})

    # Compute embeddings with Ollama
    vecs = embed_texts(chunk_texts)

    # Add to Chroma with explicit embeddings
    # (Note: in chroma>=0.5 you can pass embeddings=vecs to coll.add)
    coll.add(documents=chunk_texts, metadatas=metadatas,
             embeddings=vecs,
             ids=[f"{m['path']}#{m['chunk_id']}" for m in metadatas])

    return len(chunk_texts)
