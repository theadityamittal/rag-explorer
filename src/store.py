import os
import uuid
import chromadb
from typing import List, Dict, Tuple, Optional
from chromadb.config import Settings
from src.settings import CHROMA_DB_PATH as _DB_PATH, CHROMA_COLLECTION as _COLL_NAME

_DB_PATH = _DB_PATH
_COLL_NAME = _COLL_NAME

def get_client() -> chromadb.PersistentClient:
    os.makedirs(_DB_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=_DB_PATH)

def get_collection(client=None):
    client = client or get_client()
    return client.get_or_create_collection(name=_COLL_NAME)

def upsert_chunks(
    chunks: List[str],
    metadatas: List[Dict],
    ids: Optional[List[str]] = None,
):
    """
    Add/overwrite chunks with precomputed ids. If ids None, create uuids.
    NOTE: We embed outside chroma (we'll pass embeddings explicitly later).
    """
    coll = get_collection()
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in chunks]
    coll.add(documents=chunks, metadatas=metadatas, ids=ids)

def query_by_embedding(query_embedding: List[float], k: int = 5):
    coll = get_collection()
    res = coll.query(query_embeddings=[query_embedding], n_results=k, include=["documents", "metadatas", "distances"])
    # Standardize result shape
    out = []
    if res and res.get("documents"):
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[None]*len(docs)])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({"text": doc, "meta": meta, "distance": dist})
    return out

def reset_collection():
    client = get_client()
    try:
        client.delete_collection(_COLL_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(_COLL_NAME)
