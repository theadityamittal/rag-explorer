import logging
import os
import uuid
from typing import Dict, List, Optional

import chromadb

from src.utils.settings import CHROMA_COLLECTION, CHROMA_DB_PATH


def get_client() -> chromadb.PersistentClient:
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        return chromadb.PersistentClient(path=CHROMA_DB_PATH)
    except Exception as e:
        logging.error(f"Failed to create ChromaDB client: {e}")
        raise ConnectionError(f"Database connection failed: {e}") from e


def get_collection(client=None):
    try:
        client = client or get_client()
        # IMPORTANT: set HNSW space to cosine for text embeddings
        return client.get_or_create_collection(
            name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        logging.error(f"Failed to get/create collection '{CHROMA_COLLECTION}': {e}")
        raise ConnectionError(f"Database collection access failed: {e}") from e


def upsert_chunks(
    chunks: List[str],
    metadatas: List[Dict],
    ids: Optional[List[str]] = None,
):
    """
    Add/overwrite chunks with precomputed ids. If ids None, create uuids.
    NOTE: We embed outside chroma (we'll pass embeddings explicitly later).
    """
    try:
        coll = get_collection()
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in chunks]
        coll.add(documents=chunks, metadatas=metadatas, ids=ids)
    except Exception as e:
        logging.error(f"Failed to upsert {len(chunks)} chunks: {e}")
        raise ConnectionError(f"Failed to store chunks in database: {e}") from e


def query_by_embedding(
    query_embedding: List[float], k: int = 5, where: Optional[dict] = None
):
    try:
        coll = get_collection()
        res = coll.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        # Standardize result shape
        out = []
        if res and res.get("documents"):
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res.get("distances", [[None] * len(docs)])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                out.append({"text": doc, "meta": meta, "distance": dist})
        return out
    except Exception as e:
        logging.error(f"Failed to query embeddings: {e}")
        raise ConnectionError(f"Database query failed: {e}") from e


def reset_collection():
    client = get_client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except ValueError:
        # Collection doesn't exist, which is fine
        pass
    # Recreate with cosine space
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
