from typing import List, Dict
from src.embeddings import embed_one
from src.store import query_by_embedding

def retrieve(query: str, k: int = 5) -> List[Dict]:
    qvec = embed_one(query)
    hits = query_by_embedding(qvec, k=k)
    return hits
