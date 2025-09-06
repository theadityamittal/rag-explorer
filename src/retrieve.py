from typing import List, Dict, Optional
from src.embeddings import embed_one
from src.store import query_by_embedding

def retrieve(query: str, k: int = 5, domains: Optional[List[str]] = None) -> List[Dict]:
    qvec = embed_one(query)
    where = None
    if domains:
        # filter by metadata host
        where = {"host": {"$in": domains}}
    hits = query_by_embedding(qvec, k=k, where=where)
    return hits
