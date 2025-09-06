from typing import List, Dict
from src.rag import answer_question

def batch_ask(questions: List[str]) -> List[Dict]:
    """
    Calls the RAG pipeline for a list of questions and returns raw results.
    Each result: {"question": q, "answer": ..., "citations": [...], "confidence": ...}
    """
    out = []
    for q in questions:
        res = answer_question(q)
        res["question"] = q
        out.append(res)
    return out
