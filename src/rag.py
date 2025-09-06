from typing import Dict, List
import re

from src.retrieve import retrieve
from src.llm_local import llm_chat

# --- Tunable knobs ---
MAX_CHUNKS = 5
MAX_CHARS_PER_CHUNK = 800

# Chroma "distance" can be relatively large depending on model/metric.
# Make these lenient so obvious matches aren't refused.
STRICT_DIST = 1.20
LOOSE_DIST  = 1.40

SYSTEM_PROMPT = (
    "You are a support deflection assistant. "
    "Answer ONLY using the provided Context. "
    "If the Context does not contain enough information, reply exactly: "
    "'I don’t have enough information in the docs to answer that.' "
    "Keep answers concise and actionable."
)

_STOP = {
    "the","a","an","and","or","if","to","of","for","in","on","at","by","with",
    "is","are","be","was","were","it","this","that","as","from","into","out",
    "do","does","did","how","what","why","where","when","which","who","whom",
}

def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " …"

def _format_context(hits: List[Dict]) -> str:
    lines = []
    for i, h in enumerate(hits[:MAX_CHUNKS], start=1):
        preview = _trim(h["text"], MAX_CHARS_PER_CHUNK)
        path = h["meta"].get("path", "unknown")
        lines.append(f"[{i}] ({path})\n{preview}")
    return "\n\n".join(lines)

def _best_distance(hits: List[Dict]):
    ds = [h.get("distance") for h in hits if isinstance(h.get("distance"), (int, float))]
    return min(ds) if ds else None

def _avg_top3_distance(hits: List[Dict]):
    ds = [h.get("distance") for h in hits if isinstance(h.get("distance"), (int, float))]
    ds = sorted(ds)[:3]
    return (sum(ds)/len(ds)) if ds else None

def _tokens(s: str):
    toks = re.findall(r"[a-z0-9]+", s.lower())
    return [t for t in toks if len(t) > 2 and t not in _STOP]

def _keyword_overlap_ok(question: str, text: str, min_overlap: int = 2) -> bool:
    q = set(_tokens(question))
    t = set(_tokens(text))
    return len(q.intersection(t)) >= min_overlap

def _should_refuse(hits: List[Dict]) -> bool:
    if not hits:
        return True
    b = _best_distance(hits)
    a3 = _avg_top3_distance(hits)
    # Smaller distance => closer match. If best is still quite large AND avg is large, refuse.
    if b is not None and b > STRICT_DIST:
        if a3 is None or a3 > LOOSE_DIST:
            return True
    return False

def _confidence(hits: List[Dict]) -> float:
    # Convert distance to a 0..1-ish confidence; lenient scaling.
    b = _best_distance(hits)
    if b is None:
        return 0.6
    denom = max(LOOSE_DIST, 1e-6)
    raw = 1.0 - min(1.0, b / denom)
    return max(0.0, min(1.0, raw))

def _to_citations(hits: List[Dict], take: int = 3) -> List[Dict]:
    out = []
    for i, h in enumerate(hits[:take], start=1):
        out.append({
            "rank": i,
            "path": h["meta"].get("path"),
            "chunk_id": h["meta"].get("chunk_id"),
            "preview": _trim(h["text"], 200)
        })
    return out

def answer_question(question: str, k: int = MAX_CHUNKS) -> Dict:
    # 1) retrieve
    hits = retrieve(question, k=k)

    # 2) refusal? allow keyword-overlap escape on the top hit
    refuse = _should_refuse(hits)
    if refuse and hits:
        top = hits[0]
        if _keyword_overlap_ok(question, top["text"]):
            refuse = False

    if refuse:
        return {
            "answer": "I don’t have enough information in the docs to answer that.",
            "citations": _to_citations(hits, take=2),
            "confidence": _confidence(hits)
        }

    # 3) build context block from top chunks
    ctx = _format_context(hits)

    # 4) compose user prompt (strict grounding)
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{ctx}\n\n"
        "Instructions: Use ONLY the Context above. "
        "If the answer isn't clearly supported, reply with the exact refusal sentence."
    )

    # 5) ask local LLM
    ans = llm_chat(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)

    return {
        "answer": ans.strip(),
        "citations": _to_citations(hits, take=3),
        "confidence": _confidence(hits)
    }
