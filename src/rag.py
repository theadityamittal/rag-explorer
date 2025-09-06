import os
import re
from typing import Dict, List, Optional
from src.retrieve import retrieve
from src.llm_local import llm_chat
from src.settings import ANSWER_MIN_CONF as MIN_CONF, MAX_CHUNKS, MAX_CHARS_PER_CHUNK

# Refuse if final confidence < this threshold (override via env)
MIN_CONF = MIN_CONF

SYSTEM_PROMPT = (
    "You are a support deflection assistant for product documentation.\n"
    "Use ONLY the provided Context to answer. If the Context contains any relevant\n"
    "instructions or details for the user’s question, you MUST answer concisely\n"
    "(2–4 sentences) and include concrete commands/flags/paths when applicable.\n"
    "Refuse ONLY if the Context has no relevant information.\n"
    "Refusal text must be exactly:\n"
    "'I don’t have enough information in the docs to answer that.'"
)

_STOP = {
    "the","a","an","and","or","if","to","of","for","in","on","at","by","with",
    "is","are","be","was","were","it","this","that","as","from","into","out",
    "do","does","did","how","what","why","where","when","which","who","whom",
}

def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " …"

def _stem_simple(word: str) -> str:
    """Basic stemming for common suffixes"""
    if len(word) <= 3:
        return word
    # Handle common plural and verb endings
    if word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
        return word[:-1]
    if word.endswith('ing') and len(word) > 6:
        return word[:-3]
    if word.endswith('ed') and len(word) > 5:
        return word[:-2]
    return word

def _tokens(s: str):
    toks = re.findall(r"[a-z0-9]+", s.lower())
    # Apply basic stemming and filter
    stemmed = [_stem_simple(t) for t in toks if len(t) > 2 and t not in _STOP]
    return stemmed

def _keyword_overlap(question: str, text: str) -> int:
    q = set(_tokens(question))
    t = set(_tokens(text))
    return len(q.intersection(t))

def _overlap_ratio(question: str, text: str) -> float:
    q = set(_tokens(question))
    if not q:
        return 0.0
    inter = _keyword_overlap(question, text)
    # Cap denominator to avoid punishing short questions too much
    denom = min(5, len(q))
    return min(1.0, inter / max(1, denom))

def _format_context(hits: List[Dict]) -> str:
    lines = []
    for i, h in enumerate(hits[:MAX_CHUNKS], start=1):
        preview = _trim(h["text"], MAX_CHARS_PER_CHUNK)
        path = h["meta"].get("path", "unknown")
        lines.append(f"[{i}] ({path})\n{preview}")
    return "\n\n".join(lines)

def _similarity_from_distance(d) -> float:
    # map distance to (0,1]; smaller distance → closer to 1.0
    if not isinstance(d, (int, float)):
        return 0.5
    return 1.0 / (1.0 + max(0.0, d))

def _confidence(hits: List[Dict], question: str) -> float:
    if not hits:
        return 0.0
    # top hit signals most
    top = hits[0]
    d = top.get("distance")
    sim = _similarity_from_distance(d)      # 0..1 (higher is better)
    ovl = _overlap_ratio(question, top["text"])  # 0..1 (higher is better)
    # blend → slightly favor similarity
    conf = 0.6 * sim + 0.4 * ovl
    return round(max(0.0, min(1.0, conf)), 3)

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

def answer_question(question: str, k: int = MAX_CHUNKS, domains: Optional[List[str]] = None) -> Dict:
    hits = retrieve(question, k=k, domains=domains)

    # --- Compute final confidence from distance + keyword overlap
    conf = _confidence(hits, question)

    # New behavior: refuse ONLY if conf < MIN_CONF.
    if conf < MIN_CONF:
        return {
            "answer": "I don’t have enough information in the docs to answer that.",
            "citations": _to_citations(hits, take=2),
            "confidence": conf
        }

    ctx = _format_context(hits)
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context (numbered citations):\n{ctx}\n\n"
        "Instructions:\n"
        "1) If any part of the Context is relevant, ANSWER. Keep it to 2–4 sentences.\n"
        "2) Prefer concrete steps and fenced code blocks for commands.\n"
        "3) Do not invent facts not in the Context.\n"
        "4) Refuse ONLY if no relevant information exists in the Context.\n"
    )


    ans = llm_chat(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)

    if not ans or not ans.strip():
        # No text came back; fall back to refusal only if confidence is low
        if conf < MIN_CONF:
            return {
                "answer": "I don’t have enough information in the docs to answer that.",
                "citations": _to_citations(hits, take=3),
                "confidence": conf
            }
        # Otherwise synthesize a tiny extractive answer from the top chunk
        top = hits[0]["text"].strip()
        snippet = top[:240].split("\n\n")[0].strip()
        return {
            "answer": snippet,
            "citations": _to_citations(hits, take=3),
            "confidence": conf
        }


    return {
        "answer": ans.strip(),
        "citations": _to_citations(hits, take=3),
        "confidence": conf
    }

