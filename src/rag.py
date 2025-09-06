from typing import Dict, List
import math

from src.retrieve import retrieve
from src.llm_local import llm_chat

# --- Tunable knobs for a weekend build ---
MAX_CHUNKS = 5            # how many chunks to stuff into the prompt
MAX_CHARS_PER_CHUNK = 800 # keep context tight to avoid rambling
STRICT_DIST = 0.35        # if best distance is worse than this, probably weak
LOOSE_DIST = 0.50         # average of top-3 worse than this → definitely weak

SYSTEM_PROMPT = (
    "You are a support deflection assistant. "
    "Answer ONLY using the provided Context. "
    "If the Context does not contain enough information, reply exactly: "
    "'I don’t have enough information in the docs to answer that.' "
    "Keep answers concise and actionable."
)

def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …"

def _format_context(hits: List[Dict]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
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
    if not ds:
        return None
    return sum(ds) / len(ds)

def _should_refuse(hits: List[Dict]) -> bool:
    if not hits:
        return True
    b = _best_distance(hits)
    a3 = _avg_top3_distance(hits)
    # Smaller distance = more similar. If we have numbers and they are large, refuse.
    if b is not None and b > STRICT_DIST:
        # if even the best is weak AND average is also weak, refuse
        if a3 is None or a3 > LOOSE_DIST:
            return True
    return False

def _confidence(hits: List[Dict]) -> float:
    """
    Heuristic: 1 - normalize(best_distance, 0..LOOSE_DIST+epsilon).
    Clipped to [0,1]. If no distance, return 0.5.
    """
    b = _best_distance(hits)
    if b is None:
        return 0.5
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

    # 2) refusal if weak evidence
    if _should_refuse(hits):
        return {
            "answer": "I don’t have enough information in the docs to answer that.",
            "citations": _to_citations(hits, take=2),
            "confidence": _confidence(hits)
        }

    # 3) build context block
    ctx = _format_context(hits)

    # 4) compose user prompt
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{ctx}\n\n"
        "Instructions: Use ONLY the Context above. "
        "If the answer isn't clearly supported, reply with the exact refusal sentence."
    )

    # 5) ask the LLM (local)
    ans = llm_chat(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt
    )

    return {
        "answer": ans.strip(),
        "citations": _to_citations(hits, take=3),
        "confidence": _confidence(hits)
    }
