"""RAG compatibility bridge - bridges old answer_question function to new provider system."""

from typing import Dict, List, Optional
import logging
from ._path_helper import ensure_src_path

logger = logging.getLogger(__name__)


def answer_question(
    question: str, k: int = None, domains: Optional[List[str]] = None
) -> Dict:
    """
    Answer questions using RAG (Retrieval Augmented Generation).
    
    This function provides the same interface as src.core.rag.answer_question()
    but uses the new provider system for LLM generation when possible.
    
    Args:
        question: The question to answer
        k: Number of chunks to retrieve (defaults to MAX_CHUNKS)
        domains: Optional domain filtering
        
    Returns:
        Dict with keys: answer, citations, confidence
    """
    try:
        # Import settings for defaults
        try:
            from ..utils.settings import MAX_CHUNKS
        except ImportError:
            from src.utils.settings import MAX_CHUNKS
        
        if k is None:
            k = MAX_CHUNKS
            
        # Try using new provider system integrated with old retrieval
        from .retrieve_bridge import retrieve
        from .llm_bridge import llm_chat
        
        # Use old RAG logic but with new LLM bridge
        hits = retrieve(question, k=k, domains=domains)
        
        if not hits:
            return {
                "answer": "I don't have enough information in the docs to answer that.",
                "citations": [],
                "confidence": 0.0,
            }
        
        # Use the old confidence and formatting logic but new LLM
        conf = _calculate_confidence(hits, question)
        
        try:
            from ..utils.settings import ANSWER_MIN_CONF as MIN_CONF
        except ImportError:
            from src.utils.settings import ANSWER_MIN_CONF as MIN_CONF
            
        # Same refusal logic as old system
        if conf < MIN_CONF:
            return {
                "answer": "I don't have enough information in the docs to answer that.",
                "citations": _to_citations(hits, take=2),
                "confidence": conf,
            }
        
        # Format context and generate answer using new LLM system
        ctx = _format_context(hits)
        system_prompt = (
            "You are a support deflection assistant for product documentation.\n"
            "Use ONLY the provided Context to answer. If the Context contains any relevant\n"
            "instructions or details for the user's question, you MUST answer concisely\n"
            "(2–4 sentences) and include concrete commands/flags/paths when applicable.\n"
            "Refuse ONLY if the Context has no relevant information.\n"
            "Refusal text must be exactly:\n"
            "'I don't have enough information in the docs to answer that.'"
        )
        
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context (numbered citations):\n{ctx}\n\n"
            "Instructions:\n"
            "1) If any part of the Context is relevant, ANSWER. Keep it to 2–4 sentences.\n"
            "2) Prefer concrete steps and fenced code blocks for commands.\n"
            "3) Do not invent facts not in the Context.\n"
            "4) Refuse ONLY if no relevant information exists in the Context.\n"
        )
        
        # Use new LLM system via bridge
        ans = llm_chat(system_prompt=system_prompt, user_prompt=user_prompt)
        
        if not ans or not ans.strip():
            # Fallback logic same as old system
            if conf < MIN_CONF:
                return {
                    "answer": "I don't have enough information in the docs to answer that.",
                    "citations": _to_citations(hits, take=3),
                    "confidence": conf,
                }
            # Extract snippet from top chunk
            top = hits[0]["text"].strip()
            snippet = top[:240].split("\n\n")[0].strip()
            return {
                "answer": snippet,
                "citations": _to_citations(hits, take=3),
                "confidence": conf,
            }
        
        return {
            "answer": ans.strip(),
            "citations": _to_citations(hits, take=3),
            "confidence": conf,
        }
        
    except Exception as e:
        # Fallback to old system if new approach fails
        logger.warning(f"New RAG system failed: {e}, falling back to old system")
        try:
            from src.core.rag import answer_question as old_answer_question
            return old_answer_question(question, k=k, domains=domains)
        except ImportError:
            logger.error("Both new and old RAG systems unavailable")
            return {
                "answer": "I don't have enough information in the docs to answer that.",
                "citations": [],
                "confidence": 0.0,
            }
        except Exception as e2:
            logger.error(f"Old RAG system also failed: {e2}")
            return {
                "answer": "I don't have enough information in the docs to answer that.",
                "citations": [],
                "confidence": 0.0,
            }


def _calculate_confidence(hits: List[Dict], question: str) -> float:
    """Calculate confidence score for retrieved results."""
    try:
        # Import old confidence logic
        from src.core.rag import _confidence
        return _confidence(hits, question)
    except ImportError:
        # Simple fallback confidence calculation
        if not hits:
            return 0.0
        avg_distance = sum(h.get("distance", 1.0) for h in hits) / len(hits)
        return max(0.0, 1.0 - avg_distance)


def _format_context(hits: List[Dict]) -> str:
    """Format retrieved chunks into numbered context."""
    try:
        # Use old formatting logic
        from src.core.rag import _format_context
        return _format_context(hits)
    except ImportError:
        # Simple fallback formatting
        context_parts = []
        for i, hit in enumerate(hits, 1):
            text = hit["text"][:400]  # Limit chunk size
            context_parts.append(f"[{i}] {text}")
        return "\n\n".join(context_parts)


def _to_citations(hits: List[Dict], take: int = 3) -> List[Dict]:
    """Convert hits to citation format."""
    try:
        # Use old citation logic
        from src.core.rag import _to_citations
        return _to_citations(hits, take=take)
    except ImportError:
        # Simple fallback citation format
        citations = []
        for hit in hits[:take]:
            citation = {
                "text": hit["text"][:200] + "..." if len(hit["text"]) > 200 else hit["text"],
                "path": hit.get("meta", {}).get("path", "unknown"),
            }
            citations.append(citation)
        return citations