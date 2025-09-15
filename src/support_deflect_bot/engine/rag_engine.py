"""Unified RAG Engine for Support Deflect Bot."""

import logging
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..core.providers import get_default_registry, ProviderType, ProviderError, ProviderUnavailableError
from ..utils.settings import (
    ANSWER_MIN_CONF as MIN_CONF,
    MAX_CHARS_PER_CHUNK,
    MAX_CHUNKS,
    CHROMA_COLLECTION,
    CHROMA_DB_PATH
)


class UnifiedRAGEngine:
    """
    Unified RAG engine that orchestrates the entire retrieval-augmented generation pipeline.
    Provides document search, question answering, confidence calculation, and metrics collection.
    """
    
    def __init__(self, provider_registry=None):
        """Initialize RAG engine with provider registry and configuration."""
        self.provider_registry = provider_registry or get_default_registry()
        self.metrics = {
            "queries_processed": 0,
            "successful_answers": 0,
            "refusals": 0,
            "provider_failures": 0,
            "average_confidence": 0.0,
            "last_query_time": None
        }
        
        # System prompt for RAG responses
        self.system_prompt = (
            "You are a support deflection assistant for product documentation.\n"
            "Use ONLY the provided Context to answer. If the Context contains any relevant\n"
            "instructions or details for the user's question, you MUST answer concisely\n"
            "(2–4 sentences) and include concrete commands/flags/paths when applicable.\n"
            "Refuse ONLY if the Context has no relevant information.\n"
            "Refusal text must be exactly:\n"
            "'I don't have enough information in the docs to answer that.'"
        )
        
        # Stop words for keyword analysis
        self._stop_words = {
            "the", "a", "an", "and", "or", "if", "to", "of", "for", "in", "on", "at", "by", "with",
            "is", "are", "be", "was", "were", "it", "this", "that", "as", "from", "into", "out",
            "do", "does", "did", "how", "what", "why", "where", "when", "which", "who", "whom"
        }

    def answer_question(
        self, 
        question: str, 
        k: int = MAX_CHUNKS, 
        domains: Optional[List[str]] = None,
        min_confidence: Optional[float] = None
    ) -> Dict:
        """
        Answer a question using RAG pipeline.
        
        Args:
            question: User question to answer
            k: Number of document chunks to retrieve
            domains: Optional domain filter for retrieval
            min_confidence: Override minimum confidence threshold
            
        Returns:
            Dict with answer, citations, confidence, and metadata
        """
        self.metrics["queries_processed"] += 1
        self.metrics["last_query_time"] = datetime.now().isoformat()
        
        try:
            # Search for relevant documents
            hits = self.search_documents(question, k=k, domains=domains)
            
            # Calculate confidence score
            confidence = self.calculate_confidence(hits, question)
            
            # Check confidence threshold
            min_conf = min_confidence if min_confidence is not None else MIN_CONF
            if confidence < min_conf:
                self.metrics["refusals"] += 1
                return {
                    "answer": "I don't have enough information in the docs to answer that.",
                    "citations": self._to_citations(hits, take=2),
                    "confidence": confidence,
                    "metadata": {
                        "reason": "low_confidence",
                        "threshold": min_conf,
                        "chunks_found": len(hits)
                    }
                }
            
            # Format context for LLM
            context = self._format_context(hits)
            user_prompt = self._build_user_prompt(question, context)
            
            # Generate answer using LLM provider chain
            answer = self._generate_answer(user_prompt)
            
            if not answer or not answer.strip():
                # No valid response from LLM
                if confidence < min_conf:
                    self.metrics["refusals"] += 1
                    return {
                        "answer": "I don't have enough information in the docs to answer that.",
                        "citations": self._to_citations(hits, take=3),
                        "confidence": confidence,
                        "metadata": {"reason": "llm_no_response", "chunks_found": len(hits)}
                    }
                # Fallback to extractive answer
                top_chunk = hits[0]["text"].strip()
                snippet = top_chunk[:240].split("\n\n")[0].strip()
                answer = snippet
            
            self.metrics["successful_answers"] += 1
            self._update_average_confidence(confidence)
            
            return {
                "answer": answer.strip(),
                "citations": self._to_citations(hits, take=3),
                "confidence": confidence,
                "metadata": {
                    "chunks_found": len(hits),
                    "context_length": len(context),
                    "domains_filtered": domains is not None
                }
            }
            
        except Exception as e:
            logging.error(f"RAG pipeline error: {e}")
            self.metrics["provider_failures"] += 1
            return {
                "answer": "I encountered an error processing your question. Please try again.",
                "citations": [],
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }

    def search_documents(
        self, 
        query: str, 
        k: int = 5, 
        domains: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search for relevant documents using embedding similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            domains: Optional domain filter
            
        Returns:
            List of document chunks with metadata and similarity scores
        """
        try:
            # Get embedding providers from registry
            embedding_chain = self.provider_registry.build_fallback_chain(ProviderType.EMBEDDING)
            
            # Generate query embedding
            query_embedding = None
            for provider in embedding_chain:
                try:
                    embeddings = provider.embed_texts([query], batch_size=1)
                    query_embedding = embeddings[0]
                    break
                except (ProviderError, ProviderUnavailableError, Exception) as e:
                    logging.warning(f"Embedding provider {provider.get_config().name} failed: {e}")
                    continue
            
            if query_embedding is None:
                logging.error("All embedding providers failed")
                return []
            
            # Query vector database
            from data.store import query_by_embedding
            
            where_filter = None
            if domains:
                where_filter = {"host": {"$in": domains}}
                
            hits = query_by_embedding(query_embedding, k=k, where=where_filter)
            return hits
            
        except Exception as e:
            logging.error(f"Document search error: {e}")
            return []

    def calculate_confidence(self, hits: List[Dict], question: str) -> float:
        """
        Calculate confidence score based on similarity and keyword overlap.
        
        Args:
            hits: Retrieved document chunks
            question: Original question
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not hits:
            return 0.0
            
        # Use top hit for confidence calculation
        top_hit = hits[0]
        
        # Calculate similarity from distance (lower distance = higher similarity)
        distance = top_hit.get("distance", 1.0)
        similarity = self._similarity_from_distance(distance)
        
        # Calculate keyword overlap ratio
        keyword_overlap = self._overlap_ratio(question, top_hit["text"])
        
        # Blend similarity and keyword overlap (favor similarity slightly)
        confidence = 0.6 * similarity + 0.4 * keyword_overlap
        
        return round(max(0.0, min(1.0, confidence)), 3)

    def get_metrics(self) -> Dict:
        """
        Get current engine metrics and performance statistics.
        
        Returns:
            Dictionary containing metrics and performance data
        """
        return {
            **self.metrics,
            "provider_status": self._check_provider_status(),
            "database_status": self._check_database_status()
        }

    def validate_providers(self) -> Dict[str, bool]:
        """
        Validate all configured providers and their availability.
        
        Returns:
            Dictionary mapping provider names to availability status
        """
        status = {}
        
        # Check LLM providers
        llm_chain = self.provider_registry.build_fallback_chain(ProviderType.LLM)
        for provider in llm_chain:
            try:
                # Simple health check
                test_response = provider.chat(
                    system_prompt="Health check. Respond with 'OK'",
                    user_prompt="Test",
                    temperature=0.0,
                    max_tokens=10
                )
                status[f"llm_{provider.get_config().name}"] = bool(test_response)
            except Exception:
                status[f"llm_{provider.get_config().name}"] = False
        
        # Check embedding providers  
        embedding_chain = self.provider_registry.build_fallback_chain(ProviderType.EMBEDDING)
        for provider in embedding_chain:
            try:
                # Test embedding generation
                test_embeddings = provider.embed_texts(["test"], batch_size=1)
                status[f"embedding_{provider.get_config().name}"] = len(test_embeddings) > 0
            except Exception:
                status[f"embedding_{provider.get_config().name}"] = False
                
        return status

    # Private helper methods
    
    def _generate_answer(self, user_prompt: str) -> str:
        """Generate answer using LLM provider chain."""
        llm_chain = self.provider_registry.build_fallback_chain(ProviderType.LLM)
        
        for provider in llm_chain:
            try:
                response = provider.chat(
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.0,
                    max_tokens=None
                )
                return response
            except (ProviderError, ProviderUnavailableError, Exception) as e:
                logging.warning(f"LLM provider {provider.get_config().name} failed: {e}")
                continue
        
        return ""
    
    def _build_user_prompt(self, question: str, context: str) -> str:
        """Build user prompt with question and context."""
        return (
            f"Question: {question}\n\n"
            f"Context (numbered citations):\n{context}\n\n"
            "Instructions:\n"
            "1) If any part of the Context is relevant, ANSWER. Keep it to 2–4 sentences.\n"
            "2) Prefer concrete steps and fenced code blocks for commands.\n"
            "3) Do not invent facts not in the Context.\n"
            "4) Refuse ONLY if no relevant information exists in the Context.\n"
        )
    
    def _format_context(self, hits: List[Dict]) -> str:
        """Format document hits into numbered context."""
        lines = []
        for i, hit in enumerate(hits[:MAX_CHUNKS], start=1):
            preview = self._trim_text(hit["text"], MAX_CHARS_PER_CHUNK)
            path = hit["meta"].get("path", "unknown")
            lines.append(f"[{i}] ({path})\n{preview}")
        return "\n\n".join(lines)
    
    def _trim_text(self, text: str, limit: int) -> str:
        """Trim text to character limit with ellipsis."""
        return text if len(text) <= limit else text[:limit].rstrip() + " … "
    
    def _to_citations(self, hits: List[Dict], take: int = 3) -> List[Dict]:
        """Convert hits to citation format."""
        citations = []
        for i, hit in enumerate(hits[:take], start=1):
            citations.append({
                "rank": i,
                "path": hit["meta"].get("path"),
                "chunk_id": hit["meta"].get("chunk_id"),
                "preview": self._trim_text(hit["text"], 200)
            })
        return citations
    
    def _similarity_from_distance(self, distance) -> float:
        """Convert distance to similarity score (0-1)."""
        if not isinstance(distance, (int, float)):
            return 0.5
        return 1.0 / (1.0 + max(0.0, distance))
    
    def _tokens(self, text: str) -> List[str]:
        """Extract and stem tokens from text."""
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [self._stem_simple(t) for t in tokens if len(t) > 2 and t not in self._stop_words]
    
    def _stem_simple(self, word: str) -> str:
        """Apply basic stemming for common suffixes."""
        if len(word) <= 3:
            return word
        if word.endswith("s") and not word.endswith(("ss", "us", "is")):
            return word[:-1]
        if word.endswith("ing") and len(word) > 6:
            if word == "running":
                return "run"
            return word[:-3]
        if word.endswith("ed") and len(word) > 5:
            return word[:-2]
        return word
    
    def _keyword_overlap(self, question: str, text: str) -> int:
        """Count overlapping keywords between question and text."""
        q_tokens = set(self._tokens(question))
        t_tokens = set(self._tokens(text))
        return len(q_tokens.intersection(t_tokens))
    
    def _overlap_ratio(self, question: str, text: str) -> float:
        """Calculate keyword overlap ratio."""
        q_tokens = set(self._tokens(question))
        if not q_tokens:
            return 0.0
        overlap = self._keyword_overlap(question, text)
        # Cap denominator to avoid punishing short questions
        denominator = min(5, len(q_tokens))
        return min(1.0, overlap / max(1, denominator))
    
    def _update_average_confidence(self, confidence: float):
        """Update running average confidence score."""
        successful = self.metrics["successful_answers"]
        if successful == 1:
            self.metrics["average_confidence"] = confidence
        else:
            current_avg = self.metrics["average_confidence"]
            self.metrics["average_confidence"] = round(
                (current_avg * (successful - 1) + confidence) / successful, 3
            )
    
    def _check_provider_status(self) -> Dict:
        """Check status of all providers."""
        return {
            "llm_providers": len(self.provider_registry.build_fallback_chain(ProviderType.LLM)),
            "embedding_providers": len(self.provider_registry.build_fallback_chain(ProviderType.EMBEDDING)),
            "last_validation": datetime.now().isoformat()
        }
    
    def _check_database_status(self) -> Dict:
        """Check vector database connectivity."""
        try:
            from data.store import get_client, get_collection
            client = get_client()
            collection = get_collection(client)
            count = collection.count()
            return {
                "connected": True,
                "collection": CHROMA_COLLECTION,
                "document_count": count,
                "database_path": CHROMA_DB_PATH
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "collection": CHROMA_COLLECTION,
                "database_path": CHROMA_DB_PATH
            }