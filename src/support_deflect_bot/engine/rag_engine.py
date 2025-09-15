"""Unified RAG Engine for Support Deflect Bot."""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..core.providers import get_default_registry, ProviderType, ProviderError, ProviderUnavailableError
from ..core.resilience import (
    retry_with_backoff,
    RetryPolicy,
    CircuitBreakerConfig,
    get_circuit_breaker,
    ErrorClassifier,
    ErrorType,
    CircuitBreakerOpenException
)
from ..utils.settings import (
    ANSWER_MIN_CONF as MIN_CONF,
    MAX_CHARS_PER_CHUNK,
    MAX_CHUNKS,
    CHROMA_COLLECTION,
    CHROMA_DB_PATH,
    RAG_PIPELINE_TIMEOUT
)


class UnifiedRAGEngine:
    """
    Unified RAG engine that orchestrates the entire retrieval-augmented generation pipeline.
    Provides document search, question answering, confidence calculation, and metrics collection.
    Enhanced with comprehensive error recovery mechanisms and resilience patterns.
    """

    def __init__(self, provider_registry=None):
        """Initialize RAG engine with provider registry and configuration."""
        self.provider_registry = provider_registry or get_default_registry()
        self.metrics = {
            "queries_processed": 0,
            "successful_answers": 0,
            "refusals": 0,
            "provider_failures": 0,
            "circuit_breaker_trips": 0,
            "fallback_responses": 0,
            "timeout_errors": 0,
            "retry_attempts": 0,
            "average_confidence": 0.0,
            "last_query_time": None,
            "average_response_time": 0.0
        }

        # Initialize resilience components
        self.retry_policy = RetryPolicy(
            max_retries=2,
            base_delay=0.5,
            max_delay=10.0,
            exponential_base=1.5,
            jitter=True
        )

        # Circuit breakers for different pipeline stages
        self.search_circuit_breaker = get_circuit_breaker(
            "rag_search",
            CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                reset_timeout=30.0,
                half_open_max_calls=2
            )
        )

        self.llm_circuit_breaker = get_circuit_breaker(
            "rag_llm",
            CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=3,
                reset_timeout=60.0,
                half_open_max_calls=3
            )
        )

        # Fallback response cache
        self._response_cache = {}
        self._cache_max_size = 100
        
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
        Answer a question using RAG pipeline with comprehensive error recovery and timeout enforcement.

        Args:
            question: User question to answer
            k: Number of document chunks to retrieve
            domains: Optional domain filter for retrieval
            min_confidence: Override minimum confidence threshold

        Returns:
            Dict with answer, citations, confidence, and metadata
        """
        start_time = time.time()
        deadline = start_time + RAG_PIPELINE_TIMEOUT
        self.metrics["queries_processed"] += 1
        self.metrics["last_query_time"] = datetime.now().isoformat()

        def _check_timeout():
            """Check if we've exceeded the pipeline timeout."""
            if time.time() > deadline:
                self.metrics["timeout_errors"] += 1
                return True
            return False

        # Check for cached response first
        cache_key = self._generate_cache_key(question, k, domains, min_confidence)
        if cache_key in self._response_cache:
            cached_response = self._response_cache[cache_key]
            cached_response["metadata"]["from_cache"] = True
            return cached_response

        try:
            # Check timeout before starting search
            if _check_timeout():
                return self._timeout_response("search_stage", start_time)

            # Search for relevant documents with circuit breaker protection
            hits = self._search_documents_with_recovery(question, k, domains, deadline)

            # Calculate confidence score
            confidence = self.calculate_confidence(hits, question)

            # Check confidence threshold
            min_conf = min_confidence if min_confidence is not None else MIN_CONF
            if confidence < min_conf:
                self.metrics["refusals"] += 1
                response = {
                    "answer": "I don't have enough information in the docs to answer that.",
                    "citations": self._to_citations(hits, take=2),
                    "confidence": confidence,
                    "metadata": {
                        "reason": "low_confidence",
                        "threshold": min_conf,
                        "chunks_found": len(hits),
                        "processing_time": time.time() - start_time
                    }
                }
                self._cache_response(cache_key, response)
                return response

            # Format context for LLM
            context = self._format_context(hits)
            user_prompt = self._build_user_prompt(question, context)

            # Check timeout before generation
            if _check_timeout():
                return self._timeout_response("generation_stage", start_time)

            # Generate answer using LLM provider chain with recovery
            answer = self._generate_answer_with_recovery(user_prompt, question, hits, deadline)

            if not answer or not answer.strip():
                # Fallback to extractive answer if LLM fails
                answer = self._generate_extractive_fallback(hits, confidence, min_conf)
                if not answer:
                    self.metrics["refusals"] += 1
                    response = {
                        "answer": "I don't have enough information in the docs to answer that.",
                        "citations": self._to_citations(hits, take=3),
                        "confidence": confidence,
                        "metadata": {
                            "reason": "llm_no_response_and_low_confidence",
                            "chunks_found": len(hits),
                            "processing_time": time.time() - start_time
                        }
                    }
                    self._cache_response(cache_key, response)
                    return response
                self.metrics["fallback_responses"] += 1

            self.metrics["successful_answers"] += 1
            self._update_average_confidence(confidence)
            self._update_average_response_time(time.time() - start_time)

            response = {
                "answer": answer.strip(),
                "citations": self._to_citations(hits, take=3),
                "confidence": confidence,
                "metadata": {
                    "chunks_found": len(hits),
                    "context_length": len(context),
                    "domains_filtered": domains is not None,
                    "processing_time": time.time() - start_time,
                    "circuit_breaker_states": {
                        "search": self.search_circuit_breaker.state.value,
                        "llm": self.llm_circuit_breaker.state.value
                    }
                }
            }

            self._cache_response(cache_key, response)
            return response

        except CircuitBreakerOpenException as e:
            self.metrics["circuit_breaker_trips"] += 1
            logging.warning(f"Circuit breaker open for RAG pipeline: {e}")
            return self._generate_circuit_breaker_fallback(question)

        except Exception as e:
            logging.error(f"RAG pipeline error: {e}")
            self.metrics["provider_failures"] += 1
            error_type = ErrorClassifier.classify_error(e)

            # Try to provide a meaningful fallback based on error type
            if error_type == ErrorType.TIMEOUT:
                self.metrics["timeout_errors"] += 1
                return {
                    "answer": "The request timed out. Please try asking a simpler question or try again later.",
                    "citations": [],
                    "confidence": 0.0,
                    "metadata": {
                        "error": "timeout",
                        "processing_time": time.time() - start_time
                    }
                }

            return {
                "answer": "I encountered an error processing your question. Please try again.",
                "citations": [],
                "confidence": 0.0,
                "metadata": {
                    "error": str(e),
                    "error_type": error_type.value,
                    "processing_time": time.time() - start_time
                }
            }

    @retry_with_backoff()
    def search_documents(
        self,
        query: str,
        k: int = 5,
        domains: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search for relevant documents using embedding similarity with retry and circuit breaker protection.

        Args:
            query: Search query
            k: Number of results to return
            domains: Optional domain filter

        Returns:
            List of document chunks with metadata and similarity scores
        """
        try:
            with self.search_circuit_breaker:
                # Get embedding providers from registry
                embedding_chain = self.provider_registry.build_fallback_chain(ProviderType.EMBEDDING)

                # Generate query embedding with provider fallback
                query_embedding = None
                embedding_errors = []

                for provider in embedding_chain:
                    try:
                        embeddings = provider.embed_texts([query], batch_size=1)
                        if embeddings and len(embeddings) > 0 and embeddings[0]:
                            query_embedding = embeddings[0]
                            break
                    except (ProviderError, ProviderUnavailableError) as e:
                        embedding_errors.append(f"{provider.get_config().name}: {e}")
                        logging.warning(f"Embedding provider {provider.get_config().name} failed: {e}")
                        continue
                    except Exception as e:
                        embedding_errors.append(f"{provider.get_config().name}: {e}")
                        logging.error(f"Unexpected error from embedding provider {provider.get_config().name}: {e}")
                        continue

                if query_embedding is None:
                    logging.error(f"All embedding providers failed: {embedding_errors}")
                    return []

                # Query vector database with retry logic built into store
                from ..data.store import query_by_embedding

                where_filter = None
                if domains:
                    where_filter = {"host": {"$in": domains}}

                hits = query_by_embedding(query_embedding, k=k, where=where_filter)
                logging.debug(f"Document search successful, found {len(hits)} hits")
                return hits

        except CircuitBreakerOpenException:
            logging.warning("Search circuit breaker is open")
            raise
        except Exception as e:
            logging.error(f"Document search error: {e}")
            error_type = ErrorClassifier.classify_error(e)
            if error_type in {ErrorType.TRANSIENT, ErrorType.CONNECTION, ErrorType.TIMEOUT}:
                # Re-raise for retry
                raise
            # Non-retryable error, return empty results
            return []

    def _timeout_response(self, stage: str, start_time: float) -> Dict:
        """Generate timeout fallback response."""
        return {
            "answer": "I don't have enough information in the docs to answer that.",
            "citations": [],
            "confidence": 0.0,
            "metadata": {
                "reason": "pipeline_timeout",
                "timeout_stage": stage,
                "processing_time": time.time() - start_time,
                "timeout_threshold": RAG_PIPELINE_TIMEOUT
            }
        }

    def _search_documents_with_recovery(self, query: str, k: int, domains: Optional[List[str]], deadline: float = None) -> List[Dict]:
        """Search documents with additional recovery mechanisms."""
        try:
            return self.search_documents(query, k=k, domains=domains)
        except CircuitBreakerOpenException:
            # Circuit breaker is open, try simplified search
            logging.warning("Search circuit breaker open, attempting fallback search")
            return self._fallback_search(query, k, domains)
        except Exception as e:
            logging.error(f"Document search failed completely: {e}")
            # Return empty results - the confidence calculation will handle this
            return []

    def _fallback_search(self, query: str, k: int, domains: Optional[List[str]]) -> List[Dict]:
        """Fallback search when main search fails."""
        try:
            # Try with reduced k to minimize load
            reduced_k = max(1, k // 2)
            return self.search_documents(query, k=reduced_k, domains=None)  # Remove domain filter
        except Exception as e:
            logging.warning(f"Fallback search also failed: {e}")
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
        Get current engine metrics and performance statistics with resilience data.

        Returns:
            Dictionary containing metrics, performance data, and resilience status
        """
        return {
            **self.metrics,
            "provider_status": self._check_provider_status(),
            "database_status": self._check_database_status(),
            "circuit_breaker_status": {
                "search": self.search_circuit_breaker.get_status(),
                "llm": self.llm_circuit_breaker.get_status()
            },
            "cache_status": {
                "size": len(self._response_cache),
                "max_size": self._cache_max_size,
                "hit_ratio": self._calculate_cache_hit_ratio()
            },
            "resilience_summary": {
                "total_errors": self.metrics["provider_failures"] + self.metrics["circuit_breaker_trips"] + self.metrics["timeout_errors"],
                "recovery_rate": self._calculate_recovery_rate(),
                "system_health": self._assess_system_health()
            }
        }

    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        # This would require tracking cache hits, for now return 0
        return 0.0

    def _calculate_recovery_rate(self) -> float:
        """Calculate system recovery rate."""
        total_requests = self.metrics["queries_processed"]
        if total_requests == 0:
            return 1.0

        successful = self.metrics["successful_answers"] + self.metrics["fallback_responses"]
        return round(successful / total_requests, 3)

    def _assess_system_health(self) -> str:
        """Assess overall system health."""
        recovery_rate = self._calculate_recovery_rate()
        search_cb_state = self.search_circuit_breaker.state.value
        llm_cb_state = self.llm_circuit_breaker.state.value

        if recovery_rate >= 0.95 and search_cb_state == "closed" and llm_cb_state == "closed":
            return "healthy"
        elif recovery_rate >= 0.80 or search_cb_state == "half_open" or llm_cb_state == "half_open":
            return "degraded"
        else:
            return "unhealthy"

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
        return self._generate_answer_with_recovery(user_prompt, "", [])

    def _generate_answer_with_recovery(self, user_prompt: str, question: str, hits: List[Dict], deadline: float = None) -> str:
        """Generate answer using LLM provider chain with comprehensive error recovery."""
        try:
            # Check timeout before generation attempt
            if deadline and time.time() > deadline:
                logger.warning("Generation stage timeout reached")
                return ""

            with self.llm_circuit_breaker:
                llm_chain = self.provider_registry.build_fallback_chain(ProviderType.LLM)
                llm_errors = []

                for provider in llm_chain:
                    try:
                        response = provider.chat(
                            system_prompt=self.system_prompt,
                            user_prompt=user_prompt,
                            temperature=0.0,
                            max_tokens=None
                        )
                        if response and response.strip():
                            return response
                    except (ProviderError, ProviderUnavailableError) as e:
                        llm_errors.append(f"{provider.get_config().name}: {e}")
                        logging.warning(f"LLM provider {provider.get_config().name} failed: {e}")
                        continue
                    except Exception as e:
                        llm_errors.append(f"{provider.get_config().name}: {e}")
                        logging.error(f"Unexpected error from LLM provider {provider.get_config().name}: {e}")
                        continue

                logging.error(f"All LLM providers failed: {llm_errors}")
                return ""

        except CircuitBreakerOpenException:
            logging.warning("LLM circuit breaker is open, falling back to extractive answer")
            # Circuit breaker is open, try extractive fallback
            return self._generate_extractive_fallback(hits, 0.5, MIN_CONF) or ""
        except Exception as e:
            logging.error(f"LLM generation failed completely: {e}")
            return ""

    def _generate_extractive_fallback(self, hits: List[Dict], confidence: float, min_conf: float) -> Optional[str]:
        """Generate extractive answer as fallback when LLM fails."""
        if not hits or confidence < min_conf:
            return None

        try:
            # Use the top chunk as extractive answer
            top_chunk = hits[0]["text"].strip()
            # Take first meaningful paragraph or sentence
            sentences = top_chunk.split(". ")
            if len(sentences) > 1:
                # Take first 1-2 sentences
                answer = ". ".join(sentences[:2]) + "."
            else:
                # Take up to 240 characters
                answer = top_chunk[:240].split("\n\n")[0].strip()

            if len(answer) > 20:  # Ensure it's substantial enough
                return answer
            return None

        except Exception as e:
            logging.error(f"Extractive fallback generation failed: {e}")
            return None

    def _generate_circuit_breaker_fallback(self, question: str) -> Dict:
        """Generate fallback response when circuit breakers are open."""
        return {
            "answer": "The system is currently experiencing high load. Please try your question again in a few moments.",
            "citations": [],
            "confidence": 0.0,
            "metadata": {
                "reason": "circuit_breaker_open",
                "fallback": True,
                "suggestion": "Try again in 30-60 seconds"
            }
        }

    def _generate_cache_key(self, question: str, k: int, domains: Optional[List[str]], min_confidence: Optional[float]) -> str:
        """Generate cache key for response caching."""
        import hashlib
        key_parts = [
            question.lower().strip(),
            str(k),
            str(sorted(domains) if domains else ""),
            str(min_confidence)
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _cache_response(self, cache_key: str, response: Dict):
        """Cache response with size limit."""
        if len(self._response_cache) >= self._cache_max_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self._response_cache.keys())[:10]
            for key in oldest_keys:
                del self._response_cache[key]

        # Cache a copy without processing time for consistency
        cached_response = response.copy()
        if "metadata" in cached_response:
            cached_response["metadata"] = cached_response["metadata"].copy()
            cached_response["metadata"].pop("processing_time", None)

        self._response_cache[cache_key] = cached_response

    def _update_average_response_time(self, response_time: float):
        """Update running average response time."""
        successful = self.metrics["successful_answers"]
        if successful == 1:
            self.metrics["average_response_time"] = response_time
        else:
            current_avg = self.metrics["average_response_time"]
            self.metrics["average_response_time"] = round(
                (current_avg * (successful - 1) + response_time) / successful, 3
            )
    
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
            from ..data.store import get_client, get_collection
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