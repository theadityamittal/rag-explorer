"""Unified RAG Engine for document search and question answering."""

import logging
import re
from typing import Dict, List, Any, Optional
from rag_explorer.core.registry import ProviderRegistry, ProviderNotConfiguredError, ProviderNotAvailableError
from rag_explorer.core.providers.base import ProviderError
from .database import simple_query_by_embedding, simple_get_collection_count
from rag_explorer.utils.settings import CHROMA_COLLECTION

logger = logging.getLogger(__name__)


class UnifiedRAGEngine:
    """Unified RAG engine that orchestrates document search and question answering."""

    def __init__(self):
        """Initialize the RAG engine."""
        self.registry = ProviderRegistry()
        logger.debug("Initialized UnifiedRAGEngine")

    def search_documents(self, query: str, count: int = 5) -> List[Dict[str, Any]]:
        """Search documents by query text.

        Args:
            query: Search query text
            count: Number of results to return

        Returns:
            List of matching documents with metadata

        Raises:
            ConnectionError: If providers are not configured properly
            RuntimeError: If search operation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if count <= 0:
            raise ValueError("Count must be positive")

        try:
            # Get embedding provider
            embedding_provider = self._get_embedding_provider()

            # Generate query embedding
            query_embedding = embedding_provider.embed_one(query.strip())

            # Search database
            results = simple_query_by_embedding(
                query_embedding=query_embedding,
                top_k=count,
                collection_name=CHROMA_COLLECTION
            )

            logger.info(f"Found {len(results)} documents for query: {query[:50]}...")
            return results

        except (ProviderNotConfiguredError, ProviderNotAvailableError) as e:
            raise ConnectionError(str(e))
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            raise RuntimeError(f"Search operation failed: {e}")

    def answer_question(self, question: str, k: int = 5, min_confidence: float = 0.25) -> Dict[str, Any]:
        """Answer a question using RAG (Retrieval-Augmented Generation).

        Args:
            question: The question to answer
            k: Number of documents to retrieve for context
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary containing:
                - answer: Generated answer text
                - confidence: Confidence score (0.0-1.0)
                - sources: List of source documents used
                - retrieved_docs: Number of documents retrieved

        Raises:
            ConnectionError: If providers are not configured properly
            RuntimeError: If RAG operation fails
            ValueError: If confidence is below threshold
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if k <= 0:
            raise ValueError("k must be positive")

        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0.0 and 1.0")

        try:
            # Get providers
            llm_provider = self._get_llm_provider()
            embedding_provider = self._get_embedding_provider()

            # Search for relevant documents
            question_clean = question.strip()
            query_embedding = embedding_provider.embed_one(question_clean)

            relevant_docs = simple_query_by_embedding(
                query_embedding=query_embedding,
                top_k=k,
                collection_name=CHROMA_COLLECTION
            )

            if not relevant_docs:
                raise RuntimeError("No relevant documents found in the database")

            # Calculate confidence
            confidence = self.calculate_confidence(relevant_docs, question_clean)

            if confidence < min_confidence:
                raise ValueError(f"Confidence {confidence:.2f} below threshold {min_confidence}")

            # Build context from retrieved documents
            context_parts = []
            sources = []
            for i, doc in enumerate(relevant_docs[:k], 1):
                text = doc.get('text', '').strip()
                if text:
                    context_parts.append(f"Document {i}:\n{text}")
                    # Extract source from metadata
                    metadata = doc.get('metadata', {})
                    source = metadata.get('source') or metadata.get('path') or f"Document {i}"
                    sources.append(source)

            context = "\n\n".join(context_parts)

            # Create system prompt for RAG
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
- Use ONLY the information provided in the context below
- If the context doesn't contain enough information to answer the question, say so
- Be precise and cite specific details from the context
- Keep answers concise but complete"""

            # Create user prompt with context and question
            user_prompt = f"""Context:
{context}

Question: {question_clean}

Please answer the question based on the provided context."""

            # Generate answer using LLM
            answer = llm_provider.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1  # Low temperature for factual responses
            )

            logger.info(f"Generated answer for question: {question_clean[:50]}... (confidence: {confidence:.2f})")

            return {
                'answer': answer.strip(),
                'confidence': confidence,
                'sources': sources,
                'retrieved_docs': len(relevant_docs)
            }

        except (ProviderNotConfiguredError, ProviderNotAvailableError) as e:
            raise ConnectionError(str(e))
        except ValueError as e:
            # Re-raise ValueError (confidence/input validation)
            raise
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            raise RuntimeError(f"RAG operation failed: {e}")

    def calculate_confidence(self, hits: List[Dict[str, Any]], question: str) -> float:
        """Calculate confidence score using 4-factor weighted approach.

        Factors:
        - Similarity scores (40% weight)
        - Result count (20% weight)
        - Keyword overlap (20% weight)
        - Content length (20% weight)

        Args:
            hits: List of retrieved documents with similarity scores
            question: Original question text

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not hits:
            return 0.0

        try:
            # Factor 1: Similarity scores (40% weight)
            similarity_scores = [hit.get('similarity_score', 0.0) for hit in hits]
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            similarity_factor = min(avg_similarity, 1.0)

            # Factor 2: Result count (20% weight)
            # More results generally indicate better coverage
            result_count = len(hits)
            max_expected_results = 10  # Normalize against expected maximum
            count_factor = min(result_count / max_expected_results, 1.0)

            # Factor 3: Keyword overlap (20% weight)
            question_keywords = self._extract_keywords(question.lower())
            if question_keywords:
                overlap_scores = []
                for hit in hits:
                    text = hit.get('text', '').lower()
                    text_keywords = self._extract_keywords(text)
                    if text_keywords:
                        overlap = len(question_keywords.intersection(text_keywords))
                        overlap_score = overlap / len(question_keywords)
                        overlap_scores.append(overlap_score)

                keyword_factor = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
                keyword_factor = min(keyword_factor, 1.0)
            else:
                keyword_factor = 0.0

            # Factor 4: Content length (20% weight)
            # Longer content generally provides more context
            content_lengths = [len(hit.get('text', '')) for hit in hits]
            avg_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0.0
            expected_min_length = 100  # Minimum expected useful content length
            expected_max_length = 2000  # Maximum expected useful content length
            normalized_length = min(max(avg_length - expected_min_length, 0) /
                                   (expected_max_length - expected_min_length), 1.0)
            content_factor = normalized_length

            # Weighted combination
            confidence = (
                similarity_factor * 0.4 +   # 40% weight
                count_factor * 0.2 +        # 20% weight
                keyword_factor * 0.2 +      # 20% weight
                content_factor * 0.2        # 20% weight
            )

            # Ensure result is in valid range
            confidence = max(0.0, min(confidence, 1.0))

            logger.debug(f"Confidence calculation: similarity={similarity_factor:.3f}, "
                        f"count={count_factor:.3f}, keywords={keyword_factor:.3f}, "
                        f"content={content_factor:.3f} -> {confidence:.3f}")

            return confidence

        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.0

    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text.

        Args:
            text: Input text

        Returns:
            Set of keywords
        """
        # Simple keyword extraction - remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'what',
            'when', 'where', 'who', 'why', 'how', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        # Extract words (alphanumeric, at least 3 characters)
        words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower())
        keywords = {word for word in words if word not in stop_words}

        return keywords

    def _get_llm_provider(self):
        """Get LLM provider with clear error messages."""
        try:
            return self.registry.get_llm_provider()
        except ProviderNotConfiguredError as e:
            # Extract provider name from error and provide clear guidance
            provider_name = str(e).split()[0].lower() if str(e) else "unknown"
            if "openai" in provider_name:
                raise ConnectionError("OpenAI provider not configured. Please set OPENAI_API_KEY environment variable or change PRIMARY_LLM_PROVIDER setting.")
            elif "anthropic" in provider_name:
                raise ConnectionError("Anthropic provider not configured. Please set ANTHROPIC_API_KEY environment variable or change PRIMARY_LLM_PROVIDER setting.")
            elif "google" in provider_name:
                raise ConnectionError("Google provider not configured. Please set GEMINI_API_KEY environment variable or change PRIMARY_LLM_PROVIDER setting.")
            elif "ollama" in provider_name:
                raise ConnectionError("Ollama provider not configured. Please ensure Ollama is running or change PRIMARY_LLM_PROVIDER setting.")
            else:
                raise ConnectionError(f"LLM provider not configured: {e}")
        except ProviderNotAvailableError as e:
            raise ConnectionError(f"LLM provider is not available: {e}")

    def _get_embedding_provider(self):
        """Get embedding provider with clear error messages."""
        try:
            return self.registry.get_embedding_provider()
        except ProviderNotConfiguredError as e:
            # Extract provider name from error and provide clear guidance
            provider_name = str(e).split()[0].lower() if str(e) else "unknown"
            if "openai" in provider_name:
                raise ConnectionError("OpenAI provider not configured. Please set OPENAI_API_KEY environment variable or change PRIMARY_EMBEDDING_PROVIDER setting.")
            elif "google" in provider_name:
                raise ConnectionError("Google provider not configured. Please set GEMINI_API_KEY environment variable or change PRIMARY_EMBEDDING_PROVIDER setting.")
            elif "ollama" in provider_name:
                raise ConnectionError("Ollama provider not configured. Please ensure Ollama is running or change PRIMARY_EMBEDDING_PROVIDER setting.")
            else:
                raise ConnectionError(f"Embedding provider not configured: {e}")
        except ProviderNotAvailableError as e:
            raise ConnectionError(f"Embedding provider is not available: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get engine status and health information.

        Returns:
            Dictionary with engine status information
        """
        try:
            # Check database
            doc_count = simple_get_collection_count(CHROMA_COLLECTION)

            # Check providers
            llm_status = "available"
            embedding_status = "available"

            try:
                self._get_llm_provider()
            except Exception as e:
                llm_status = f"error: {e}"

            try:
                self._get_embedding_provider()
            except Exception as e:
                embedding_status = f"error: {e}"

            return {
                "status": "healthy" if "error" not in llm_status and "error" not in embedding_status else "degraded",
                "database": {
                    "collection": CHROMA_COLLECTION,
                    "document_count": doc_count
                },
                "providers": {
                    "llm": llm_status,
                    "embedding": embedding_status
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }