"""Unified Query Service for RAG Explorer engine."""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class UnifiedQueryService:
    """Unified query service that handles query preprocessing and optimization."""

    def __init__(self):
        """Initialize the query service."""
        logger.debug("Initialized UnifiedQueryService")

    def preprocess_query(self, query: str) -> str:
        """Preprocess query text for better search results.

        Performs basic text cleaning and normalization:
        - Strips whitespace
        - Normalizes spaces
        - Removes special characters (optional)
        - Converts to lowercase for consistency
        - Removes common stop words that don't add semantic value

        Args:
            query: Raw query string

        Returns:
            Preprocessed query string

        Raises:
            ValueError: If query is empty or invalid
        """
        if not query:
            raise ValueError("Query cannot be empty")

        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        try:
            # Step 1: Basic cleaning
            processed = query.strip()

            if not processed:
                raise ValueError("Query cannot be empty after cleaning")

            # Step 2: Normalize whitespace (replace multiple spaces with single space)
            processed = re.sub(r'\s+', ' ', processed)

            # Step 3: Remove excessive punctuation but keep meaningful ones
            # Keep periods, commas, question marks, hyphens, apostrophes
            processed = re.sub(r'[^\w\s\.\,\?\-\']', ' ', processed)

            # Step 4: Remove very short words (1-2 characters) that are likely noise
            # But keep common meaningful short words
            keep_short = {'is', 'in', 'on', 'at', 'to', 'it', 'of', 'or', 'up'}
            words = processed.split()
            filtered_words = []

            for word in words:
                word_clean = word.strip('.,?!').lower()
                if len(word_clean) >= 3 or word_clean in keep_short:
                    filtered_words.append(word)

            processed = ' '.join(filtered_words)

            # Step 5: Final cleanup
            processed = processed.strip()

            if not processed:
                # If preprocessing removed everything, return original cleaned query
                processed = query.strip()

            logger.debug(f"Query preprocessing: '{query}' -> '{processed}'")
            return processed

        except Exception as e:
            logger.warning(f"Query preprocessing failed: {e}, using original query")
            return query.strip()

    def extract_keywords(self, query: str) -> list[str]:
        """Extract meaningful keywords from query.

        Args:
            query: Query string

        Returns:
            List of keywords extracted from the query
        """
        if not query:
            return []

        try:
            # Preprocess first
            processed = self.preprocess_query(query)

            # Extract words that are likely to be meaningful keywords
            words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', processed.lower())

            # Remove common stop words that don't add semantic value
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
                'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
                'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'
            }

            keywords = [word for word in words if word not in stop_words]

            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = []
            for word in keywords:
                if word not in seen:
                    seen.add(word)
                    unique_keywords.append(word)

            logger.debug(f"Extracted {len(unique_keywords)} keywords from query: {unique_keywords}")
            return unique_keywords

        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []

    def expand_query(self, query: str, synonyms: Optional[dict] = None) -> str:
        """Expand query with synonyms for better search coverage.

        Args:
            query: Original query
            synonyms: Optional dictionary of word -> list of synonyms

        Returns:
            Expanded query string
        """
        if not query:
            return query

        if not synonyms:
            # Basic built-in synonyms for common technical terms
            synonyms = {
                'function': ['method', 'procedure'],
                'method': ['function', 'procedure'],
                'class': ['object', 'type'],
                'variable': ['var', 'field'],
                'error': ['exception', 'bug', 'issue'],
                'fix': ['solve', 'repair', 'correct'],
                'create': ['make', 'build', 'generate'],
                'delete': ['remove', 'destroy'],
                'update': ['modify', 'change', 'edit']
            }

        try:
            processed = self.preprocess_query(query)
            words = processed.lower().split()
            expanded_words = []

            for word in words:
                expanded_words.append(word)
                if word in synonyms:
                    # Add first synonym for context
                    expanded_words.append(synonyms[word][0])

            expanded_query = ' '.join(expanded_words)
            logger.debug(f"Query expansion: '{query}' -> '{expanded_query}'")
            return expanded_query

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original query")
            return query

    def validate_query(self, query: str, max_length: int = 1000) -> bool:
        """Validate if query is suitable for processing.

        Args:
            query: Query to validate
            max_length: Maximum allowed query length

        Returns:
            True if query is valid, False otherwise
        """
        if not query or not isinstance(query, str):
            return False

        if len(query.strip()) == 0:
            return False

        if len(query) > max_length:
            logger.warning(f"Query too long: {len(query)} > {max_length}")
            return False

        # Check if query has at least some meaningful content
        meaningful_chars = re.findall(r'[a-zA-Z0-9]', query)
        if len(meaningful_chars) < 2:
            return False

        return True

    def get_query_stats(self, query: str) -> dict:
        """Get statistics about a query.

        Args:
            query: Query to analyze

        Returns:
            Dictionary with query statistics
        """
        if not query:
            return {
                "length": 0,
                "word_count": 0,
                "keywords": [],
                "is_valid": False
            }

        try:
            processed = self.preprocess_query(query)
            keywords = self.extract_keywords(query)

            return {
                "original_length": len(query),
                "processed_length": len(processed),
                "word_count": len(processed.split()),
                "keywords": keywords,
                "keyword_count": len(keywords),
                "is_valid": self.validate_query(query),
                "processed_query": processed
            }

        except Exception as e:
            logger.warning(f"Query stats calculation failed: {e}")
            return {
                "length": len(query),
                "word_count": 0,
                "keywords": [],
                "is_valid": False,
                "error": str(e)
            }