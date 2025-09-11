"""Unified Query Service for Support Deflect Bot."""

import logging
import re
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime

try:
    from ..core.providers import get_default_registry, ProviderType, ProviderError, ProviderUnavailableError
except ImportError:
    # Fallback to old provider system during transition
    from support_deflect_bot_old.core.providers import get_default_registry, ProviderType, ProviderError, ProviderUnavailableError


class UnifiedQueryService:
    """
    Unified query service that handles query preprocessing, document retrieval,
    result ranking, and query analytics.
    """
    
    def __init__(self, provider_registry=None):
        """Initialize query service with provider registry and configuration."""
        self.provider_registry = provider_registry or get_default_registry()
        
        # Query analytics and metrics
        self.query_stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "average_results_returned": 0.0,
            "last_query_time": None,
            "common_query_patterns": {}
        }
        
        # Stop words for query preprocessing
        self._stop_words = {
            "the", "a", "an", "and", "or", "if", "to", "of", "for", "in", "on", "at", "by", "with",
            "is", "are", "be", "was", "were", "it", "this", "that", "as", "from", "into", "out",
            "do", "does", "did", "how", "what", "why", "where", "when", "which", "who", "whom",
            "can", "could", "would", "should", "will", "shall", "may", "might", "must"
        }
        
        # Query expansion patterns
        self._expansion_patterns = {
            r'\binstall\b': ['setup', 'configure', 'deploy'],
            r'\berror\b': ['issue', 'problem', 'bug', 'fail'],
            r'\bconfig\b': ['configuration', 'settings', 'options'],
            r'\brun\b': ['execute', 'start', 'launch'],
            r'\bapi\b': ['endpoint', 'interface', 'service']
        }

    def preprocess_query(
        self, 
        query: str, 
        expand_query: bool = True,
        min_length: int = 2,
        max_length: int = 1000
    ) -> Dict[str, any]:
        """
        Preprocess and analyze user query for optimal retrieval.
        
        Args:
            query: Raw user query
            expand_query: Whether to apply query expansion
            min_length: Minimum query length
            max_length: Maximum query length
            
        Returns:
            Dictionary with processed query and metadata
        """
        self.query_stats["total_queries"] += 1
        self.query_stats["last_query_time"] = datetime.now().isoformat()
        
        # Basic validation
        original_query = query
        if not query or len(query.strip()) < min_length:
            return {
                "processed_query": "",
                "original_query": original_query,
                "valid": False,
                "reason": "Query too short",
                "tokens": [],
                "keywords": [],
                "query_type": "invalid"
            }
        
        if len(query) > max_length:
            query = query[:max_length].rstrip()
        
        # Clean and normalize query
        cleaned_query = self._clean_query(query)
        
        # Extract tokens and keywords
        tokens = self._tokenize(cleaned_query)
        keywords = self._extract_keywords(tokens)
        
        # Determine query type
        query_type = self._classify_query(cleaned_query, keywords)
        
        # Apply query expansion if requested
        expanded_query = cleaned_query
        expansion_terms = []
        if expand_query:
            expanded_query, expansion_terms = self._expand_query(cleaned_query)
        
        # Track query patterns
        self._track_query_patterns(query_type, keywords)
        
        return {
            "processed_query": expanded_query,
            "original_query": original_query,
            "cleaned_query": cleaned_query,
            "valid": True,
            "tokens": tokens,
            "keywords": keywords,
            "query_type": query_type,
            "expansion_terms": expansion_terms,
            "metadata": {
                "token_count": len(tokens),
                "keyword_count": len(keywords),
                "has_technical_terms": self._has_technical_terms(keywords),
                "processing_time": datetime.now().isoformat()
            }
        }

    def retrieve_documents(
        self, 
        processed_query: Dict[str, any], 
        k: int = 5,
        domains: Optional[List[str]] = None,
        use_hybrid_search: bool = True
    ) -> List[Dict]:
        """
        Retrieve relevant documents using processed query.
        
        Args:
            processed_query: Output from preprocess_query
            k: Number of documents to retrieve
            domains: Optional domain filter
            use_hybrid_search: Whether to combine semantic and keyword search
            
        Returns:
            List of retrieved documents with scores
        """
        if not processed_query.get("valid", False):
            return []
        
        try:
            query_text = processed_query["processed_query"]
            
            # Generate query embedding
            query_embedding = self._get_query_embedding(query_text)
            if query_embedding is None:
                self.query_stats["failed_retrievals"] += 1
                return []
            
            # Perform semantic search
            semantic_results = self._semantic_search(
                query_embedding, 
                k=k * 2,  # Get more for ranking
                domains=domains
            )
            
            # Apply hybrid search if requested
            if use_hybrid_search and processed_query.get("keywords"):
                keyword_results = self._keyword_search(
                    processed_query["keywords"],
                    k=k,
                    domains=domains
                )
                # Combine and rerank results
                combined_results = self._combine_search_results(
                    semantic_results, 
                    keyword_results, 
                    processed_query
                )
            else:
                combined_results = semantic_results
            
            # Final ranking and selection
            ranked_results = self.rank_results(combined_results, processed_query)
            final_results = ranked_results[:k]
            
            self.query_stats["successful_retrievals"] += 1
            self._update_average_results(len(final_results))
            
            return final_results
            
        except Exception as e:
            logging.error(f"Error retrieving documents: {e}")
            self.query_stats["failed_retrievals"] += 1
            return []

    def rank_results(
        self, 
        results: List[Dict], 
        processed_query: Dict[str, any],
        boost_recent: bool = True,
        boost_exact_matches: bool = True
    ) -> List[Dict]:
        """
        Rank and score retrieved documents based on relevance.
        
        Args:
            results: Raw search results
            processed_query: Processed query information
            boost_recent: Whether to boost recently updated content
            boost_exact_matches: Whether to boost exact keyword matches
            
        Returns:
            Ranked list of documents with relevance scores
        """
        if not results:
            return []
        
        keywords = processed_query.get("keywords", [])
        query_text = processed_query.get("cleaned_query", "")
        
        # Calculate relevance scores for each result
        scored_results = []
        for result in results:
            score_components = {}
            
            # Base similarity score (from vector search)
            base_score = self._get_base_similarity_score(result)
            score_components["similarity"] = base_score
            
            # Keyword overlap score
            keyword_score = self.calculate_keyword_overlap(
                query_text, 
                result.get("text", ""), 
                keywords
            )
            score_components["keyword_overlap"] = keyword_score
            
            # Title relevance boost
            title_score = self._calculate_title_relevance(
                keywords, 
                result.get("meta", {}).get("title", "")
            )
            score_components["title_relevance"] = title_score
            
            # Recency boost if requested
            recency_score = 0.0
            if boost_recent:
                recency_score = self._calculate_recency_boost(result)
            score_components["recency"] = recency_score
            
            # Exact match boost if requested
            exact_match_score = 0.0
            if boost_exact_matches:
                exact_match_score = self._calculate_exact_match_boost(keywords, result.get("text", ""))
            score_components["exact_match"] = exact_match_score
            
            # Source quality score (based on metadata)
            quality_score = self._calculate_source_quality(result)
            score_components["source_quality"] = quality_score
            
            # Combined relevance score (weighted sum)
            final_score = (
                0.4 * base_score +           # Semantic similarity
                0.25 * keyword_score +       # Keyword overlap
                0.15 * title_score +         # Title relevance
                0.1 * recency_score +        # Recency boost
                0.05 * exact_match_score +   # Exact matches
                0.05 * quality_score         # Source quality
            )
            
            # Add scoring details to result
            result["relevance_score"] = round(final_score, 4)
            result["score_components"] = score_components
            scored_results.append(result)
        
        # Sort by relevance score (descending)
        ranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)
        
        return ranked_results

    def calculate_keyword_overlap(
        self, 
        query: str, 
        text: str, 
        keywords: Optional[List[str]] = None
    ) -> float:
        """
        Calculate keyword overlap score between query and text.
        
        Args:
            query: Query text
            text: Document text
            keywords: Optional pre-extracted keywords
            
        Returns:
            Overlap score between 0.0 and 1.0
        """
        if not query or not text:
            return 0.0
        
        # Use provided keywords or extract from query
        if keywords is None:
            query_tokens = self._tokenize(query)
            keywords = self._extract_keywords(query_tokens)
        
        if not keywords:
            return 0.0
        
        # Extract keywords from text
        text_tokens = self._tokenize(text)
        text_keywords = set(self._extract_keywords(text_tokens))
        
        # Calculate overlap
        query_keywords = set(keywords)
        overlap = len(query_keywords.intersection(text_keywords))
        
        # Normalize by query keyword count (with minimum threshold)
        denominator = min(5, len(query_keywords))  # Cap to avoid penalizing short queries
        overlap_ratio = overlap / max(1, denominator)
        
        return min(1.0, overlap_ratio)

    def apply_domain_filters(
        self, 
        results: List[Dict], 
        domains: Optional[List[str]] = None,
        include_localhost: bool = True
    ) -> List[Dict]:
        """
        Apply domain-based filtering to search results.
        
        Args:
            results: Search results to filter
            domains: List of allowed domains
            include_localhost: Whether to include local files
            
        Returns:
            Filtered results
        """
        if not domains:
            return results
        
        filtered_results = []
        allowed_domains = set(domains)
        
        if include_localhost:
            allowed_domains.add("localhost")
        
        for result in results:
            meta = result.get("meta", {})
            host = meta.get("host", "")
            
            # Check if host is in allowed domains
            if host in allowed_domains:
                filtered_results.append(result)
            # Also check for partial matches (subdomain support)
            elif any(host.endswith(f".{domain}") for domain in allowed_domains):
                filtered_results.append(result)
        
        return filtered_results

    def get_query_analytics(self) -> Dict:
        """
        Get comprehensive query analytics and insights.
        
        Returns:
            Dictionary with query statistics and patterns
        """
        return {
            **self.query_stats,
            "success_rate": (
                self.query_stats["successful_retrievals"] / 
                max(1, self.query_stats["total_queries"])
            ) if self.query_stats["total_queries"] > 0 else 0.0,
            "provider_status": self._check_embedding_providers()
        }

    # Private helper methods
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text."""
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters that don't add meaning
        cleaned = re.sub(r'[^\w\s\-\?\!\.]+', ' ', cleaned)
        
        # Normalize case for processing (but preserve original for some uses)
        return cleaned
    
    def _tokenize(self, text: str) -> List[str]:
        """Extract tokens from text."""
        # Extract alphanumeric tokens
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        return [token for token in tokens if len(token) > 1]
    
    def _extract_keywords(self, tokens: List[str]) -> List[str]:
        """Extract meaningful keywords from tokens."""
        keywords = []
        for token in tokens:
            # Apply basic stemming
            stemmed = self._stem_simple(token)
            # Filter out stop words and very short tokens
            if len(stemmed) > 2 and stemmed not in self._stop_words:
                keywords.append(stemmed)
        return keywords
    
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
    
    def _classify_query(self, query: str, keywords: List[str]) -> str:
        """Classify query type for optimization."""
        query_lower = query.lower()
        
        # Check for common patterns
        if any(word in query_lower for word in ["how", "tutorial", "guide", "step"]):
            return "how_to"
        elif any(word in query_lower for word in ["error", "issue", "problem", "bug", "fail"]):
            return "troubleshooting"
        elif any(word in query_lower for word in ["install", "setup", "configure", "deploy"]):
            return "installation"
        elif any(word in query_lower for word in ["api", "function", "method", "class"]):
            return "technical_reference"
        elif len(keywords) <= 2:
            return "simple"
        else:
            return "complex"
    
    def _has_technical_terms(self, keywords: List[str]) -> bool:
        """Check if query contains technical terms."""
        technical_patterns = [
            r'api', r'sdk', r'cli', r'json', r'xml', r'http', r'ssl', r'auth',
            r'config', r'env', r'var', r'function', r'method', r'class', r'module'
        ]
        keywords_text = ' '.join(keywords)
        return any(re.search(pattern, keywords_text, re.IGNORECASE) for pattern in technical_patterns)
    
    def _expand_query(self, query: str) -> Tuple[str, List[str]]:
        """Apply query expansion using predefined patterns."""
        expanded_query = query
        expansion_terms = []
        
        for pattern, expansions in self._expansion_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                # Add expansion terms to query
                expansion_terms.extend(expansions)
                expanded_query += " " + " ".join(expansions)
        
        return expanded_query, expansion_terms
    
    def _track_query_patterns(self, query_type: str, keywords: List[str]):
        """Track common query patterns for analytics."""
        # Track query types
        if query_type not in self.query_stats["common_query_patterns"]:
            self.query_stats["common_query_patterns"][query_type] = 0
        self.query_stats["common_query_patterns"][query_type] += 1
    
    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for query text."""
        embedding_chain = self.provider_registry.build_fallback_chain(ProviderType.EMBEDDING)
        
        for provider in embedding_chain:
            try:
                embeddings = provider.embed_texts([query], batch_size=1)
                return embeddings[0] if embeddings else None
            except (ProviderError, ProviderUnavailableError, Exception) as e:
                logging.warning(f"Embedding provider {provider.get_config().name} failed: {e}")
                continue
        
        return None
    
    def _semantic_search(
        self, 
        query_embedding: List[float], 
        k: int, 
        domains: Optional[List[str]]
    ) -> List[Dict]:
        """Perform semantic search using query embedding."""
        try:
            from src.data.store import query_by_embedding
            
            where_filter = None
            if domains:
                where_filter = {"host": {"$in": domains}}
            
            results = query_by_embedding(query_embedding, k=k, where=where_filter)
            return results
            
        except Exception as e:
            logging.error(f"Semantic search error: {e}")
            return []
    
    def _keyword_search(
        self, 
        keywords: List[str], 
        k: int, 
        domains: Optional[List[str]]
    ) -> List[Dict]:
        """Perform keyword-based search (placeholder - would need full-text search setup)."""
        # This is a simplified keyword search
        # In production, you'd use a proper full-text search index
        return []  # Placeholder
    
    def _combine_search_results(
        self, 
        semantic_results: List[Dict], 
        keyword_results: List[Dict], 
        processed_query: Dict
    ) -> List[Dict]:
        """Combine semantic and keyword search results."""
        # Simple combination - dedupe by document ID and merge scores
        seen_ids = set()
        combined = []
        
        # Add semantic results first
        for result in semantic_results:
            doc_id = result.get("meta", {}).get("chunk_id", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined.append(result)
        
        # Add unique keyword results
        for result in keyword_results:
            doc_id = result.get("meta", {}).get("chunk_id", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined.append(result)
        
        return combined
    
    def _get_base_similarity_score(self, result: Dict) -> float:
        """Extract base similarity score from search result."""
        distance = result.get("distance", 1.0)
        # Convert distance to similarity (lower distance = higher similarity)
        return 1.0 / (1.0 + max(0.0, distance))
    
    def _calculate_title_relevance(self, keywords: List[str], title: str) -> float:
        """Calculate relevance boost based on title matches."""
        if not title or not keywords:
            return 0.0
        
        title_tokens = self._tokenize(title)
        title_keywords = set(self._extract_keywords(title_tokens))
        query_keywords = set(keywords)
        
        if not query_keywords:
            return 0.0
        
        overlap = len(query_keywords.intersection(title_keywords))
        return min(1.0, overlap / len(query_keywords))
    
    def _calculate_recency_boost(self, result: Dict) -> float:
        """Calculate recency boost (placeholder - would need timestamp metadata)."""
        # This would require timestamp metadata in the documents
        return 0.0  # Placeholder
    
    def _calculate_exact_match_boost(self, keywords: List[str], text: str) -> float:
        """Calculate boost for exact keyword matches."""
        if not keywords or not text:
            return 0.0
        
        text_lower = text.lower()
        exact_matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return min(1.0, exact_matches / len(keywords))
    
    def _calculate_source_quality(self, result: Dict) -> float:
        """Calculate source quality score based on metadata."""
        meta = result.get("meta", {})
        
        # Simple quality scoring based on available metadata
        quality_score = 0.5  # Base score
        
        # Boost for having a title
        if meta.get("title"):
            quality_score += 0.2
        
        # Boost for local documentation (typically higher quality)
        if meta.get("host") == "localhost":
            quality_score += 0.2
        
        # Boost for certain file types or paths that indicate documentation
        path = meta.get("path", "")
        if any(pattern in path.lower() for pattern in ["docs", "readme", "guide", "tutorial"]):
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _update_average_results(self, result_count: int):
        """Update running average of results returned."""
        total = self.query_stats["successful_retrievals"]
        if total == 1:
            self.query_stats["average_results_returned"] = result_count
        else:
            current_avg = self.query_stats["average_results_returned"]
            self.query_stats["average_results_returned"] = (
                (current_avg * (total - 1) + result_count) / total
            )
    
    def _check_embedding_providers(self) -> Dict:
        """Check status of embedding providers."""
        providers = self.provider_registry.build_fallback_chain(ProviderType.EMBEDDING)
        return {
            "available_providers": len(providers),
            "provider_names": [p.get_config().name for p in providers]
        }