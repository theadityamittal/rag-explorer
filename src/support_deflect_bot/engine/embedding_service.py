"""Unified Embedding Service for Support Deflect Bot."""

import hashlib
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

try:
    from ..core.providers import get_default_registry, ProviderType, ProviderError, ProviderUnavailableError
except ImportError:
    # Fallback to old provider system during transition
    from support_deflect_bot_old.core.providers import get_default_registry, ProviderType, ProviderError, ProviderUnavailableError
try:
    from ..utils.settings import USER_AGENT
except ImportError:
    # Fallback to old settings during transition
    from support_deflect_bot_old.utils.settings import USER_AGENT


class UnifiedEmbeddingService:
    """
    Unified embedding service that handles embedding generation, caching,
    batch processing, and provider management.
    """
    
    def __init__(self, provider_registry=None, cache_enabled: bool = True, cache_path: Optional[str] = None):
        """Initialize embedding service with provider registry and configuration."""
        self.provider_registry = provider_registry or get_default_registry()
        self.cache_enabled = cache_enabled
        self.cache_path = cache_path or os.path.join(os.getcwd(), ".embedding_cache")
        
        # Create cache directory if enabled
        if self.cache_enabled:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        
        # Embedding metrics and analytics
        self.embedding_stats = {
            "total_embeddings_generated": 0,
            "cached_embeddings_served": 0,
            "failed_embeddings": 0,
            "batch_operations": 0,
            "average_batch_size": 0.0,
            "total_processing_time": 0.0,
            "last_embedding_time": None,
            "provider_usage": {}
        }
        
        # Dimension cache for different providers
        self._dimension_cache = {}
        
        # Load embedding cache if enabled
        self._embedding_cache = {}
        if self.cache_enabled:
            self._load_cache()

    def generate_embeddings(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 10,
        use_cache: bool = True,
        provider_preference: Optional[str] = None
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s) using provider fallback chain.
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing multiple texts
            use_cache: Whether to use cached embeddings
            provider_preference: Preferred provider name (optional)
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        start_time = time.time()
        is_single_text = isinstance(texts, str)
        text_list = [texts] if is_single_text else texts
        
        if not text_list:
            return [] if not is_single_text else []
        
        try:
            # Check cache first if enabled
            cached_embeddings = {}
            uncached_texts = []
            uncached_indices = []
            
            if use_cache and self.cache_enabled:
                for i, text in enumerate(text_list):
                    cache_key = self._get_cache_key(text)
                    if cache_key in self._embedding_cache:
                        cached_embeddings[i] = self._embedding_cache[cache_key]
                        self.embedding_stats["cached_embeddings_served"] += 1
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
            else:
                uncached_texts = text_list
                uncached_indices = list(range(len(text_list)))
            
            # Generate embeddings for uncached texts
            new_embeddings = {}
            if uncached_texts:
                generated = self._generate_embeddings_with_fallback(
                    uncached_texts, 
                    batch_size, 
                    provider_preference
                )
                
                if generated and len(generated) == len(uncached_texts):
                    for i, embedding in zip(uncached_indices, generated):
                        new_embeddings[i] = embedding
                        
                        # Cache the embedding if enabled
                        if self.cache_enabled:
                            cache_key = self._get_cache_key(uncached_texts[uncached_indices.index(i)])
                            self._embedding_cache[cache_key] = embedding
                    
                    self.embedding_stats["total_embeddings_generated"] += len(generated)
                else:
                    self.embedding_stats["failed_embeddings"] += len(uncached_texts)
                    return [] if not is_single_text else []
            
            # Combine cached and new embeddings in correct order
            all_embeddings = []
            for i in range(len(text_list)):
                if i in cached_embeddings:
                    all_embeddings.append(cached_embeddings[i])
                elif i in new_embeddings:
                    all_embeddings.append(new_embeddings[i])
                else:
                    # This shouldn't happen, but handle gracefully
                    self.embedding_stats["failed_embeddings"] += 1
                    return [] if not is_single_text else []
            
            # Update stats
            processing_time = time.time() - start_time
            self.embedding_stats["total_processing_time"] += processing_time
            self.embedding_stats["last_embedding_time"] = datetime.now().isoformat()
            
            # Save cache if enabled and there were new embeddings
            if self.cache_enabled and new_embeddings:
                self._save_cache()
            
            return all_embeddings[0] if is_single_text else all_embeddings
            
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            self.embedding_stats["failed_embeddings"] += len(text_list)
            return [] if not is_single_text else []

    def batch_embed(
        self, 
        texts: List[str], 
        batch_size: int = 10,
        show_progress: bool = False,
        fail_on_error: bool = False
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Process large batches of texts with progress tracking and error handling.
        
        Args:
            texts: List of texts to embed
            batch_size: Size of each processing batch
            show_progress: Whether to log progress
            fail_on_error: Whether to fail completely on any error
            
        Returns:
            Tuple of (embeddings, failed_indices)
        """
        if not texts:
            return [], []
        
        start_time = time.time()
        self.embedding_stats["batch_operations"] += 1
        self._update_average_batch_size(len(texts))
        
        all_embeddings = []
        failed_indices = []
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(texts), batch_size):
            batch_end = min(batch_idx + batch_size, len(texts))
            batch_texts = texts[batch_idx:batch_end]
            
            if show_progress:
                batch_num = (batch_idx // batch_size) + 1
                logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            
            try:
                batch_embeddings = self.generate_embeddings(
                    batch_texts, 
                    batch_size=batch_size,
                    use_cache=True
                )
                
                if batch_embeddings and len(batch_embeddings) == len(batch_texts):
                    all_embeddings.extend(batch_embeddings)
                else:
                    # Handle partial or complete batch failure
                    if fail_on_error:
                        raise RuntimeError(f"Batch {batch_num} failed to generate embeddings")
                    
                    # Add placeholder embeddings and track failures
                    for i in range(len(batch_texts)):
                        global_idx = batch_idx + i
                        failed_indices.append(global_idx)
                        all_embeddings.append([0.0] * self._get_default_dimension())
                
            except Exception as e:
                logging.error(f"Error processing batch {batch_num}: {e}")
                
                if fail_on_error:
                    raise
                
                # Add placeholder embeddings for entire failed batch
                for i in range(len(batch_texts)):
                    global_idx = batch_idx + i
                    failed_indices.append(global_idx)
                    all_embeddings.append([0.0] * self._get_default_dimension())
            
            # Small delay between batches to be respectful to API providers
            if batch_idx + batch_size < len(texts):
                time.sleep(0.1)
        
        processing_time = time.time() - start_time
        
        if show_progress:
            success_rate = (len(texts) - len(failed_indices)) / len(texts) * 100
            logging.info(
                f"Batch embedding completed: {len(texts)} texts, "
                f"{success_rate:.1f}% success rate, {processing_time:.2f}s"
            )
        
        return all_embeddings, failed_indices

    def get_embedding_dimension(self, provider_name: Optional[str] = None) -> int:
        """
        Get the embedding dimension for a specific provider or the default.
        
        Args:
            provider_name: Name of the provider (optional)
            
        Returns:
            Embedding dimension
        """
        if provider_name and provider_name in self._dimension_cache:
            return self._dimension_cache[provider_name]
        
        # Try to determine dimension by generating a test embedding
        try:
            test_embedding = self.generate_embeddings(
                "test", 
                use_cache=False,
                provider_preference=provider_name
            )
            
            if test_embedding and isinstance(test_embedding, list):
                dimension = len(test_embedding)
                
                # Cache the dimension
                if provider_name:
                    self._dimension_cache[provider_name] = dimension
                
                return dimension
                
        except Exception as e:
            logging.warning(f"Could not determine embedding dimension: {e}")
        
        # Return default dimension as fallback
        return self._get_default_dimension()

    def validate_providers(self, test_text: str = "test embedding") -> Dict[str, Dict]:
        """
        Validate all embedding providers and their capabilities.
        
        Args:
            test_text: Text to use for testing
            
        Returns:
            Dictionary with provider validation results
        """
        providers = self.provider_registry.build_fallback_chain(ProviderType.EMBEDDING)
        validation_results = {}
        
        for provider in providers:
            provider_name = provider.get_config().name
            result = {
                "available": False,
                "dimension": None,
                "response_time": None,
                "error": None
            }
            
            try:
                start_time = time.time()
                embeddings = provider.embed_texts([test_text], batch_size=1)
                response_time = time.time() - start_time
                
                if embeddings and len(embeddings) > 0 and len(embeddings[0]) > 0:
                    result["available"] = True
                    result["dimension"] = len(embeddings[0])
                    result["response_time"] = round(response_time, 3)
                    
                    # Cache the dimension
                    self._dimension_cache[provider_name] = result["dimension"]
                else:
                    result["error"] = "Empty or invalid embedding returned"
                    
            except Exception as e:
                result["error"] = str(e)
            
            validation_results[provider_name] = result
        
        return validation_results

    def cache_embeddings(
        self, 
        texts: List[str], 
        embeddings: List[List[float]],
        overwrite: bool = False
    ) -> int:
        """
        Manually cache embeddings for given texts.
        
        Args:
            texts: List of texts
            embeddings: Corresponding embedding vectors
            overwrite: Whether to overwrite existing cache entries
            
        Returns:
            Number of embeddings cached
        """
        if not self.cache_enabled:
            logging.warning("Embedding cache is disabled")
            return 0
        
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")
        
        cached_count = 0
        
        for text, embedding in zip(texts, embeddings):
            cache_key = self._get_cache_key(text)
            
            if cache_key not in self._embedding_cache or overwrite:
                self._embedding_cache[cache_key] = embedding
                cached_count += 1
        
        if cached_count > 0:
            self._save_cache()
        
        return cached_count

    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the embedding cache.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_size = len(self._embedding_cache) if self.cache_enabled else 0
        cache_file_size = 0
        
        if self.cache_enabled and os.path.exists(self.cache_path):
            cache_file_size = os.path.getsize(self.cache_path)
        
        return {
            "cache_enabled": self.cache_enabled,
            "cache_entries": cache_size,
            "cache_file_size_bytes": cache_file_size,
            "cache_file_size_mb": round(cache_file_size / (1024 * 1024), 2),
            "cache_path": self.cache_path if self.cache_enabled else None
        }

    def clear_cache(self) -> bool:
        """
        Clear the embedding cache.
        
        Returns:
            True if cache was cleared successfully
        """
        if not self.cache_enabled:
            return False
        
        try:
            self._embedding_cache = {}
            if os.path.exists(self.cache_path):
                os.remove(self.cache_path)
            return True
        except Exception as e:
            logging.error(f"Error clearing cache: {e}")
            return False

    def get_analytics(self) -> Dict:
        """
        Get comprehensive embedding service analytics.
        
        Returns:
            Dictionary with analytics and performance metrics
        """
        total_operations = (
            self.embedding_stats["total_embeddings_generated"] + 
            self.embedding_stats["cached_embeddings_served"]
        )
        
        cache_hit_rate = (
            self.embedding_stats["cached_embeddings_served"] / max(1, total_operations)
        ) if total_operations > 0 else 0.0
        
        average_time_per_embedding = (
            self.embedding_stats["total_processing_time"] / 
            max(1, self.embedding_stats["total_embeddings_generated"])
        ) if self.embedding_stats["total_embeddings_generated"] > 0 else 0.0
        
        return {
            **self.embedding_stats,
            "cache_hit_rate": round(cache_hit_rate, 3),
            "average_time_per_embedding": round(average_time_per_embedding, 4),
            "cache_stats": self.get_cache_stats(),
            "provider_validation": self.validate_providers(),
            "dimension_cache": self._dimension_cache
        }

    # Private helper methods
    
    def _generate_embeddings_with_fallback(
        self, 
        texts: List[str], 
        batch_size: int,
        provider_preference: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings using provider fallback chain."""
        providers = self.provider_registry.build_fallback_chain(ProviderType.EMBEDDING)
        
        # If preference specified, try that provider first
        if provider_preference:
            preferred_providers = [p for p in providers if p.get_config().name == provider_preference]
            other_providers = [p for p in providers if p.get_config().name != provider_preference]
            providers = preferred_providers + other_providers
        
        for provider in providers:
            provider_name = provider.get_config().name
            
            try:
                embeddings = provider.embed_texts(texts, batch_size=batch_size)
                
                if embeddings and len(embeddings) == len(texts):
                    # Track provider usage
                    if provider_name not in self.embedding_stats["provider_usage"]:
                        self.embedding_stats["provider_usage"][provider_name] = 0
                    self.embedding_stats["provider_usage"][provider_name] += len(texts)
                    
                    return embeddings
                    
            except (ProviderError, ProviderUnavailableError, Exception) as e:
                logging.warning(f"Embedding provider {provider_name} failed: {e}")
                continue
        
        # All providers failed
        raise RuntimeError("All embedding providers failed")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Use SHA256 hash of text for consistent caching
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def _get_default_dimension(self) -> int:
        """Get default embedding dimension."""
        # Common embedding dimensions
        return 768  # Default for many models
    
    def _load_cache(self):
        """Load embedding cache from file."""
        if not os.path.exists(self.cache_path):
            return
        
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                self._embedding_cache = json.load(f)
            logging.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
        except Exception as e:
            logging.warning(f"Could not load embedding cache: {e}")
            self._embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to file."""
        if not self.cache_enabled:
            return
        
        try:
            temp_path = self.cache_path + ".tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self._embedding_cache, f, ensure_ascii=False)
            os.replace(temp_path, self.cache_path)
        except Exception as e:
            logging.error(f"Could not save embedding cache: {e}")
    
    def _update_average_batch_size(self, batch_size: int):
        """Update running average batch size."""
        operations = self.embedding_stats["batch_operations"]
        if operations == 1:
            self.embedding_stats["average_batch_size"] = batch_size
        else:
            current_avg = self.embedding_stats["average_batch_size"]
            self.embedding_stats["average_batch_size"] = (
                (current_avg * (operations - 1) + batch_size) / operations
            )