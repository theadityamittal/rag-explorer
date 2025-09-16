"""Ollama provider implementation for local models - backward compatibility."""

import logging
import time
import os
from typing import List, Optional, Dict, Any

from ..base import (
    CombinedProvider, ProviderConfig, ProviderType,
    ProviderError, ProviderUnavailableError
)

from ....utils.settings import OLLAMA_LLM_MODEL, OLLAMA_EMBEDDING_MODEL, OLLAMA_HOST

logger = logging.getLogger(__name__)

class OllamaProvider(CombinedProvider):
    """Local Ollama provider for backward compatibility and privacy-focused deployment."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Ollama provider.
        
        Args:
            api_key: Not used for Ollama (local deployment)
            **kwargs: Additional configuration options including host
        """
        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            raise ProviderUnavailableError(
                "Ollama SDK not available. Install with: pip install ollama",
                provider="ollama"
            )
        
        super().__init__(api_key=None, **kwargs)  # No API key needed for local
        
        # Get Ollama configuration from settings
        
        self.default_llm_model = OLLAMA_LLM_MODEL
        self.default_embedding_model = OLLAMA_EMBEDDING_MODEL
        self.ollama_host = kwargs.get('host', OLLAMA_HOST)

        self.ollama.pull(self.default_llm_model)
        self.ollama.pull(self.default_embedding_model)
        
        # Configure Ollama host if specified
        if self.ollama_host:
            os.environ['OLLAMA_HOST'] = self.ollama_host
        
        logger.info(f"Initialized Ollama provider with models: {self.default_llm_model}, {self.default_embedding_model}")
        if self.ollama_host:
            logger.info(f"Using Ollama host: {self.ollama_host}")
    
    def is_available(self) -> bool:
        """Check if Ollama is available and running."""
        try:
            # Test connection to Ollama service
            models = self.ollama.list()
            return True
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            return False
    
    def chat(self, 
             system_prompt: str, 
             user_prompt: str,
             model: Optional[str] = None,
             temperature: float = 0.0,
             max_tokens: Optional[int] = None,
             **kwargs) -> str:
        """Generate chat completion using local Ollama models.
        
        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Specific model to use (defaults to configured model)
            temperature: Randomness in generation (0.0-1.0)
            max_tokens: Maximum tokens to generate (num_predict in Ollama)
            **kwargs: Additional Ollama parameters
            
        Returns:
            Generated response text
            
        Raises:
            ProviderError: If Ollama call fails
            ProviderUnavailableError: If Ollama not available
        """
        if not self.is_available():
            raise ProviderUnavailableError("Ollama service not available", provider="ollama")
        
        model = model or self.default_llm_model
        
        try:
            # Format messages for Ollama
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Ollama API options
            options = {
                "temperature": temperature,
            }
            
            if max_tokens:
                options["num_predict"] = max_tokens
            
            # Add any additional options
            options.update(kwargs)
            
            # Make the API call
            response = self.ollama.chat(
                model=model,
                messages=messages,
                options=options
            )
            
            if 'message' in response and 'content' in response['message']:
                return response['message']['content'].strip()
            else:
                raise ProviderError("Unexpected response format from Ollama", provider="ollama")
            
        except Exception as e:
            if isinstance(e, ProviderUnavailableError):
                raise
            raise ProviderError(f"Ollama chat failed: {e}", provider="ollama", original_error=e)
    
    def embed_texts(self, 
                   texts: List[str],
                   model: Optional[str] = None,
                   batch_size: int = 10) -> List[List[float]]:
        """Generate embeddings for multiple texts using local Ollama models.
        
        Args:
            texts: List of texts to embed
            model: Specific embedding model to use
            batch_size: Number of texts to process at once (not strictly needed for local)
            
        Returns:
            List of embedding vectors
            
        Raises:
            ProviderError: If Ollama call fails
        """
        if not texts:
            return []
        
        if not self.is_available():
            raise ProviderUnavailableError("Ollama service not available", provider="ollama")
        
        model = model or self.default_embedding_model
        embeddings = []
        
        try:
            # Process texts (Ollama handles one at a time)
            for text in texts:
                if not text.strip():
                    # Use zero vector for empty text
                    embeddings.append([0.0] * 768)  # Default dimension
                    continue
                
                response = self.ollama.embeddings(model=model, prompt=text)
                
                if 'embedding' in response:
                    embeddings.append(response['embedding'])
                else:
                    # Fallback to zero vector
                    logger.warning(f"No embedding in Ollama response for text: {text[:50]}...")
                    embeddings.append([0.0] * 768)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Ollama embeddings failed: {e}")
            # Fallback: return zero vectors for all texts
            return [[0.0] * 768 for _ in texts]
    
    def embed_one(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding for single text using local Ollama.
        
        Args:
            text: Text to embed
            model: Specific embedding model to use
            
        Returns:
            Embedding vector
        """
        if not text.strip():
            return [0.0] * 768  # Default dimension for empty text
        
        embeddings = self.embed_texts([text], model=model)
        return embeddings[0] if embeddings else [0.0] * 768
    
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get embedding dimension for the specified model.
        
        Args:
            model: Embedding model name
            
        Returns:
            Embedding vector dimension
        """
        model = model or self.default_embedding_model
        
        # Common Ollama embedding model dimensions
        dimensions = {
            'nomic-embed-text': 768,
            'all-minilm': 384,
            'mxbai-embed-large': 1024,
        }
        
        return dimensions.get(model, 768)  # Default dimension
    
    def pull_model(self, model_name: str) -> bool:
        """Pull/download a model to local Ollama instance.
        
        Args:
            model_name: Name of model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.ollama.pull(model_name)
            logger.info(f"Successfully pulled Ollama model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull Ollama model {model_name}: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List available models in local Ollama instance.
        
        Returns:
            List of model names
        """
        try:
            models = self.ollama.list()
            if 'models' in models:
                return [model['name'] for model in models['models']]
            return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific model.
        
        Args:
            model: Model name to get info for
            
        Returns:
            Model information dictionary
        """
        model = model or self.default_llm_model
        
        try:
            # Try to get model info from Ollama
            models = self.ollama.list()
            if 'models' in models:
                for model_info in models['models']:
                    if model in model_info['name']:
                        return {
                            'name': model_info['name'],
                            'size': model_info.get('size', 0),
                            'modified_at': model_info.get('modified_at'),
                            'details': model_info.get('details', {}),
                            'provider': 'ollama',
                            'deployment': 'local'
                        }
        except Exception as e:
            logger.debug(f"Failed to get Ollama model info: {e}")
        
        # Fallback to basic info
        return {
            'name': model,
            'provider': 'ollama',
            'deployment': 'local',
            'cost': 'free',
            'privacy': 'high'
        }
    
    def stream_chat(self, 
                   system_prompt: str, 
                   user_prompt: str,
                   model: Optional[str] = None,
                   temperature: float = 0.0,
                   max_tokens: Optional[int] = None,
                   **kwargs):
        """Generate streaming chat completion using Ollama.
        
        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Specific model to use
            temperature: Randomness in generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Yields:
            Streaming response chunks
        """
        if not self.is_available():
            raise ProviderUnavailableError("Ollama service not available", provider="ollama")
        
        model = model or self.default_llm_model
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens
            options.update(kwargs)
            
            # Create streaming response
            stream = self.ollama.chat(
                model=model,
                messages=messages,
                stream=True,
                options=options
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            raise ProviderError(f"Ollama streaming failed: {e}", provider="ollama", original_error=e)

    def retrieve_best_embeddings(self,
                                query_embedding: List[float],
                                top_k: int = 5,
                                similarity_threshold: float = 0.7,
                                **kwargs) -> List[Dict[str, Any]]:
        """Retrieve the best matching embeddings using local similarity computation.

        Args:
            query_embedding: The query embedding vector to find matches for
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score to include
            **kwargs: Additional parameters (e.g., database, collection_name)

        Returns:
            List of dictionaries containing matched embeddings and metadata

        Raises:
            ProviderError: If retrieval operation fails
            ValueError: If parameters are invalid
        """
        import numpy as np

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        if not self.is_available():
            raise ProviderUnavailableError("Ollama service not available", provider="ollama")

        try:
            # Get database connection from kwargs
            database = kwargs.get('database')
            collection_name = kwargs.get('collection_name', 'embeddings')

            if not database:
                raise ProviderError("Database connection required for embedding retrieval", provider="ollama")

            # Normalize query embedding for cosine similarity
            query_embedding = np.array(query_embedding)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                raise ValueError("Query embedding cannot be zero vector")
            query_embedding = query_embedding / query_norm

            results = []

            # Ollama works best with local vector storage
            if hasattr(database, 'search_similar'):
                # For local vector databases optimized for Ollama
                search_results = database.search_similar(
                    query_vector=query_embedding.tolist(),
                    limit=top_k,
                    threshold=similarity_threshold,
                    collection=collection_name
                )

                for result in search_results:
                    results.append({
                        'text': result.get('text', ''),
                        'embedding': result.get('embedding', []),
                        'similarity_score': result.get('similarity', 0.0),
                        'metadata': result.get('metadata', {})
                    })

            elif hasattr(database, 'query'):
                # For SQLite or other local databases with vector extensions
                try:
                    cursor = database.query(f"""
                        SELECT text, embedding, metadata,
                               (1 - (embedding <-> ?)) as similarity_score
                        FROM {collection_name}
                        WHERE (1 - (embedding <-> ?)) >= ?
                        ORDER BY similarity_score DESC
                        LIMIT ?
                    """, [query_embedding.tolist(), query_embedding.tolist(),
                          similarity_threshold, top_k])

                    for row in cursor.fetchall():
                        results.append({
                            'text': row[0],
                            'embedding': row[1],
                            'similarity_score': row[3],
                            'metadata': row[2] or {}
                        })
                except Exception:
                    # Fallback to manual computation
                    logger.info("Vector query failed, using manual similarity computation")
                    results = self._manual_similarity_search(
                        database, collection_name, query_embedding,
                        top_k, similarity_threshold
                    )
            else:
                # Manual similarity computation for simple storage
                results = self._manual_similarity_search(
                    database, collection_name, query_embedding,
                    top_k, similarity_threshold
                )

            logger.debug(f"Ollama retrieved {len(results)} embeddings with similarity >= {similarity_threshold}")
            return results

        except Exception as e:
            if isinstance(e, (ValueError, ProviderError, ProviderUnavailableError)):
                raise
            raise ProviderError(f"Ollama embedding retrieval failed: {e}", provider="ollama", original_error=e)

    def _manual_similarity_search(self, database, collection_name: str,
                                query_embedding: np.ndarray, top_k: int,
                                similarity_threshold: float) -> List[Dict[str, Any]]:
        """Manual similarity search for databases without vector support."""
        import numpy as np

        logger.info("Using manual similarity computation for Ollama provider")

        # Get all embeddings from database
        all_embeddings = database.get_all_embeddings(collection_name)
        similarities = []

        for item in all_embeddings:
            try:
                stored_embedding = np.array(item['embedding'])
                stored_norm = np.linalg.norm(stored_embedding)

                if stored_norm > 0:
                    stored_embedding = stored_embedding / stored_norm
                    similarity = np.dot(query_embedding, stored_embedding)

                    if similarity >= similarity_threshold:
                        similarities.append({
                            'text': item.get('text', ''),
                            'embedding': item['embedding'],
                            'similarity_score': float(similarity),
                            'metadata': item.get('metadata', {})
                        })
            except Exception as e:
                logger.warning(f"Failed to compute similarity for item: {e}")
                continue

        # Sort by similarity and take top_k
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:top_k]