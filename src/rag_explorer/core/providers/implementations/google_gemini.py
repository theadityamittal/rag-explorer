"""Google Gemini provider implementations with free and paid tiers."""

import logging
from typing import List, Optional

from ..base import (
    CombinedProvider, ProviderError, ProviderUnavailableError
)

from ....utils.settings import GEMINI_LLM_MODEL, GEMINI_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class GoogleGeminiProvider(CombinedProvider):
    """Base class for Google Gemini providers."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Google Gemini provider.
        
        Args:
            api_key: Google API key
            **kwargs: Additional configuration options
        """
        try:
            from google import genai
            self.genai = genai
        except ImportError:
            raise ProviderUnavailableError(
                "Google GenerativeAI SDK not available. Install with: pip install -q -U google-genai",
                provider="google"
            )
        
        super().__init__(api_key=api_key, **kwargs)
        
        
        # Default models from settings

        self.default_llm_model = GEMINI_LLM_MODEL
        self.default_embedding_model = GEMINI_EMBEDDING_MODEL

        # Configure Google API
        try:
            if self.api_key:
                self.client = genai.Client(api_key=self.api_key)
        except Exception:
            self.client = None
            raise ProviderUnavailableError(
                "Google API is unable to connect. Verify the API key is configured",
                provider="google"
            )
        
        logger.info(f"Initialized Google Gemini Client with models: {self.default_llm_model}, {self.default_embedding_model}")
    
    def is_available(self) -> bool:
        """Check if Google Gemini provider is available."""
        if not self.api_key:
            return False
        
        try:
            # Test with model list
            if not self.client:
                raise ProviderUnavailableError(
                    "Gemini Provider Client is unavailable. Verify API key",
                    provider="google"
                )
            else:
                return True
        except ProviderUnavailableError as e:
            logger.debug(f"Provider Unavailable Error: {e}")
            return False
    
    def chat(self, 
             system_prompt: str, 
             user_prompt: str,
             model: Optional[str] = None,
             temperature: float = 0.0,
             **kwargs) -> str:
        """Generate chat completion using Google Gemini.
        
        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Specific model to use (defaults to configured model)
            temperature: Randomness in generation (0.0-1.0)
            max_tokens: Maximum tokens to generate (max_output_tokens in Gemini)
            **kwargs: Additional Gemini API parameters
            
        Returns:
            Generated response text
            
        Raises:
            ProviderError: If API call fails
            ProviderRateLimitError: If rate limit exceeded
        """
        final_response = ""
        try:
            from google.genai import types

            if not self.client:
                raise ProviderUnavailableError("Google Gemini Client not available", provider="google")
            else:
                
                # Configure generation settings
                generation_config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                )
                
                response = self.client.models.generate_content(
                    model=self.default_llm_model,
                    config=generation_config,
                    contents=user_prompt
                )

                if not response or not response.text:
                    raise ProviderError("Empty response from Gemini", provider="google")
                else:
                    final_response = response.text
                
        except Exception as e:
            error_str = str(e).lower()

            # Check for API errors
            if 'api' in error_str or 'forbidden' in error_str or 'unauthorized' in error_str:
                raise ProviderError(f"Google Gemini API error: {e}", provider="google", original_error=e)
            
            # Generic error
            raise ProviderError(f"Google Gemini chat failed: {e}", provider="google_gemini", original_error=e)
        finally:
            return final_response
    
    def embed_texts(self, 
                   texts: List[str],
                   model: Optional[str] = None,
                   batch_size: int = 10) -> List[List[float]]:
        """Generate embeddings for multiple texts using Google Gemini.
        
        Args:
            texts: List of texts to embed
            model: Specific embedding model to use
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
            
        Raises:
            ProviderError: If API call fails
        """
        if not texts:
            return []
        
        model = model or self.default_embedding_model
        embeddings = []
        
        try:
            if not self.client:
                raise ProviderUnavailableError(
                    message="Google Gemini Provider client is not set",
                    provider="google"
                )
            
            else:
                from google.genai import types

                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    result = self.client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=batch,
                        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
                    ).embeddings

                    if result:
                        embeddings.extend([e.values for e in result])
            
            return embeddings
            
        except ProviderUnavailableError as e:
            raise ProviderError(message=e, provider="google")
        except Exception as e:  
            raise ProviderError(f"Google Gemini embeddings failed: {e}", provider="google", original_error=e)
    
    def embed_one(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding for single text using Google Gemini.
        
        Args:
            text: Text to embed
            model: Specific embedding model to use
        Returns:
            Embedding vector
        """
        DIMENSION = self.get_embedding_dimension(model=model)

        if not text.strip():
            return [0.0] * DIMENSION  # Default dimension for empty text
        
        embeddings = self.embed_texts([text], model=model)
        return embeddings[0] if embeddings else [0.0] * DIMENSION
    
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get embedding dimension for the specified model.
        
        Args:
            model: Embedding model name
            
        Returns:
            Embedding vector dimension
        """
        model = model or self.default_embedding_model
        
        # Google embedding model dimensions
        dimensions = {
            'gemini-embedding-001': 768,
            'text-embedding-004': 768,
            'embedding-001': 768,
        }
        
        return dimensions.get(model, 768)  # Default dimension

    def retrieve_best_embeddings(self,
                                query_embedding: List[float],
                                top_k: int = 5,
                                similarity_threshold: float = 0.7,
                                **kwargs) -> List[Dict[str, Any]]:
        """Retrieve the best matching embeddings using Google Gemini's capabilities.

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
            raise ProviderUnavailableError("Google Gemini provider not available", provider="google")

        try:
            # Get database connection from kwargs
            database = kwargs.get('database')
            collection_name = kwargs.get('collection_name', 'embeddings')

            if not database:
                raise ProviderError("Database connection required for embedding retrieval", provider="google")

            # Normalize query embedding for cosine similarity
            query_embedding = np.array(query_embedding)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                raise ValueError("Query embedding cannot be zero vector")
            query_embedding = query_embedding / query_norm

            results = []

            # Google Cloud specific integrations
            if hasattr(database, 'search_ann'):
                # For Google Cloud Vector Search or similar services
                search_results = database.search_ann(
                    query_vector=query_embedding.tolist(),
                    num_neighbors=top_k,
                    min_score=similarity_threshold,
                    index_name=collection_name
                )

                for result in search_results:
                    results.append({
                        'text': result.get('text', ''),
                        'embedding': result.get('embedding', []),
                        'similarity_score': result.get('distance', 0.0),
                        'metadata': result.get('metadata', {})
                    })

            elif hasattr(database, 'vector_search'):
                # For Firestore or other Google Cloud databases with vector search
                search_results = database.vector_search(
                    collection=collection_name,
                    query_vector=query_embedding.tolist(),
                    limit=top_k,
                    similarity_threshold=similarity_threshold
                )

                for result in search_results:
                    results.append({
                        'text': result.get('content', ''),
                        'embedding': result.get('embedding_vector', []),
                        'similarity_score': result.get('similarity', 0.0),
                        'metadata': result.get('document_metadata', {})
                    })

            elif hasattr(database, 'query'):
                # For PostgreSQL with pgvector or similar
                try:
                    cursor = database.query(f"""
                        SELECT text, embedding, metadata,
                               1 - (embedding <=> %s) as similarity_score
                        FROM {collection_name}
                        WHERE 1 - (embedding <=> %s) >= %s
                        ORDER BY embedding <=> %s
                        LIMIT %s
                    """, [query_embedding.tolist(), query_embedding.tolist(),
                          similarity_threshold, query_embedding.tolist(), top_k])

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

            logger.debug(f"Google Gemini retrieved {len(results)} embeddings with similarity >= {similarity_threshold}")
            return results

        except Exception as e:
            if isinstance(e, (ValueError, ProviderError, ProviderUnavailableError)):
                raise
            raise ProviderError(f"Google Gemini embedding retrieval failed: {e}", provider="google", original_error=e)

    def _manual_similarity_search(self, database, collection_name: str,
                                query_embedding: np.ndarray, top_k: int,
                                similarity_threshold: float) -> List[Dict[str, Any]]:
        """Manual similarity search for databases without vector support."""
        import numpy as np

        logger.info("Using manual similarity computation for Google Gemini provider")

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