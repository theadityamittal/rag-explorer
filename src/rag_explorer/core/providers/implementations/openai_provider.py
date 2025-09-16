"""OpenAI provider implementation with GPT models and embeddings."""

import logging
import time
from typing import List, Optional, Dict, Any

from ..base import (
    CombinedProvider, ProviderConfig, ProviderType, ProviderError, ProviderUnavailableError
)

from ....utils.settings import OPENAI_LLM_MODEL, OPENAI_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class OpenAIProvider(CombinedProvider):
    """OpenAI provider with GPT models and embeddings - Primary legally compliant provider."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            **kwargs: Additional configuration options
        """
        try:
            import openai
            self.openai = openai
        except ImportError:
            raise ProviderUnavailableError(
                "OpenAI SDK not available. Install with: pip install openai",
                provider="openai"
            )
        
        super().__init__(api_key=api_key, **kwargs)

        # Default models from settings
        
        self.default_llm_model = OPENAI_LLM_MODEL
        self.default_embedding_model = OPENAI_EMBEDDING_MODEL
        
        # Initialize OpenAI client
        try:
            if self.api_key:
                self.client = openai.OpenAI(api_key=self.api_key)
        except:
            # Try to use default client (environment OPENAI_API_KEY)
            self.client = None
            raise ProviderUnavailableError(
                "Google API is unable to connect. Verify the API key is configured",
                provider="openai"
            )
        
        logger.info(f"Initialized OpenAI provider with models: {self.default_llm_model}, {self.default_embedding_model}")
    
    def is_available(self) -> bool:
        """Check if OpenAI provider is available and properly configured."""
        if not self.api_key:
            return False
        
        try:
            if not self.client:
                raise ProviderUnavailableError(
                    "Gemini Provider Client is unavailable. Verify API key",
                    provider="openai"
                )
            else:
                return True
        except Exception as e:
            logger.debug(f"OpenAI availability check failed: {e}")
            return False
    
    def chat(self, 
             system_prompt: str, 
             user_prompt: str,
             model: Optional[str] = None,
             temperature: float = 0.0,
             max_tokens: Optional[int] = None,
             **kwargs) -> str:
        """Generate chat completion using OpenAI GPT models.
        
        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Specific model to use (defaults to configured model)
            temperature: Randomness in generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI API parameters
            
        Returns:
            Generated response text
            
        Raises:
            ProviderError: If API call fails
            ProviderRateLimitError: If rate limit exceeded
        """
        final_response = ""
        try:

            if not self.client:
                raise ProviderUnavailableError("OpenAI client not available", provider="openai")
            
            else:
                model = model or self.default_llm_model

                response = self.client.responses.create(
                    model=self.default_llm_model,
                    reasoning={"effort": "low"},
                    temperature=temperature,
                    input=[
                        {
                            "role": "developer",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]
                )

                if not response or not response.output_text:
                    raise ProviderError("Empty response from OpenAI", provider="openai")
                else:
                    final_response = response.output_text
            
        except self.openai.APIError as e:
            raise ProviderError(f"OpenAI API error: {e}", provider="openai", original_error=e)
        except Exception as e:
            raise ProviderError(f"OpenAI chat failed: {e}", provider="openai", original_error=e)
        finally:
            return final_response
    
    def embed_texts(self, 
                   texts: List[str],
                   model: Optional[str] = None,
                   batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI.
        
        Args:
            texts: List of texts to embed
            model: Specific embedding model to use
            batch_size: Number of texts to process at once (OpenAI supports large batches)
            
        Returns:
            List of embedding vectors
            
        Raises:
            ProviderError: If API call fails
        """
        if not self.client:
            raise ProviderUnavailableError("OpenAI client not available", provider="openai")
        
        if not texts:
            return []
        
        model = model or self.default_embedding_model
        embeddings = []
        
        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = self.client.embeddings.create(
                    model=model,
                    input=batch
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except self.openai.APIError as e:
            raise ProviderError(f"OpenAI embeddings API error: {e}", provider="openai", original_error=e)
        except Exception as e:
            raise ProviderError(f"OpenAI embeddings failed: {e}", provider="openai", original_error=e)
    
    def embed_one(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding for single text using OpenAI.
        
        Args:
            text: Text to embed
            model: Specific embedding model to use
            
        Returns:
            Embedding vector
        """
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.get_embedding_dimension(model)
        
        embeddings = self.embed_texts([text], model=model)
        return embeddings[0] if embeddings else [0.0] * 1536  # Default dimension
    
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get embedding dimension for the specified model.
        
        Args:
            model: Embedding model name
            
        Returns:
            Embedding vector dimension
        """
        model = model or self.default_embedding_model
        
        # OpenAI embedding model dimensions
        dimensions = {
            'text-embedding-3-small': 768,
            'text-embedding-3-large': 1536,
            'text-embedding-ada-002': 1536,
        }
        
        return dimensions.get(model, 1536)  # Default to ada-002 dimension
        """Count tokens in text using tiktoken for accurate counting.
        
        Args:
            text: Text to count tokens for
            model: Model to use for tokenization
            
        Returns:
            Number of tokens
        """
        model = model or self.default_llm_model
        
        try:
            import tiktoken
            
            # Get encoding for the model
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
            
        except ImportError:
            logger.warning("tiktoken not available, using estimation")
            return self.estimate_tokens(text)
        except Exception as e:
            logger.warning(f"tiktoken failed: {e}, using estimation")
            return self.estimate_tokens(text)

    def retrieve_best_embeddings(self,
                                query_embedding: List[float],
                                top_k: int = 5,
                                similarity_threshold: float = 0.7,
                                **kwargs) -> List[Dict[str, Any]]:
        """Retrieve the best matching embeddings using OpenAI's vector similarity.

        Args:
            query_embedding: The query embedding vector to find matches for
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score to include
            **kwargs: Additional parameters (e.g., database_connection, collection_name)

        Returns:
            List of dictionaries containing matched embeddings and metadata

        Raises:
            ProviderError: If retrieval operation fails
            ValueError: If parameters are invalid
        """
        import numpy as np
        from typing import Tuple

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        try:
            # Get database connection from kwargs
            database = kwargs.get('database')
            collection_name = kwargs.get('collection_name', 'embeddings')

            if not database:
                raise ProviderError("Database connection required for embedding retrieval", provider="openai")

            # Normalize query embedding for cosine similarity
            query_embedding = np.array(query_embedding)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                raise ValueError("Query embedding cannot be zero vector")
            query_embedding = query_embedding / query_norm

            # Retrieve embeddings from database (implementation depends on database type)
            # This is a template that would need to be adapted for specific database
            results = []

            # Example implementation for a vector database
            if hasattr(database, 'search_vectors'):
                # For vector databases like Pinecone, Weaviate, etc.
                search_results = database.search_vectors(
                    vector=query_embedding.tolist(),
                    top_k=top_k,
                    min_score=similarity_threshold,
                    collection=collection_name
                )

                for result in search_results:
                    results.append({
                        'text': result.get('text', ''),
                        'embedding': result.get('embedding', []),
                        'similarity_score': result.get('score', 0.0),
                        'metadata': result.get('metadata', {})
                    })

            elif hasattr(database, 'query'):
                # For SQL-like databases with vector extensions
                cursor = database.query(f"""
                    SELECT text, embedding, metadata,
                           (embedding <=> %s) as similarity_score
                    FROM {collection_name}
                    WHERE (embedding <=> %s) < %s
                    ORDER BY similarity_score
                    LIMIT %s
                """, [query_embedding.tolist(), query_embedding.tolist(),
                      1.0 - similarity_threshold, top_k])

                for row in cursor.fetchall():
                    results.append({
                        'text': row[0],
                        'embedding': row[1],
                        'similarity_score': 1.0 - row[3],  # Convert distance to similarity
                        'metadata': row[2] or {}
                    })

            else:
                # Fallback: Manual similarity computation for simple storage
                logger.warning("Using fallback similarity computation - consider using a vector database")

                # Get all embeddings from database
                all_embeddings = database.get_all_embeddings(collection_name)
                similarities = []

                for item in all_embeddings:
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

                # Sort by similarity and take top_k
                similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
                results = similarities[:top_k]

            logger.debug(f"Retrieved {len(results)} embeddings with similarity >= {similarity_threshold}")
            return results

        except Exception as e:
            if isinstance(e, (ValueError, ProviderError)):
                raise
            raise ProviderError(f"OpenAI embedding retrieval failed: {e}", provider="openai", original_error=e)