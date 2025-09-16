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

    def get_config(self):
        """Return provider configuration and capabilities."""
        from ..base import ProviderConfig, ProviderType
        return ProviderConfig(
            name="openai",
            provider_type=ProviderType.BOTH,
            requires_api_key=True
        )

    def health_check(self):
        """Perform health check and return detailed status."""
        try:
            if not self.is_available():
                return {
                    "status": "unavailable",
                    "message": "Provider not available",
                    "provider": "openai"
                }

            # Try a simple API call to test connectivity
            test_result = self.embed_one("test")
            return {
                "status": "healthy",
                "message": "Provider is working correctly",
                "provider": "openai",
                "embedding_dimension": len(test_result)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "provider": "openai"
            }

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

