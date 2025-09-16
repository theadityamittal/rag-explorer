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
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ProviderUnavailableError(
                "Google GenerativeAI SDK not available. Install with: pip install google-generativeai",
                provider="google"
            )

        # Default models from settings
        self.default_llm_model = GEMINI_LLM_MODEL
        self.default_embedding_model = GEMINI_EMBEDDING_MODEL

        super().__init__(api_key=api_key, **kwargs)

        # Configure Google API
        try:
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.client = genai
        except Exception:
            self.client = None
            raise ProviderUnavailableError(
                "Google API is unable to connect. Verify the API key is configured",
                provider="google"
            )
        
        logger.info(f"Initialized Google Gemini Client with models: {self.default_llm_model}, {self.default_embedding_model}")

    def get_config(self):
        """Return provider configuration and capabilities."""
        from ..base import ProviderConfig, ProviderType
        return ProviderConfig(
            name="google",
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
                    "provider": "google"
                }

            # Try a simple API call to test connectivity
            test_result = self.embed_one("test")
            return {
                "status": "healthy",
                "message": "Provider is working correctly",
                "provider": "google",
                "embedding_dimension": len(test_result)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "provider": "google"
            }

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
            import google.generativeai.types as types

            if not self.client:
                raise ProviderUnavailableError("Google Gemini Client not available", provider="google")
            else:
                
                # Create model instance
                model = self.client.GenerativeModel(
                    model_name=self.default_llm_model,
                    system_instruction=system_prompt
                )

                # Configure generation settings
                generation_config = self.client.GenerationConfig(
                    temperature=temperature
                )

                response = model.generate_content(
                    user_prompt,
                    generation_config=generation_config
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
                for text in texts:
                    result = self.client.embed_content(
                        model=self.default_embedding_model,
                        content=text,
                        task_type="semantic_similarity"
                    )

                    if hasattr(result, 'embedding'):
                        embeddings.append(result.embedding)
                    elif 'embedding' in result:
                        embeddings.append(result['embedding'])
            
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

