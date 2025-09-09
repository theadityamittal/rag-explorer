"""Google Gemini provider implementations with free and paid tiers."""

import logging
import time
from typing import List, Optional, Dict, Any

from ..base import (
    CombinedProvider, ProviderConfig, ProviderType, ProviderTier,
    ProviderError, ProviderRateLimitError, ProviderUnavailableError
)

logger = logging.getLogger(__name__)


class GoogleGeminiBaseProvider(CombinedProvider):
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
                provider="google_gemini"
            )
        
        super().__init__(api_key=api_key, **kwargs)
        
        # Configure Google API
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        # Default models from settings
        from ....utils.settings import GOOGLE_MODEL
        self.default_llm_model = GOOGLE_MODEL
        self.default_embedding_model = "text-embedding-004"  # Google's embedding model
        
        # Initialize models
        try:
            self.llm_model = genai.GenerativeModel(self.default_llm_model)
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini LLM model: {e}")
            self.llm_model = None
        
        logger.info(f"Initialized Google Gemini provider with models: {self.default_llm_model}, {self.default_embedding_model}")
    
    def is_available(self) -> bool:
        """Check if Google Gemini provider is available."""
        if not self.api_key:
            return False
        
        try:
            # Test with model list
            models = list(self.genai.list_models())
            return len(models) > 0
        except Exception as e:
            logger.debug(f"Google Gemini availability check failed: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            if not self.api_key:
                return {
                    'status': 'unhealthy',
                    'error': 'Google API key not configured',
                    'provider': 'google_gemini'
                }
            
            # Test API connectivity
            start_time = time.time()
            models = list(self.genai.list_models())
            response_time = time.time() - start_time
            
            # Check if required models are available
            model_names = [model.name.split('/')[-1] for model in models]
            has_llm_model = any(self.default_llm_model in name for name in model_names)
            has_embedding_model = any(self.default_embedding_model in name for name in model_names)
            
            return {
                'status': 'healthy' if (has_llm_model and has_embedding_model) else 'degraded',
                'response_time_ms': round(response_time * 1000, 2),
                'models_available': len(models),
                'default_llm_available': has_llm_model,
                'default_embedding_available': has_embedding_model,
                'provider': 'google_gemini',
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'provider': 'google_gemini',
                'timestamp': time.time()
            }
    
    def chat(self, 
             system_prompt: str, 
             user_prompt: str,
             model: Optional[str] = None,
             temperature: float = 0.0,
             max_tokens: Optional[int] = None,
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
        if not self.llm_model:
            raise ProviderUnavailableError("Google Gemini model not available", provider="google_gemini")
        
        try:
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            
            # Configure generation settings
            generation_config = self.genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
            
            response = self.llm_model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            if response.parts:
                return response.text.strip()
            else:
                raise ProviderError("Empty response from Gemini", provider="google_gemini")
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for rate limit errors
            if 'quota' in error_str or 'rate limit' in error_str:
                raise ProviderRateLimitError(f"Google Gemini rate limit exceeded: {e}", provider="google_gemini", original_error=e)
            
            # Check for API errors
            if 'api' in error_str or 'forbidden' in error_str or 'unauthorized' in error_str:
                raise ProviderError(f"Google Gemini API error: {e}", provider="google_gemini", original_error=e)
            
            # Generic error
            raise ProviderError(f"Google Gemini chat failed: {e}", provider="google_gemini", original_error=e)
    
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
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                for text in batch:
                    if not text.strip():
                        # Use zero vector for empty text
                        embeddings.append([0.0] * 768)  # Default dimension
                        continue
                    
                    result = self.genai.embed_content(
                        model=f"models/{model}",
                        content=text
                    )
                    
                    embeddings.append(result['embedding'])
            
            return embeddings
            
        except Exception as e:
            error_str = str(e).lower()
            
            if 'quota' in error_str or 'rate limit' in error_str:
                raise ProviderRateLimitError(f"Google Gemini embedding rate limit: {e}", provider="google_gemini", original_error=e)
            
            raise ProviderError(f"Google Gemini embeddings failed: {e}", provider="google_gemini", original_error=e)
    
    def embed_one(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding for single text using Google Gemini.
        
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
        
        # Google embedding model dimensions
        dimensions = {
            'text-embedding-004': 768,
            'embedding-001': 768,
        }
        
        return dimensions.get(model, 768)  # Default dimension
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text using Google's token counting if available.
        
        Args:
            text: Text to count tokens for
            model: Model to use for tokenization
            
        Returns:
            Number of tokens
        """
        model = model or self.default_llm_model
        
        try:
            # Try to use Gemini's token counting if available
            if self.llm_model:
                response = self.llm_model.count_tokens(text)
                return response.total_tokens
        except Exception as e:
            logger.debug(f"Google token counting failed: {e}")
        
        # Fallback to estimation
        return self.estimate_tokens(text)


class GoogleGeminiFreeProvider(GoogleGeminiBaseProvider):
    """Google Gemini free tier provider (restricted regions)."""
    
    def get_config(self) -> ProviderConfig:
        """Get Google Gemini free tier configuration."""
        return ProviderConfig(
            name="Google Gemini (Free)",
            provider_type=ProviderType.BOTH,
            cost_per_million_tokens_input=0.0,     # FREE tier
            cost_per_million_tokens_output=0.0,
            max_context_length=1000000,            # 1M context window
            rate_limit_rpm=60,                     # Free tier limit
            rate_limit_tpm=60000,                  # Free tier limit
            supports_streaming=True,
            requires_api_key=True,
            tier=ProviderTier.FREE,
            regions_supported=['US', 'CA', 'AU', 'JP'],  # Restricted in GDPR regions
            gdpr_compliant=False,                  # Free tier not GDPR compliant
            models_available=['gemini-2.5-pro', 'text-embedding-004']
        )


class GoogleGeminiPaidProvider(GoogleGeminiBaseProvider):
    """Google Gemini paid tier provider (globally compliant)."""
    
    def get_config(self) -> ProviderConfig:
        """Get Google Gemini paid tier configuration."""
        return ProviderConfig(
            name="Google Gemini (Paid)",
            provider_type=ProviderType.BOTH,
            cost_per_million_tokens_input=7.0,     # Paid tier pricing (Gemini Pro)
            cost_per_million_tokens_output=21.0,   # Higher output cost
            max_context_length=1000000,            # 1M context window
            rate_limit_rpm=300,                    # Higher rate limits
            rate_limit_tpm=4000000,                # Higher token limits
            supports_streaming=True,
            requires_api_key=True,
            tier=ProviderTier.PAID,
            regions_supported=['global'],          # Works everywhere
            gdpr_compliant=True,                   # Paid tier is GDPR compliant
            models_available=['gemini-2.5-pro', 'text-embedding-004']
        )