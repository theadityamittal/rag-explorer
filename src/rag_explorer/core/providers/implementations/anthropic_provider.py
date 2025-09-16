"""Claude API provider implementation using direct Anthropic API."""

import logging
import time
from typing import List, Optional, Dict, Any

from ..base import (
    LLMProvider, ProviderConfig, ProviderType,
    ProviderError, ProviderUnavailableError
)

from ....utils.settings import ANTHROPIC_LLM_MODEL

logger = logging.getLogger(__name__)


class ClaudeAPIProvider(LLMProvider):
    """Claude API provider using direct Anthropic API - Premium cost-effective option."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize Claude API provider.
        
        Args:
            api_key: Anthropic API key
            **kwargs: Additional configuration options
        """
        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            raise ProviderUnavailableError(
                "Anthropic SDK not available. Install with: pip install anthropic",
                provider="anthropic"
            )
        
        super().__init__(api_key=api_key, **kwargs)

        self.default_llm_model = ANTHROPIC_LLM_MODEL
        
        # Initialize Anthropic client
        try:

            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
        except Exception:
            self.client = None
            raise ProviderUnavailableError(
                "Anthropic API is unable to connect. Verify the API key is configured",
                provider="anthropic"
            )
        
        logger.info(f"Initialized Claude API provider with model: {self.default_llm_model}")
    
    def is_available(self) -> bool:
        """Check if Anthropic API provider is available and properly configured."""
        if not self.api_key:
            return False
        
        try:
            # Test with model list
            if not self.client:
                raise ProviderUnavailableError(
                    "Anthropic Provider Client is unavailable. Verify API key",
                    provider="anthropic"
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
             max_tokens: Optional[int] = None,
             **kwargs) -> str:
        """Generate chat completion using Claude API.
        
        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Specific model to use (defaults to configured model)
            temperature: Randomness in generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic API parameters
            
        Returns:
            Generated response text
            
        Raises:
            ProviderError: If API call fails
            ProviderRateLimitError: If rate limit exceeded
        """
        if not self.client:
            raise ProviderUnavailableError("Anthropic API client not available", provider="anthropic")
        
        model = model or self.default_llm_model
        max_tokens = max_tokens or 4000  # Default max tokens for Claude
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                **kwargs
            )
            
            return response.content[0].text.strip()
            
        except self.anthropic.APIError as e:
            raise ProviderError(f"Anthropic API error: {e}", provider="anthropic", original_error=e)
        except Exception as e:
            raise ProviderError(f"Anthropic API chat failed: {e}", provider="anthropic", original_error=e)