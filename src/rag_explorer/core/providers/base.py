"""Base classes for LLM and embedding providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Type of provider capability."""
    LLM = "llm"
    EMBEDDING = "embedding"
    BOTH = "both"


@dataclass
class ProviderConfig:
    """Configuration and metadata for a provider."""
    name: str
    provider_type: ProviderType
    requires_api_key: bool


class BaseProvider(ABC):
    """Base class for all providers."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.config = self.get_config()
        self._validate_initialization()
    
    @abstractmethod
    def get_config(self) -> ProviderConfig:
        """Return provider configuration and capabilities."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and properly configured."""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return detailed status."""
        pass
    
    def _validate_initialization(self):
        """Validate provider initialization."""
        if self.config.requires_api_key and not self.api_key:
            raise ValueError(f"API key required for {self.config.name}")
        
        logger.debug(f"Initialized provider: {self.config.name}")


class LLMProvider(BaseProvider):
    """Base class for LLM providers."""
    
    @abstractmethod
    def chat(self, 
             system_prompt: str, 
             user_prompt: str,
             model: Optional[str] = None,
             temperature: float = 0.0,
             **kwargs) -> str:
        """Generate chat completion.
        
        Args:
            system_prompt: System message to set behavior
            user_prompt: User's query or input
            model: Specific model to use (optional)
            temperature: Randomness in generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters
            
        Returns:
            Generated response text
            
        Raises:
            ProviderError: If API call fails
            ValueError: If parameters are invalid
        """
        pass


class EmbeddingProvider(BaseProvider):
    """Base class for embedding providers."""
    
    @abstractmethod
    def embed_texts(self, 
                   texts: List[str],
                   model: Optional[str] = None,
                   batch_size: int = 10) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            model: Specific model to use (optional)
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
            
        Raises:
            ProviderError: If API call fails
            ValueError: If texts are empty or invalid
        """
        pass
    
    @abstractmethod
    def embed_one(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding for single text.
        
        Args:
            text: Text to embed
            model: Specific model to use (optional)
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get embedding dimension for the model.

        Args:
            model: Model to check dimension for

        Returns:
            Embedding vector dimension
        """
        pass



class CombinedProvider(LLMProvider, EmbeddingProvider):
    """Provider that supports both LLM and embedding capabilities."""
    pass


class ProviderError(Exception):
    """Base exception for provider-related errors."""
    
    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(message)


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is unavailable or misconfigured."""
    pass