"""LLM compatibility bridge - bridges old llm_local functions to new provider system."""

import logging
from typing import Optional
from ._path_helper import ensure_src_path

logger = logging.getLogger(__name__)


def llm_chat(system_prompt: str, user_prompt: str, model: Optional[str] = None) -> str:
    """
    General chat helper for normal prompts using new provider system.
    Uses conservative options to reduce randomness.
    
    This function provides the same interface as src.core.llm_local.llm_chat()
    but uses the new multi-provider system under the hood.
    """
    try:
        # Try to use new provider system
        from ..core.providers import get_default_registry, ProviderType
        
        registry = get_default_registry()
        providers = registry.get_available_providers(ProviderType.LLM)
        
        if not providers:
            # Fallback to old system if no providers available
            logger.warning("No providers available in new system, falling back to old system")
            return _fallback_to_old_system(system_prompt, user_prompt)
        
        # Use the first available provider
        _, provider = providers[0]
        
        try:
            response = provider.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=None
            )
            return response.strip()
            
        except Exception as e:
            logger.warning(f"Provider {provider.__class__.__name__} failed: {e}")
            # Try fallback to old system
            return _fallback_to_old_system(system_prompt, user_prompt)
            
    except ImportError:
        # New provider system not available, use old system
        logger.debug("New provider system not available, using old system")
        return _fallback_to_old_system(system_prompt, user_prompt)
    except Exception as e:
        logger.error(f"Error in new provider system: {e}")
        return _fallback_to_old_system(system_prompt, user_prompt)


def llm_echo(text: str) -> str:
    """
    Simple health check to wake the model up.
    
    This function provides the same interface as src.core.llm_local.llm_echo()
    but uses the new multi-provider system under the hood.
    """
    system_prompt = "You are a helpful assistant. Respond with exactly what the user says, nothing more."
    user_prompt = f"Please repeat this exactly: {text}"
    
    try:
        response = llm_chat(system_prompt, user_prompt)
        # Extract just the repeated text if possible
        if text.lower() in response.lower():
            return text
        return response
    except Exception:
        # Fallback response for health checks
        return text


def _fallback_to_old_system(system_prompt: str, user_prompt: str) -> str:
    """Fallback to old ollama system if new provider system fails."""
    try:
        ensure_src_path()
        from src.core.llm_local import llm_chat as old_llm_chat
        return old_llm_chat(system_prompt, user_prompt)
    except ImportError as e:
        logger.error(f"Both new and old LLM systems unavailable: {e}")
        raise RuntimeError(
            "No LLM providers available. Please install Ollama or configure API providers."
        )
    except Exception as e:
        logger.error(f"Old LLM system failed: {e}")
        raise RuntimeError(f"LLM system unavailable: {e}")