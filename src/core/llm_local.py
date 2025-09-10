from typing import Optional

try:
    # Try to use new provider system (preferred)
    from support_deflect_bot.core.providers import (
        get_default_registry, 
        ProviderType,
        ProviderError,
        ProviderUnavailableError
    )
    USE_NEW_SYSTEM = True
except ImportError:
    # Fallback to direct Ollama for backward compatibility
    import ollama
    from support_deflect_bot.utils.settings import OLLAMA_MODEL
    USE_NEW_SYSTEM = False
    MODEL_NAME = OLLAMA_MODEL


def llm_chat(system_prompt: str, user_prompt: str) -> str:
    """
    General chat helper for normal prompts.
    Uses Gemini as primary with Ollama fallback, or direct Ollama if new system unavailable.
    Uses conservative options to reduce randomness.
    """
    if USE_NEW_SYSTEM:
        return _llm_chat_new_system(system_prompt, user_prompt)
    else:
        return _llm_chat_ollama_direct(system_prompt, user_prompt)


def _llm_chat_new_system(system_prompt: str, user_prompt: str) -> str:
    """Chat using new multi-provider system with fallback chain."""
    registry = get_default_registry()
    
    # Build fallback chain for LLM providers (Gemini primary, Ollama fallback)
    chain = registry.build_fallback_chain(ProviderType.LLM)
    
    for provider in chain:
        try:
            response = provider.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=None
            )
            return response
        except (ProviderError, ProviderUnavailableError, Exception) as e:
            # Log error but continue to next provider
            print(f"Provider {provider.get_config().name} failed: {e}")
            continue
    
    # If all providers fail, raise error
    raise RuntimeError("All LLM providers failed")


def _llm_chat_ollama_direct(system_prompt: str, user_prompt: str) -> str:
    """Direct Ollama chat for backward compatibility."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={
            "temperature": 0.0,
            "top_p": 0.0,
        },
    )
    return resp["message"]["content"].strip()


def llm_echo(text: str) -> str:
    """
    Simple health check to wake the model up.
    Uses same provider system as llm_chat.
    """
    system_prompt = (
        "You are a healthcheck. "
        "If you are awake, respond with 'Yeah Yeah! I'm awake!'"
    )
    return llm_chat(system_prompt, text)
