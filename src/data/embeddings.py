import logging
from typing import List

try:
    # Try to use new provider system (preferred)
    from support_deflect_bot.core.providers import (
        get_default_registry,
        ProviderType,
        ProviderError,
        ProviderUnavailableError,
    )

    USE_NEW_SYSTEM = True
except ImportError:
    # Fallback to direct Ollama for backward compatibility
    import ollama
    from support_deflect_bot.utils.settings import OLLAMA_EMBED_MODEL as EMBED_MODEL

    USE_NEW_SYSTEM = False


def embed_texts(texts: List[str], batch_size: int = 10) -> List[List[float]]:
    """
    Batch embedder with chunking to avoid memory issues and improve performance.
    Uses Gemini as primary with Ollama fallback, or direct Ollama if new system unavailable.
    Processes texts in smaller batches to balance efficiency and memory usage.
    """
    if not texts:
        return []

    if USE_NEW_SYSTEM:
        return _embed_texts_new_system(texts, batch_size)
    else:
        return _embed_texts_ollama_direct(texts, batch_size)


def _embed_texts_new_system(
    texts: List[str], batch_size: int = 10
) -> List[List[float]]:
    """Embed texts using new multi-provider system with fallback chain."""
    registry = get_default_registry()

    # Build fallback chain for embedding providers (Gemini primary, Ollama fallback)
    chain = registry.build_fallback_chain(ProviderType.EMBEDDING)

    for provider in chain:
        try:
            return provider.embed_texts(texts, batch_size=batch_size)
        except (ProviderError, ProviderUnavailableError, Exception) as e:
            # Log error but continue to next provider
            logging.warning(f"Provider {provider.get_config().name} failed: {e}")
            continue

    # If all providers fail, return zero vectors
    logging.error("All embedding providers failed, returning zero vectors")
    return [[0.0] * 768 for _ in texts]


def _embed_texts_ollama_direct(
    texts: List[str], batch_size: int = 10
) -> List[List[float]]:
    """Direct Ollama embedding for backward compatibility."""
    vectors = []

    # Process in batches to avoid overwhelming the API
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            for text in batch:
                if not text.strip():  # Skip empty texts
                    vectors.append([0.0] * 768)  # Default dimension, adjust if needed
                    continue

                resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
                vectors.append(resp["embedding"])

        except Exception as e:
            logging.error(f"Failed to embed batch starting at index {i}: {e}")
            # Fallback: add zero vectors for failed batch
            for _ in batch:
                vectors.append([0.0] * 768)  # Default dimension

    return vectors


def embed_one(text: str) -> List[float]:
    """
    Embed a single text. Uses same provider system as embed_texts.
    """
    if not text.strip():
        return [0.0] * 768  # Default dimension for empty text

    try:
        return embed_texts([text], batch_size=1)[0]
    except Exception as e:
        logging.error(f"Failed to embed single text: {e}")
        return [0.0] * 768  # Default dimension as fallback
