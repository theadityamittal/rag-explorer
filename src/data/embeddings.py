import logging
from typing import List

import ollama

from src.utils.settings import OLLAMA_EMBED_MODEL as EMBED_MODEL


def embed_texts(texts: List[str], batch_size: int = 10) -> List[List[float]]:
    """
    Batch embedder with chunking to avoid memory issues and improve performance.
    Processes texts in smaller batches to balance efficiency and memory usage.
    """
    if not texts:
        return []

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
    Embed a single text. Optimized to call embed_texts with batch_size=1.
    """
    if not text.strip():
        return [0.0] * 768  # Default dimension for empty text

    try:
        return embed_texts([text], batch_size=1)[0]
    except Exception as e:
        logging.error(f"Failed to embed single text: {e}")
        return [0.0] * 768  # Default dimension as fallback
