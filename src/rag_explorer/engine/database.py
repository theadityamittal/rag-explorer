"""Simple ChromaDB functions for RAG Explorer engine.

This module provides simplified database operations without complex retry/circuit breaker patterns.
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from rag_explorer.utils.settings import CHROMA_DB_PATH, CHROMA_COLLECTION

logger = logging.getLogger(__name__)


def get_simple_client() -> chromadb.PersistentClient:
    """Get a simple ChromaDB client without connection pooling."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        return client
    except Exception as e:
        raise ConnectionError(f"Failed to connect to ChromaDB: {e}")


def get_or_create_collection(client: chromadb.PersistentClient, collection_name: str = CHROMA_COLLECTION):
    """Get or create a ChromaDB collection."""
    try:
        return client.get_or_create_collection(name=collection_name)
    except Exception as e:
        raise RuntimeError(f"Failed to get/create collection '{collection_name}': {e}")


def simple_add_documents_with_embeddings(
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    collection_name: str = CHROMA_COLLECTION
) -> Dict[str, Any]:
    """Add documents with embeddings to ChromaDB collection.

    Args:
        texts: List of document texts
        embeddings: List of embedding vectors
        metadatas: Optional metadata for each document
        collection_name: Name of the collection

    Returns:
        Dictionary with operation results

    Raises:
        ValueError: If input validation fails
        RuntimeError: If database operation fails
    """
    if not texts or not embeddings:
        raise ValueError("Texts and embeddings cannot be empty")

    if len(texts) != len(embeddings):
        raise ValueError("Number of texts must match number of embeddings")

    if metadatas and len(metadatas) != len(texts):
        raise ValueError("Number of metadatas must match number of texts")

    try:
        client = get_simple_client()
        collection = get_or_create_collection(client, collection_name)

        # Generate unique IDs for documents
        ids = [f"doc_{i}_{hash(text[:100])}" for i, text in enumerate(texts)]

        # Add documents to collection
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(texts),
            ids=ids
        )

        logger.info(f"Added {len(texts)} documents to collection '{collection_name}'")
        return {
            "status": "success",
            "documents_added": len(texts),
            "collection": collection_name
        }

    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        raise RuntimeError(f"Database operation failed: {e}")


def simple_query_by_embedding(
    query_embedding: List[float],
    top_k: int = 5,
    collection_name: str = CHROMA_COLLECTION
) -> List[Dict[str, Any]]:
    """Query documents by embedding similarity.

    Args:
        query_embedding: Query embedding vector
        top_k: Number of results to return
        collection_name: Name of the collection

    Returns:
        List of matching documents with metadata

    Raises:
        ValueError: If input validation fails
        RuntimeError: If database operation fails
    """
    if not query_embedding:
        raise ValueError("Query embedding cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    try:
        client = get_simple_client()
        collection = get_or_create_collection(client, collection_name)

        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                result = {
                    'text': doc,
                    'similarity_score': 1.0 - results['distances'][0][i] if results['distances'] else 0.0,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'id': results['ids'][0][i] if results['ids'] else f"doc_{i}"
                }
                formatted_results.append(result)

        logger.debug(f"Retrieved {len(formatted_results)} documents from collection '{collection_name}'")
        return formatted_results

    except Exception as e:
        logger.error(f"Failed to query documents: {e}")
        raise RuntimeError(f"Database query failed: {e}")


def simple_get_collection_count(collection_name: str = CHROMA_COLLECTION) -> int:
    """Get the number of documents in a collection.

    Args:
        collection_name: Name of the collection

    Returns:
        Number of documents in the collection

    Raises:
        RuntimeError: If database operation fails
    """
    try:
        client = get_simple_client()
        collection = get_or_create_collection(client, collection_name)

        count = collection.count()
        logger.debug(f"Collection '{collection_name}' contains {count} documents")
        return count

    except Exception as e:
        logger.error(f"Failed to get collection count: {e}")
        raise RuntimeError(f"Database operation failed: {e}")


def simple_reset_collection(collection_name: str = CHROMA_COLLECTION) -> Dict[str, Any]:
    """Reset (delete and recreate) a collection.

    Args:
        collection_name: Name of the collection to reset

    Returns:
        Dictionary with operation results

    Raises:
        RuntimeError: If database operation fails
    """
    try:
        client = get_simple_client()

        # Try to delete existing collection
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection '{collection_name}'")
        except ValueError:
            # Collection doesn't exist, which is fine
            pass

        # Create new collection
        collection = client.create_collection(name=collection_name)

        logger.info(f"Created new collection '{collection_name}'")
        return {
            "status": "success",
            "action": "reset",
            "collection": collection_name
        }

    except Exception as e:
        logger.error(f"Failed to reset collection: {e}")
        raise RuntimeError(f"Database operation failed: {e}")


def simple_collection_exists(collection_name: str = CHROMA_COLLECTION) -> bool:
    """Check if a collection exists.

    Args:
        collection_name: Name of the collection to check

    Returns:
        True if collection exists, False otherwise
    """
    try:
        client = get_simple_client()
        collections = client.list_collections()
        return any(col.name == collection_name for col in collections)

    except Exception as e:
        logger.warning(f"Failed to check collection existence: {e}")
        return False