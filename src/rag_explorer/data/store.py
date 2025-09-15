import logging
import os
import time
import uuid
from threading import Lock
from typing import Dict, List, Optional

import chromadb

from rag_explorer.utils.settings import (
    CHROMA_COLLECTION,
    CHROMA_DB_PATH,
    DB_POOL_SIZE,
    DB_CONNECTION_TIMEOUT,
    DB_POOL_CLEANUP_INTERVAL,
    DATABASE_QUERY_TIMEOUT
)
from rag_explorer.utils.stderr_suppressor import filter_stderr_lines
from rag_explorer.core.resilience import (
    retry_with_backoff,
    RetryPolicy,
    CircuitBreakerConfig,
    get_circuit_breaker,
    ErrorClassifier,
    ErrorType
)


logger = logging.getLogger(__name__)


class ChromaDBConnectionPool:
    """Connection pool for ChromaDB clients with health checking and automatic cleanup."""

    def __init__(self, max_connections: int = DB_POOL_SIZE):
        self.max_connections = max_connections
        self._connections: List[chromadb.PersistentClient] = []
        self._in_use: set = set()
        self._lock = Lock()
        self._last_cleanup = time.time()

    def get_connection(self) -> chromadb.PersistentClient:
        """Get a connection from the pool or create a new one."""
        with self._lock:
            # Clean up stale connections periodically
            if time.time() - self._last_cleanup > DB_POOL_CLEANUP_INTERVAL:
                self._cleanup_stale_connections()

            # Try to reuse existing connection
            for conn in self._connections[:]:
                if conn not in self._in_use:
                    if self._is_connection_healthy(conn):
                        self._in_use.add(conn)
                        return conn
                    else:
                        # Remove unhealthy connection
                        self._connections.remove(conn)

            # Create new connection if under limit
            if len(self._connections) < self.max_connections:
                conn = self._create_client()
                self._connections.append(conn)
                self._in_use.add(conn)
                return conn

            # Pool exhausted, create temporary connection
            logger.warning(f"ChromaDB connection pool exhausted, creating temporary connection")
            return self._create_client()

    def return_connection(self, connection: chromadb.PersistentClient):
        """Return a connection to the pool."""
        with self._lock:
            if connection in self._in_use:
                self._in_use.remove(connection)

    def _create_client(self) -> chromadb.PersistentClient:
        """Create a new ChromaDB client."""
        try:
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            with filter_stderr_lines():
                return chromadb.PersistentClient(path=CHROMA_DB_PATH)
        except Exception as e:
            logger.error(f"Failed to create ChromaDB client: {e}")
            raise ConnectionError(f"Database connection failed: {e}") from e

    def _is_connection_healthy(self, connection: chromadb.PersistentClient) -> bool:
        """Check if a connection is healthy."""
        try:
            # Simple health check - try to list collections
            with filter_stderr_lines():
                connection.list_collections()
            return True
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            return False

    def _cleanup_stale_connections(self):
        """Remove stale connections from the pool."""
        self._last_cleanup = time.time()

        # Remove unhealthy connections not in use
        for conn in self._connections[:]:
            if conn not in self._in_use and not self._is_connection_healthy(conn):
                self._connections.remove(conn)
                logger.info("Removed unhealthy connection from pool")

    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            # ChromaDB clients don't have explicit close methods
            # Clear the pools to release references
            self._connections.clear()
            self._in_use.clear()
            logger.info("Closed all connections in ChromaDB pool")


# Global connection pool instance
_connection_pool = ChromaDBConnectionPool()

# Circuit breaker for database operations
_db_circuit_breaker = get_circuit_breaker(
    "chromadb",
    CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=3,
        reset_timeout=60.0,
        half_open_max_calls=3
    )
)

# Retry policy for database operations
_db_retry_policy = RetryPolicy(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True
)


@retry_with_backoff(_db_retry_policy, _db_circuit_breaker)
def get_client() -> chromadb.PersistentClient:
    """Get a ChromaDB client from the connection pool with retry and circuit breaker protection."""
    return _connection_pool.get_connection()


def return_client(client: chromadb.PersistentClient):
    """Return a client to the connection pool."""
    _connection_pool.return_connection(client)


@retry_with_backoff(_db_retry_policy, _db_circuit_breaker)
def get_collection(client):
    """Get or create a collection with retry and circuit breaker protection.

    Args:
        client: ChromaDB client instance (required - use get_client() to obtain)
    """
    if client is None:
        raise ValueError("Client is required. Use get_client() to obtain a client and manage it properly.")

    try:
        # IMPORTANT: set HNSW space to cosine for text embeddings
        with filter_stderr_lines():
            collection = client.get_or_create_collection(
                name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
            )
        return collection
    except Exception as e:
        logger.error(f"Failed to get/create collection '{CHROMA_COLLECTION}': {e}")
        raise ConnectionError(f"Database collection access failed: {e}") from e


@retry_with_backoff(_db_retry_policy, _db_circuit_breaker)
def upsert_chunks(
    chunks: List[str],
    metadatas: List[Dict],
    ids: Optional[List[str]] = None,
):
    """
    Add/overwrite chunks with precomputed ids. If ids None, create uuids.
    NOTE: We embed outside chroma (we'll pass embeddings explicitly later).
    Enhanced with retry logic and circuit breaker protection.
    """
    if not chunks:
        logger.warning("Attempted to upsert empty chunks list")
        return

    client = get_client()
    try:
        coll = get_collection(client)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in chunks]

        # Validate inputs
        if len(chunks) != len(metadatas) or len(chunks) != len(ids):
            raise ValueError("Chunks, metadatas, and ids must have the same length")

        with filter_stderr_lines():
            coll.add(documents=chunks, metadatas=metadatas, ids=ids)

        logger.debug(f"Successfully upserted {len(chunks)} chunks")

    except Exception as e:
        logger.error(f"Failed to upsert {len(chunks)} chunks: {e}")
        # Classify error for retry decisions
        error_type = ErrorClassifier.classify_error(e)
        if error_type == ErrorType.PERMANENT:
            logger.error(f"Permanent error detected, not retrying: {e}")
        raise ConnectionError(f"Failed to store chunks in database: {e}") from e
    finally:
        return_client(client)


@retry_with_backoff(_db_retry_policy, _db_circuit_breaker)
def add_documents_with_embeddings(
    documents: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict],
    ids: Optional[List[str]] = None,
):
    """
    Add documents with precomputed embeddings to the collection.
    Enhanced with retry logic and circuit breaker protection.
    """
    if not documents:
        logger.warning("Attempted to add empty documents list")
        return

    client = get_client()
    try:
        coll = get_collection(client)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        # Validate inputs
        if len(documents) != len(embeddings) or len(documents) != len(metadatas) or len(documents) != len(ids):
            raise ValueError("Documents, embeddings, metadatas, and ids must have the same length")

        with filter_stderr_lines():
            coll.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

        logger.debug(f"Successfully added {len(documents)} documents with embeddings")

    except Exception as e:
        logger.error(f"Failed to add {len(documents)} documents: {e}")
        # Classify error for retry decisions
        error_type = ErrorClassifier.classify_error(e)
        if error_type == ErrorType.PERMANENT:
            logger.error(f"Permanent error detected, not retrying: {e}")
        raise ConnectionError(f"Failed to store documents in database: {e}") from e
    finally:
        return_client(client)


@retry_with_backoff(_db_retry_policy, _db_circuit_breaker)
def delete_by_where(where: Dict):
    """
    Delete documents matching the where clause.
    Enhanced with retry logic and circuit breaker protection.
    """
    if not where:
        raise ValueError("Where clause cannot be empty")

    client = get_client()
    try:
        coll = get_collection(client)
        with filter_stderr_lines():
            coll.delete(where=where)
        logger.debug(f"Successfully deleted documents matching where clause: {where}")

    except Exception as e:
        logger.error(f"Failed to delete documents with where clause {where}: {e}")
        # Classify error for retry decisions
        error_type = ErrorClassifier.classify_error(e)
        if error_type == ErrorType.PERMANENT:
            logger.error(f"Permanent error detected, not retrying: {e}")
        raise ConnectionError(f"Failed to delete documents from database: {e}") from e
    finally:
        return_client(client)


@retry_with_backoff(_db_retry_policy, _db_circuit_breaker)
def query_by_embedding(
    query_embedding: List[float], k: int = 5, where: Optional[dict] = None
):
    """
    Query embeddings with retry and circuit breaker protection.
    Enhanced with improved error handling and connection management.
    """
    if not query_embedding:
        raise ValueError("Query embedding cannot be empty")

    if k <= 0:
        raise ValueError("k must be positive")

    client = get_client()
    try:
        coll = get_collection(client)
        with filter_stderr_lines():
            res = coll.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
                where=where,
            )

        # Standardize result shape
        out = []
        if res and res.get("documents"):
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res.get("distances", [[None] * len(docs)])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                out.append({"text": doc, "meta": meta, "distance": dist})

        logger.debug(f"Successfully queried embeddings, returned {len(out)} results")
        return out

    except Exception as e:
        logger.error(f"Failed to query embeddings: {e}")
        # Classify error for retry decisions
        error_type = ErrorClassifier.classify_error(e)
        if error_type == ErrorType.PERMANENT:
            logger.error(f"Permanent error detected, not retrying: {e}")
        raise ConnectionError(f"Database query failed: {e}") from e
    finally:
        return_client(client)


@retry_with_backoff(_db_retry_policy, _db_circuit_breaker)
def reset_collection():
    """Reset collection with retry and circuit breaker protection."""
    client = get_client()
    try:
        # Try to delete existing collection
        try:
            with filter_stderr_lines():
                client.delete_collection(CHROMA_COLLECTION)
            logger.info(f"Deleted existing collection '{CHROMA_COLLECTION}'")
        except ValueError:
            # Collection doesn't exist, which is fine
            logger.info(f"Collection '{CHROMA_COLLECTION}' does not exist, creating new one")

        # Recreate with cosine space
        with filter_stderr_lines():
            collection = client.get_or_create_collection(
                name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
            )
        logger.info(f"Successfully reset collection '{CHROMA_COLLECTION}'")
        return collection

    except Exception as e:
        logger.error(f"Failed to reset collection: {e}")
        raise ConnectionError(f"Database reset failed: {e}") from e
    finally:
        return_client(client)


def get_database_status() -> Dict:
    """Get comprehensive database status including connection pool and circuit breaker state."""
    try:
        # Get circuit breaker status
        cb_status = _db_circuit_breaker.get_status()

        # Get connection pool status
        with _connection_pool._lock:
            pool_status = {
                "total_connections": len(_connection_pool._connections),
                "in_use_connections": len(_connection_pool._in_use),
                "available_connections": len(_connection_pool._connections) - len(_connection_pool._in_use),
                "max_connections": _connection_pool.max_connections,
                "last_cleanup": _connection_pool._last_cleanup
            }

        # Test database connectivity
        try:
            client = get_client()
            with filter_stderr_lines():
                collections = client.list_collections()
            return_client(client)
            db_healthy = True
            collection_count = len(collections)
        except Exception as e:
            db_healthy = False
            collection_count = None
            logger.warning(f"Database health check failed: {e}")

        return {
            "healthy": db_healthy,
            "collection_count": collection_count,
            "target_collection": CHROMA_COLLECTION,
            "database_path": CHROMA_DB_PATH,
            "connection_pool": pool_status,
            "circuit_breaker": cb_status,
            "retry_policy": {
                "max_retries": _db_retry_policy.max_retries,
                "base_delay": _db_retry_policy.base_delay,
                "max_delay": _db_retry_policy.max_delay
            }
        }

    except Exception as e:
        logger.error(f"Failed to get database status: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "connection_pool": {"error": "Unable to access pool status"},
            "circuit_breaker": {"error": "Unable to access circuit breaker status"}
        }


def cleanup_database_connections():
    """Cleanup database connections and resources."""
    try:
        _connection_pool.close_all()
        logger.info("Database connections cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during database cleanup: {e}")


# Register cleanup function for graceful shutdown
import atexit
atexit.register(cleanup_database_connections)
