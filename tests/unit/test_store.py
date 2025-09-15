"""Unit tests for the data store module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import uuid

from support_deflect_bot.data.store import (
    get_client,
    get_collection,
    upsert_chunks,
    query_by_embedding,
    reset_collection
)


class TestDataStore:
    """Test suite for data store module functions."""

    @patch('support_deflect_bot.data.store.chromadb.PersistentClient')
    @patch('support_deflect_bot.data.store.os.makedirs')
    def test_get_client_success(self, mock_makedirs, mock_client_class):
        """Test successful ChromaDB client creation."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        result = get_client()

        assert result == mock_client
        mock_makedirs.assert_called_once()
        mock_client_class.assert_called_once()

    @patch('support_deflect_bot.data.store.chromadb.PersistentClient')
    @patch('support_deflect_bot.data.store.os.makedirs')
    def test_get_client_failure(self, mock_makedirs, mock_client_class):
        """Test ChromaDB client creation failure."""
        mock_client_class.side_effect = Exception("Database connection failed")

        with pytest.raises(ConnectionError, match="Database connection failed"):
            get_client()

    @patch('support_deflect_bot.data.store.get_client')
    def test_get_collection_success(self, mock_get_client):
        """Test successful collection retrieval."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        result = get_collection()

        assert result == mock_collection
        mock_client.get_or_create_collection.assert_called_once()

    @patch('support_deflect_bot.data.store.get_client')
    def test_get_collection_with_client(self, mock_get_client):
        """Test collection retrieval with provided client."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection

        result = get_collection(client=mock_client)

        assert result == mock_collection
        # get_client should not be called when client is provided
        mock_get_client.assert_not_called()

    @patch('support_deflect_bot.data.store.get_client')
    def test_get_collection_failure(self, mock_get_client):
        """Test collection retrieval failure."""
        mock_client = Mock()
        mock_client.get_or_create_collection.side_effect = Exception("Collection error")
        mock_get_client.return_value = mock_client

        with pytest.raises(ConnectionError, match="Database collection access failed"):
            get_collection()

    @patch('support_deflect_bot.data.store.get_collection')
    @patch('support_deflect_bot.data.store.uuid.uuid4')
    def test_upsert_chunks_without_ids(self, mock_uuid, mock_get_collection):
        """Test upserting chunks without providing IDs."""
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_uuid.side_effect = [uuid.UUID('12345678-1234-5678-1234-567812345678')]

        chunks = ["chunk1", "chunk2"]
        metadatas = [{"key": "value1"}, {"key": "value2"}]

        upsert_chunks(chunks, metadatas)

        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        assert call_args["documents"] == chunks
        assert call_args["metadatas"] == metadatas
        assert len(call_args["ids"]) == 2

    @patch('support_deflect_bot.data.store.get_collection')
    def test_upsert_chunks_with_ids(self, mock_get_collection):
        """Test upserting chunks with provided IDs."""
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection

        chunks = ["chunk1", "chunk2"]
        metadatas = [{"key": "value1"}, {"key": "value2"}]
        ids = ["id1", "id2"]

        upsert_chunks(chunks, metadatas, ids)

        mock_collection.add.assert_called_once_with(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

    @patch('support_deflect_bot.data.store.get_collection')
    def test_upsert_chunks_failure(self, mock_get_collection):
        """Test upsert chunks failure."""
        mock_collection = Mock()
        mock_collection.add.side_effect = Exception("Database error")
        mock_get_collection.return_value = mock_collection

        chunks = ["chunk1"]
        metadatas = [{"key": "value1"}]

        with pytest.raises(ConnectionError, match="Failed to store chunks in database"):
            upsert_chunks(chunks, metadatas)

    @patch('support_deflect_bot.data.store.get_collection')
    def test_query_by_embedding_success(self, mock_get_collection):
        """Test successful embedding query."""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"path": "file1.txt"}, {"path": "file2.txt"}]],
            "distances": [[0.2, 0.4]]
        }
        mock_get_collection.return_value = mock_collection

        query_embedding = [0.1, 0.2, 0.3]
        results = query_by_embedding(query_embedding, k=2)

        assert len(results) == 2
        assert results[0]["text"] == "doc1"
        assert results[0]["meta"]["path"] == "file1.txt"
        assert results[0]["distance"] == 0.2
        assert results[1]["text"] == "doc2"
        assert results[1]["meta"]["path"] == "file2.txt"
        assert results[1]["distance"] == 0.4

        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=2,
            include=["documents", "metadatas", "distances"],
            where=None
        )

    @patch('support_deflect_bot.data.store.get_collection')
    def test_query_by_embedding_with_filter(self, mock_get_collection):
        """Test embedding query with where filter."""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"host": "example.com"}]],
            "distances": [[0.3]]
        }
        mock_get_collection.return_value = mock_collection

        query_embedding = [0.1, 0.2, 0.3]
        where_filter = {"host": {"$in": ["example.com"]}}
        results = query_by_embedding(query_embedding, k=5, where=where_filter)

        assert len(results) == 1
        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"],
            where=where_filter
        )

    @patch('support_deflect_bot.data.store.get_collection')
    def test_query_by_embedding_empty_results(self, mock_get_collection):
        """Test embedding query with empty results."""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        mock_get_collection.return_value = mock_collection

        query_embedding = [0.1, 0.2, 0.3]
        results = query_by_embedding(query_embedding)

        assert results == []

    @patch('support_deflect_bot.data.store.get_collection')
    def test_query_by_embedding_no_distances(self, mock_get_collection):
        """Test embedding query without distances in results."""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"path": "file1.txt"}]]
            # No distances key
        }
        mock_get_collection.return_value = mock_collection

        query_embedding = [0.1, 0.2, 0.3]
        results = query_by_embedding(query_embedding)

        assert len(results) == 1
        assert results[0]["text"] == "doc1"
        assert results[0]["distance"] is None

    @patch('support_deflect_bot.data.store.get_collection')
    def test_query_by_embedding_failure(self, mock_get_collection):
        """Test embedding query failure."""
        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("Query failed")
        mock_get_collection.return_value = mock_collection

        query_embedding = [0.1, 0.2, 0.3]

        with pytest.raises(ConnectionError, match="Database query failed"):
            query_by_embedding(query_embedding)

    @patch('support_deflect_bot.data.store.get_client')
    def test_reset_collection_success(self, mock_get_client):
        """Test successful collection reset."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        result = reset_collection()

        assert result == mock_collection
        mock_client.delete_collection.assert_called_once()
        mock_client.get_or_create_collection.assert_called_once()

    @patch('support_deflect_bot.data.store.get_client')
    def test_reset_collection_nonexistent(self, mock_get_client):
        """Test collection reset when collection doesn't exist."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.delete_collection.side_effect = ValueError("Collection not found")
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        result = reset_collection()

        # Should not raise error and should still create new collection
        assert result == mock_collection
        mock_client.get_or_create_collection.assert_called_once()

    def test_query_by_embedding_result_structure(self):
        """Test the structure of query results."""
        with patch('support_deflect_bot.data.store.get_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_collection.query.return_value = {
                "documents": [["test document"]],
                "metadatas": [[{"path": "test.txt", "chunk_id": "1"}]],
                "distances": [[0.5]]
            }
            mock_get_collection.return_value = mock_collection

            results = query_by_embedding([0.1, 0.2, 0.3])

            assert len(results) == 1
            result = results[0]
            assert "text" in result
            assert "meta" in result
            assert "distance" in result
            assert isinstance(result["meta"], dict)

    @patch('support_deflect_bot.data.store.get_collection')
    def test_upsert_chunks_empty_lists(self, mock_get_collection):
        """Test upserting empty lists."""
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection

        upsert_chunks([], [])

        # Should still call add with empty lists
        mock_collection.add.assert_called_once_with(
            documents=[],
            metadatas=[],
            ids=[]
        )

    @patch('support_deflect_bot.data.store.get_collection')
    def test_query_by_embedding_malformed_response(self, mock_get_collection):
        """Test handling of malformed query response."""
        mock_collection = Mock()
        mock_collection.query.return_value = {}  # Empty response
        mock_get_collection.return_value = mock_collection

        results = query_by_embedding([0.1, 0.2, 0.3])

        assert results == []

    @patch('support_deflect_bot.data.store.get_collection')
    def test_query_by_embedding_none_response(self, mock_get_collection):
        """Test handling of None query response."""
        mock_collection = Mock()
        mock_collection.query.return_value = None
        mock_get_collection.return_value = mock_collection

        results = query_by_embedding([0.1, 0.2, 0.3])

        assert results == []

    @patch('support_deflect_bot.data.store.CHROMA_COLLECTION', 'test_collection')
    @patch('support_deflect_bot.data.store.get_client')
    def test_collection_name_configuration(self, mock_get_client):
        """Test that collection uses configured name."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        get_collection()

        mock_client.get_or_create_collection.assert_called_once_with(
            name='test_collection',
            metadata={"hnsw:space": "cosine"}
        )

    @patch('support_deflect_bot.data.store.get_collection')
    def test_query_by_embedding_mismatched_arrays(self, mock_get_collection):
        """Test query handling when response arrays have different lengths."""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"path": "file1.txt"}]],  # Only one metadata
            "distances": [[0.2, 0.4]]
        }
        mock_get_collection.return_value = mock_collection

        results = query_by_embedding([0.1, 0.2, 0.3])

        # Should handle gracefully, likely by zipping until shortest array
        assert len(results) == 1  # Limited by shortest array (metadatas)

    def test_upsert_chunks_mismatched_lengths(self):
        """Test upsert with mismatched chunk and metadata lengths."""
        with patch('support_deflect_bot.data.store.get_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection

            chunks = ["chunk1", "chunk2"]
            metadatas = [{"key": "value1"}]  # Only one metadata for two chunks

            # This should still work - ChromaDB will handle the mismatch
            upsert_chunks(chunks, metadatas)

            mock_collection.add.assert_called_once()

    @patch('support_deflect_bot.data.store.filter_stderr_lines')
    @patch('support_deflect_bot.data.store.chromadb.PersistentClient')
    @patch('support_deflect_bot.data.store.os.makedirs')
    def test_stderr_filtering_applied(self, mock_makedirs, mock_client_class, mock_filter):
        """Test that stderr filtering is applied during operations."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_filter.return_value.__enter__ = Mock()
        mock_filter.return_value.__exit__ = Mock()

        get_client()

        # Verify stderr filtering was used
        mock_filter.assert_called()