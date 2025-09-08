import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.data.store import (
    get_client, get_collection, upsert_chunks, 
    query_by_embedding, reset_collection
)


class TestGetClient:
    """Test ChromaDB client creation."""

    @patch('src.data.store.chromadb.PersistentClient')
    @patch('src.data.store.os.makedirs')
    def test_get_client_success(self, mock_makedirs, mock_client_class):
        """Test successful client creation."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        with patch('src.data.store.CHROMA_DB_PATH', './test_db'):
            result = get_client()
        
        mock_makedirs.assert_called_once_with('./test_db', exist_ok=True)
        mock_client_class.assert_called_once_with(path='./test_db')
        assert result == mock_client

    @patch('src.data.store.chromadb.PersistentClient')
    @patch('src.data.store.os.makedirs')
    @patch('src.data.store.logging.error')
    def test_get_client_failure(self, mock_log, mock_makedirs, mock_client_class):
        """Test client creation failure."""
        mock_client_class.side_effect = Exception("DB Connection Error")
        
        with pytest.raises(ConnectionError, match="Database connection failed"):
            get_client()
        
        mock_log.assert_called_once()


class TestGetCollection:
    """Test collection retrieval/creation."""

    @patch('src.data.store.get_client')
    def test_get_collection_success(self, mock_get_client):
        """Test successful collection retrieval."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        
        result = get_collection()
        
        assert result == mock_collection
        mock_client.get_or_create_collection.assert_called_once()

    @patch('src.data.store.get_client')
    def test_get_collection_with_client(self, mock_get_client):
        """Test collection retrieval with provided client."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        
        result = get_collection(client=mock_client)
        
        assert result == mock_collection
        mock_get_client.assert_not_called()  # Should not create new client

    @patch('src.data.store.get_client')
    @patch('src.data.store.logging.error')
    def test_get_collection_failure(self, mock_log, mock_get_client):
        """Test collection retrieval failure."""
        mock_client = Mock()
        mock_client.get_or_create_collection.side_effect = Exception("Collection Error")
        mock_get_client.return_value = mock_client
        
        with pytest.raises(ConnectionError, match="Database collection access failed"):
            get_collection()
        
        mock_log.assert_called_once()


class TestUpsertChunks:
    """Test chunk upserting functionality."""

    @patch('src.data.store.get_collection')
    def test_upsert_chunks_with_ids(self, mock_get_collection):
        """Test upserting chunks with provided IDs."""
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        
        chunks = ["chunk1", "chunk2"]
        metadatas = [{"path": "doc1.md"}, {"path": "doc2.md"}]
        ids = ["id1", "id2"]
        
        upsert_chunks(chunks, metadatas, ids)
        
        mock_collection.add.assert_called_once_with(
            documents=chunks, metadatas=metadatas, ids=ids
        )

    @patch('src.data.store.get_collection')
    @patch('src.data.store.uuid.uuid4')
    def test_upsert_chunks_generate_ids(self, mock_uuid, mock_get_collection):
        """Test upserting chunks with generated IDs."""
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_uuid.side_effect = [Mock(spec=str), Mock(spec=str)]
        
        chunks = ["chunk1", "chunk2"]
        metadatas = [{"path": "doc1.md"}, {"path": "doc2.md"}]
        
        upsert_chunks(chunks, metadatas, ids=None)
        
        mock_collection.add.assert_called_once()
        assert mock_uuid.call_count == 2

    @patch('src.data.store.get_collection')
    @patch('src.data.store.logging.error')
    def test_upsert_chunks_failure(self, mock_log, mock_get_collection):
        """Test upserting failure handling."""
        mock_collection = Mock()
        mock_collection.add.side_effect = Exception("Upsert Error")
        mock_get_collection.return_value = mock_collection
        
        with pytest.raises(ConnectionError, match="Failed to store chunks in database"):
            upsert_chunks(["chunk"], [{"path": "doc.md"}])
        
        mock_log.assert_called_once()


class TestQueryByEmbedding:
    """Test embedding-based querying."""

    @patch('src.data.store.get_collection')
    def test_query_by_embedding_success(self, mock_get_collection):
        """Test successful embedding query."""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"path": "test1.md"}, {"path": "test2.md"}]],
            "distances": [[0.1, 0.3]]
        }
        mock_get_collection.return_value = mock_collection
        
        query_embedding = [0.1] * 768
        results = query_by_embedding(query_embedding, k=2)
        
        assert len(results) == 2
        assert results[0]["text"] == "doc1"
        assert results[0]["meta"] == {"path": "test1.md"}
        assert results[0]["distance"] == 0.1
        
        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=2,
            include=["documents", "metadatas", "distances"],
            where=None
        )

    @patch('src.data.store.get_collection')
    def test_query_by_embedding_with_where(self, mock_get_collection):
        """Test embedding query with where clause."""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"path": "test1.md"}]],
            "distances": [[0.1]]
        }
        mock_get_collection.return_value = mock_collection
        
        query_embedding = [0.1] * 768
        where_clause = {"host": {"$in": ["example.com"]}}
        
        results = query_by_embedding(query_embedding, k=1, where=where_clause)
        
        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
            where=where_clause
        )

    @patch('src.data.store.get_collection')
    def test_query_by_embedding_empty_result(self, mock_get_collection):
        """Test embedding query with empty results."""
        mock_collection = Mock()
        mock_collection.query.return_value = {"documents": None}
        mock_get_collection.return_value = mock_collection
        
        results = query_by_embedding([0.1] * 768)
        assert results == []

    @patch('src.data.store.get_collection')
    def test_query_by_embedding_no_distances(self, mock_get_collection):
        """Test embedding query without distances."""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"path": "test.md"}]]
            # No distances key
        }
        mock_get_collection.return_value = mock_collection
        
        results = query_by_embedding([0.1] * 768)
        
        assert len(results) == 1
        assert results[0]["distance"] is None

    @patch('src.data.store.get_collection')
    @patch('src.data.store.logging.error')
    def test_query_by_embedding_failure(self, mock_log, mock_get_collection):
        """Test embedding query failure."""
        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("Query Error")
        mock_get_collection.return_value = mock_collection
        
        with pytest.raises(ConnectionError, match="Database query failed"):
            query_by_embedding([0.1] * 768)
        
        mock_log.assert_called_once()


class TestResetCollection:
    """Test collection reset functionality."""

    @patch('src.data.store.get_client')
    def test_reset_collection_success(self, mock_get_client):
        """Test successful collection reset."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.delete_collection.return_value = None
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        
        with patch('src.data.store.CHROMA_COLLECTION', 'test_collection'):
            result = reset_collection()
        
        mock_client.delete_collection.assert_called_once_with('test_collection')
        mock_client.get_or_create_collection.assert_called_once()
        assert result == mock_collection

    @patch('src.data.store.get_client')
    def test_reset_collection_not_exists(self, mock_get_client):
        """Test resetting non-existent collection."""
        mock_client = Mock()
        mock_collection = Mock()
        # Collection doesn't exist - delete raises ValueError
        mock_client.delete_collection.side_effect = ValueError("Collection not found")
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        
        result = reset_collection()
        
        # Should handle ValueError gracefully
        assert result == mock_collection
        mock_client.get_or_create_collection.assert_called_once()