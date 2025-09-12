"""
Unit tests for UnifiedDocumentProcessor.

Tests the document processing service that handles local file processing,
web content crawling, text chunking, and storage operations.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, mock_open
from tests.base import BaseEngineTest
from support_deflect_bot.engine import UnifiedDocumentProcessor


class TestUnifiedDocumentProcessor(BaseEngineTest):
    """Test the UnifiedDocumentProcessor service."""
    
    @pytest.fixture
    def doc_processor(self):
        """Create UnifiedDocumentProcessor instance for testing."""
        mock_registry = {}
        return UnifiedDocumentProcessor(provider_registry=mock_registry)
    
    @pytest.mark.unit
    @pytest.mark.engine
    def test_init_creates_correct_attributes(self, doc_processor):
        """Test document processor initialization creates correct attributes."""
        # Assert
        assert hasattr(doc_processor, 'provider_registry')
        assert hasattr(doc_processor, 'collection_stats')
        assert hasattr(doc_processor, 'processing_stats')
        
        # Check stats structure
        expected_stats = ["documents_processed", "chunks_created", "total_size",
                         "processing_time", "last_processed"]
        for stat in expected_stats:
            assert stat in doc_processor.processing_stats
            
    @pytest.mark.unit
    @pytest.mark.engine
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch.object(UnifiedDocumentProcessor, '_read_documents_from_directory')
    @patch.object(UnifiedDocumentProcessor, '_chunk_text')
    @patch.object(UnifiedDocumentProcessor, '_store_chunks')
    async def test_process_local_directory_processes_files(
        self, mock_store, mock_chunk, mock_read, mock_isdir, mock_exists, doc_processor
    ):
        """Test local directory processing reads and processes files."""
        # Arrange
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_read.return_value = {
            "test1.md": "# Test Document 1\nContent for test document 1",
            "test2.md": "# Test Document 2\nContent for test document 2"
        }
        mock_chunk.return_value = ["Chunk 1", "Chunk 2"]
        mock_store.return_value = 4  # 2 docs * 2 chunks each
        
        # Act
        result = await doc_processor.process_local_directory("/test/docs")
        
        # Assert
        assert result["status"] == "success"
        assert result["documents_processed"] == 2
        assert result["chunks_created"] == 4
        mock_read.assert_called_once_with("/test/docs", [".md", ".txt", ".rst"])
        assert mock_chunk.call_count == 2  # Called for each document
        mock_store.assert_called()
        
    @pytest.mark.unit
    @pytest.mark.engine
    @patch('os.path.exists')
    def test_process_local_directory_handles_missing_directory(
        self, mock_exists, doc_processor
    ):
        """Test handling of non-existent directory."""
        # Arrange
        mock_exists.return_value = False
        
        # Act
        result = asyncio.run(doc_processor.process_local_directory("/nonexistent"))
        
        # Assert
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()
        
    @pytest.mark.unit
    @pytest.mark.engine
    @patch.object(UnifiedDocumentProcessor, '_fetch_html')
    @patch.object(UnifiedDocumentProcessor, '_html_to_text')
    @patch.object(UnifiedDocumentProcessor, '_index_single_url')
    async def test_process_web_content_crawls_urls(
        self, mock_index, mock_html_to_text, mock_fetch, doc_processor
    ):
        """Test web content processing crawls and indexes URLs."""
        # Arrange
        mock_fetch.return_value = {
            "status_code": 200,
            "html": "<html><head><title>Test Page</title></head><body>Test content</body></html>",
            "etag": "test-etag"
        }
        mock_html_to_text.return_value = ("Test Page", "Test content")
        mock_index.return_value = 2  # 2 chunks created
        
        urls = ["https://example.com/page1", "https://example.com/page2"]
        
        # Act
        result = await doc_processor.process_web_content(urls, max_depth=1)
        
        # Assert
        assert result["status"] == "success"
        assert result["urls_processed"] >= 2
        assert result["chunks_created"] >= 4  # 2 urls * 2 chunks each
        assert mock_fetch.call_count == 2
        assert mock_index.call_count == 2
        
    @pytest.mark.unit
    @pytest.mark.engine
    async def test_process_batch_urls_handles_failures_gracefully(self, doc_processor):
        """Test batch URL processing handles individual URL failures."""
        # Arrange
        urls = [
            "https://valid.com/page1",
            "https://invalid.com/page2",  # Will fail
            "https://valid.com/page3"
        ]
        
        with patch.object(doc_processor, '_fetch_html') as mock_fetch:
            # First and third calls succeed, second fails
            mock_fetch.side_effect = [
                {"status_code": 200, "html": "<html>Content 1</html>"},
                Exception("Network error"),
                {"status_code": 200, "html": "<html>Content 3</html>"}
            ]
            
            with patch.object(doc_processor, '_html_to_text', return_value=("Title", "Content")):
                with patch.object(doc_processor, '_index_single_url', return_value=1):
                    # Act
                    result = await doc_processor.process_batch_urls(urls)
                    
                    # Assert
                    assert result["status"] == "partial_success"
                    assert result["urls_processed"] == 2  # Only 2 succeeded
                    assert result["failed_urls"] == 1
                    assert len(result["errors"]) == 1
                    
    @pytest.mark.unit
    @pytest.mark.engine
    def test_get_collection_stats_returns_metrics(self, doc_processor):
        """Test collection stats returns comprehensive metrics."""
        # Arrange
        doc_processor.collection_stats.update({
            "total_documents": 100,
            "total_chunks": 500,
            "collection_size": "10MB"
        })
        
        # Act
        stats = doc_processor.get_collection_stats()
        
        # Assert
        expected_keys = ["total_documents", "total_chunks", "collection_size",
                        "last_updated", "health_status"]
        for key in expected_keys:
            assert key in stats
            
    @pytest.mark.unit
    @pytest.mark.engine
    def test_validate_sources_checks_accessibility(self, doc_processor):
        """Test source validation checks if sources are accessible."""
        # Arrange
        sources = [
            "/valid/path/doc.md",
            "https://valid-url.com/doc",
            "/invalid/path/doc.md",
            "https://invalid-url.com/doc"
        ]
        
        with patch('os.path.exists') as mock_exists:
            with patch('urllib.request.urlopen') as mock_urlopen:
                # First path exists, second doesn't
                mock_exists.side_effect = [True, False]
                # First URL works, second fails
                mock_urlopen.side_effect = [Mock(), Exception("URL error")]
                
                # Act
                validation = doc_processor.validate_sources(sources)
                
                # Assert
                assert len(validation) == 4
                assert validation["/valid/path/doc.md"] is True
                assert validation["/invalid/path/doc.md"] is False
                assert validation["https://valid-url.com/doc"] is True
                assert validation["https://invalid-url.com/doc"] is False
                
    @pytest.mark.unit
    @pytest.mark.engine
    @patch('os.listdir')
    @patch('os.path.join')
    @patch('builtins.open', new_callable=mock_open)
    def test_read_documents_from_directory_reads_files(
        self, mock_file, mock_join, mock_listdir, doc_processor
    ):
        """Test reading documents from directory filters by extension."""
        # Arrange
        mock_listdir.return_value = ["doc1.md", "doc2.txt", "doc3.pdf", "README.md"]
        mock_join.side_effect = lambda dir, file: f"{dir}/{file}"
        mock_file.return_value.read.side_effect = [
            "Content of doc1.md",
            "Content of doc2.txt", 
            "Content of README.md"  # PDF will be skipped
        ]
        
        # Act
        documents = doc_processor._read_documents_from_directory(
            "/test/dir", [".md", ".txt"]
        )
        
        # Assert
        assert len(documents) == 3  # PDF excluded
        assert "doc1.md" in documents
        assert "doc2.txt" in documents  
        assert "README.md" in documents
        assert "doc3.pdf" not in documents
        assert documents["doc1.md"] == "Content of doc1.md"
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_chunk_text_creates_overlapping_chunks(self, doc_processor):
        """Test text chunking creates overlapping chunks."""
        # Arrange
        long_text = "This is a test document. " * 50  # ~1250 characters
        chunk_size = 500
        overlap = 100
        
        # Act
        chunks = doc_processor._chunk_text(long_text, chunk_size, overlap)
        
        # Assert
        assert len(chunks) > 1  # Should create multiple chunks
        assert all(len(chunk) <= chunk_size + overlap for chunk in chunks)
        
        # Check overlap - first chunk's end should overlap with second chunk's start
        if len(chunks) > 1:
            # There should be some common content between adjacent chunks
            first_end = chunks[0][-overlap:]
            second_start = chunks[1][:overlap]
            # Not exact overlap due to word boundaries, but should share some words
            first_words = set(first_end.split())
            second_words = set(second_start.split())
            assert len(first_words & second_words) > 0
            
    @pytest.mark.unit
    @pytest.mark.engine
    def test_chunk_text_handles_short_text(self, doc_processor):
        """Test text chunking handles text shorter than chunk size."""
        # Arrange
        short_text = "This is a short text."
        chunk_size = 1000
        
        # Act
        chunks = doc_processor._chunk_text(short_text, chunk_size)
        
        # Assert
        assert len(chunks) == 1
        assert chunks[0] == short_text
        
    @pytest.mark.unit
    @pytest.mark.engine
    @patch('chromadb.Client')
    def test_store_chunks_adds_to_collection(self, mock_chroma, doc_processor):
        """Test chunk storage adds chunks to ChromaDB collection."""
        # Arrange
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        
        chunk_texts = ["Chunk 1 content", "Chunk 2 content"]
        metadatas = [
            {"source": "doc1.md", "chunk_id": "chunk_1"},
            {"source": "doc1.md", "chunk_id": "chunk_2"}
        ]
        
        # Act
        count = doc_processor._store_chunks(chunk_texts, metadatas)
        
        # Assert
        assert count == 2
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        assert "documents" in call_args
        assert "metadatas" in call_args
        assert "ids" in call_args
        assert len(call_args["documents"]) == 2
        
    @pytest.mark.unit
    @pytest.mark.engine
    @patch.object(UnifiedDocumentProcessor, '_chunk_text')
    @patch.object(UnifiedDocumentProcessor, '_store_chunks')
    def test_index_single_url_processes_content(
        self, mock_store, mock_chunk, doc_processor
    ):
        """Test single URL indexing processes and stores content."""
        # Arrange
        url = "https://example.com/doc"
        html = "<html><body>Test content</body></html>"
        title = "Test Document"
        text = "Test content for indexing"
        
        mock_chunk.return_value = ["Chunk 1", "Chunk 2"]
        mock_store.return_value = 2
        
        # Act
        chunk_count = doc_processor._index_single_url(url, html, title, text)
        
        # Assert
        assert chunk_count == 2
        mock_chunk.assert_called_once_with(text, 900, 150)  # Default chunk params
        mock_store.assert_called_once()
        
    @pytest.mark.unit
    @pytest.mark.engine
    @patch('urllib.request.urlopen')
    @patch('urllib.request.Request')
    def test_fetch_html_makes_http_request(self, mock_request, mock_urlopen, doc_processor):
        """Test HTML fetching makes proper HTTP request."""
        # Arrange
        mock_response = Mock()
        mock_response.read.return_value = b"<html>Test content</html>"
        mock_response.getcode.return_value = 200
        mock_response.info.return_value = {"etag": "test-etag"}
        mock_urlopen.return_value = mock_response
        
        # Act
        result = doc_processor._fetch_html("https://example.com/test")
        
        # Assert
        assert result["status_code"] == 200
        assert result["html"] == "<html>Test content</html>"
        assert result["etag"] == "test-etag"
        mock_request.assert_called_once()
        mock_urlopen.assert_called_once()
        
    @pytest.mark.unit
    @pytest.mark.engine
    @patch('urllib.request.urlopen')
    def test_fetch_html_handles_http_errors(self, mock_urlopen, doc_processor):
        """Test HTML fetching handles HTTP errors gracefully."""
        # Arrange
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            url="https://example.com/test",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None
        )
        
        # Act & Assert
        with pytest.raises(HTTPError):
            doc_processor._fetch_html("https://example.com/test")
            
    @pytest.mark.unit
    @pytest.mark.engine
    def test_html_to_text_extracts_content(self, doc_processor):
        """Test HTML to text conversion extracts title and content."""
        # Arrange
        html = """
        <html>
        <head><title>Test Page Title</title></head>
        <body>
            <h1>Main Heading</h1>
            <p>This is test content.</p>
            <script>alert('ignore me');</script>
            <style>body { color: red; }</style>
            <p>More content here.</p>
        </body>
        </html>
        """
        
        # Act
        title, text = doc_processor._html_to_text(html)
        
        # Assert
        assert title == "Test Page Title"
        assert "Main Heading" in text
        assert "This is test content" in text
        assert "More content here" in text
        # Script and style content should be excluded
        assert "alert('ignore me')" not in text
        assert "color: red" not in text
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_extract_links_finds_urls(self, doc_processor):
        """Test link extraction finds URLs in HTML."""
        # Arrange
        html = """
        <html>
        <body>
            <a href="/relative-link">Relative</a>
            <a href="https://external.com/page">External</a>
            <a href="mailto:test@example.com">Email</a>
            <a href="#fragment">Fragment</a>
            <a href="javascript:void(0)">JavaScript</a>
        </body>
        </html>
        """
        base_url = "https://example.com"
        
        # Act
        links = doc_processor._extract_links(html, base_url)
        
        # Assert
        assert isinstance(links, set)
        # Should convert relative to absolute
        assert "https://example.com/relative-link" in links
        # Should include external links
        assert "https://external.com/page" in links
        # Should exclude non-HTTP links
        assert "mailto:test@example.com" not in links
        assert "#fragment" not in links
        assert "javascript:void(0)" not in links


class TestUnifiedDocumentProcessorIntegration(BaseEngineTest):
    """Integration tests for document processor with mocked dependencies."""
    
    @pytest.mark.unit
    @pytest.mark.engine
    @patch('chromadb.Client')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open)
    async def test_full_document_processing_pipeline(
        self, mock_file, mock_listdir, mock_isdir, mock_exists, mock_chroma
    ):
        """Test complete document processing pipeline."""
        # Arrange
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_listdir.return_value = ["test1.md", "test2.md"]
        mock_file.return_value.read.side_effect = [
            "# Document 1\nContent for document 1",
            "# Document 2\nContent for document 2"
        ]
        
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        
        doc_processor = UnifiedDocumentProcessor()
        
        # Act
        result = await doc_processor.process_local_directory("/test/docs")
        
        # Assert
        assert result["status"] == "success"
        assert result["documents_processed"] == 2
        assert result["chunks_created"] > 0
        mock_collection.add.assert_called()
        
    @pytest.mark.unit
    @pytest.mark.engine
    def test_error_handling_with_storage_failure(self, doc_processor):
        """Test document processor handles storage failures gracefully."""
        # Arrange
        chunk_texts = ["Test chunk"]
        metadatas = [{"source": "test.md"}]
        
        with patch('chromadb.Client') as mock_chroma:
            mock_collection = Mock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
            mock_collection.add.side_effect = Exception("Storage failed")
            
            # Act & Assert
            with pytest.raises(Exception, match="Storage failed"):
                doc_processor._store_chunks(chunk_texts, metadatas)