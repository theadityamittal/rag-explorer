import pytest

from src.data.chunker import build_docs_from_files, chunk_text


class TestChunkText:
    """Test the chunk_text function."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "This is a test document. " * 100  # Create a long text
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) <= 120 for chunk in chunks)  # Allow for overlap

    def test_chunk_text_short_text(self):
        """Test chunking of text shorter than chunk size."""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_empty_string(self):
        """Test chunking of empty string."""
        chunks = chunk_text("", chunk_size=100, overlap=20)
        assert chunks == []

    def test_chunk_text_whitespace_only(self):
        """Test chunking of whitespace-only string."""
        chunks = chunk_text("   \n\t   ", chunk_size=100, overlap=20)
        assert chunks == []

    def test_chunk_text_overlap(self):
        """Test that chunks have proper overlap."""
        text = "A" * 200
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        assert len(chunks) >= 2
        # Check that consecutive chunks have some overlap
        if len(chunks) > 1:
            # Last 20 chars of first chunk should match first 20 chars of second
            first_end = chunks[0][-20:]
            second_start = chunks[1][:20]
            assert first_end == second_start

    def test_chunk_text_custom_params(self):
        """Test chunking with custom chunk size and overlap."""
        text = "Word " * 100
        chunks = chunk_text(text, chunk_size=50, overlap=10)

        assert len(chunks) > 1
        assert all(len(chunk) <= 60 for chunk in chunks)  # Allow for overlap

    def test_chunk_text_no_overlap(self):
        """Test chunking with no overlap."""
        text = "A" * 200
        chunks = chunk_text(text, chunk_size=100, overlap=0)

        assert len(chunks) == 2
        assert chunks[0] == "A" * 100
        assert chunks[1] == "A" * 100


class TestBuildDocsFromFiles:
    """Test the build_docs_from_files function."""

    def test_build_docs_identity(self):
        """Test that build_docs_from_files returns input unchanged."""
        files = {"doc1.md": "Content 1", "doc2.txt": "Content 2"}
        result = build_docs_from_files(files)
        assert result == files

    def test_build_docs_empty(self):
        """Test with empty input."""
        result = build_docs_from_files({})
        assert result == {}

    def test_build_docs_preserves_structure(self):
        """Test that complex file structure is preserved."""
        files = {
            "/path/to/file1.md": "# Header\nContent here",
            "relative/path/file2.txt": "Simple text content",
            "file3.md": "",
        }
        result = build_docs_from_files(files)
        assert result == files
        assert len(result) == 3
