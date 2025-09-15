"""Unit tests for the text chunking module."""

import pytest
from support_deflect_bot.data.chunker import chunk_text, build_docs_from_files


class TestChunker:
    """Test suite for text chunking functions."""

    def test_chunk_text_basic(self):
        """Test basic text chunking functionality."""
        text = "A" * 100
        chunks = chunk_text(text, chunk_size=50, overlap=10)

        assert len(chunks) == 3
        assert len(chunks[0]) == 50
        assert len(chunks[1]) == 50
        assert len(chunks[2]) == 10  # Remaining characters

    def test_chunk_text_with_overlap(self):
        """Test text chunking with overlap."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunk_text(text, chunk_size=10, overlap=3)

        assert len(chunks) >= 2
        # Check that overlap is preserved
        if len(chunks) > 1:
            # Last 3 chars of first chunk should overlap with first 3 of second
            assert chunks[0][-3:] == chunks[1][:3]

    def test_chunk_text_short_text(self):
        """Test chunking text shorter than chunk size."""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=100, overlap=10)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_empty_text(self):
        """Test chunking empty text."""
        chunks = chunk_text("", chunk_size=100, overlap=10)
        assert chunks == []

    def test_chunk_text_whitespace_only(self):
        """Test chunking whitespace-only text."""
        chunks = chunk_text("   \n\t   ", chunk_size=100, overlap=10)
        assert chunks == []

    def test_chunk_text_strips_whitespace(self):
        """Test that chunks are stripped of leading/trailing whitespace."""
        text = "  chunk1  " + " " * 90 + "  chunk2  "
        chunks = chunk_text(text, chunk_size=50, overlap=10)

        for chunk in chunks:
            assert not chunk.startswith(" ")
            assert not chunk.endswith(" ")

    def test_chunk_text_no_overlap(self):
        """Test chunking with no overlap."""
        text = "A" * 100
        chunks = chunk_text(text, chunk_size=25, overlap=0)

        assert len(chunks) == 4
        assert all(len(chunk) == 25 for chunk in chunks[:3])
        assert len(chunks[3]) == 25

    def test_chunk_text_large_overlap(self):
        """Test chunking with overlap larger than chunk size."""
        text = "A" * 100
        chunks = chunk_text(text, chunk_size=20, overlap=25)

        # Should still work, though not very useful
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_text_exact_chunk_size(self):
        """Test text that is exactly one chunk size."""
        text = "A" * 50
        chunks = chunk_text(text, chunk_size=50, overlap=10)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_with_newlines(self):
        """Test chunking text with newlines."""
        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        chunks = chunk_text(text, chunk_size=20, overlap=5)

        assert len(chunks) > 1
        # Newlines should be preserved in chunks
        assert any('\n' in chunk for chunk in chunks)

    def test_chunk_text_special_characters(self):
        """Test chunking text with special characters."""
        text = "Hello! @#$%^&*()_+ 世界 🌍 测试"
        chunks = chunk_text(text, chunk_size=15, overlap=3)

        assert len(chunks) >= 1
        # Special characters should be preserved
        combined = "".join(chunks)
        for char in "!@#$%^&*()_+":
            if char in text:
                assert char in combined

    def test_chunk_text_default_parameters(self):
        """Test chunking with default parameters."""
        text = "A" * 2000  # Longer than default chunk size
        chunks = chunk_text(text)

        assert len(chunks) > 1
        # First chunk should be close to default size (900)
        assert 800 <= len(chunks[0]) <= 900

    def test_chunk_text_edge_case_tiny_chunks(self):
        """Test with very small chunk size."""
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=1, overlap=0)

        assert len(chunks) == len(text)
        assert "".join(chunks) == text

    def test_chunk_text_edge_case_negative_overlap(self):
        """Test with negative overlap (should be handled gracefully)."""
        text = "A" * 100
        chunks = chunk_text(text, chunk_size=25, overlap=-5)

        # Should still produce valid chunks
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_text_preserves_content(self):
        """Test that chunking preserves all content."""
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        # Combine all chunks and remove overlap to verify content preservation
        combined = chunks[0]
        for i in range(1, len(chunks)):
            # Find overlap and remove it
            chunk = chunks[i]
            overlap_size = min(20, len(combined), len(chunk))
            for j in range(overlap_size, 0, -1):
                if combined[-j:] == chunk[:j]:
                    combined += chunk[j:]
                    break
            else:
                # No overlap found, just append
                combined += chunk

        # Should contain all original content (minus some whitespace differences)
        assert len(combined.replace(" ", "")) >= len(text.replace(" ", "")) * 0.95

    def test_build_docs_from_files_passthrough(self):
        """Test that build_docs_from_files passes through input unchanged."""
        files = {
            "file1.txt": "Content of file 1",
            "file2.md": "# Header\nContent of file 2",
            "file3.py": "def function():\n    pass"
        }

        result = build_docs_from_files(files)

        assert result == files
        assert result is not files  # Should be a copy, not the same object

    def test_build_docs_from_files_empty(self):
        """Test build_docs_from_files with empty input."""
        result = build_docs_from_files({})
        assert result == {}

    def test_build_docs_from_files_single_file(self):
        """Test build_docs_from_files with single file."""
        files = {"test.txt": "Test content"}
        result = build_docs_from_files(files)

        assert result == files
        assert "test.txt" in result
        assert result["test.txt"] == "Test content"

    def test_build_docs_from_files_with_empty_content(self):
        """Test build_docs_from_files with empty file content."""
        files = {
            "empty.txt": "",
            "whitespace.txt": "   \n\t   ",
            "normal.txt": "Normal content"
        }

        result = build_docs_from_files(files)

        assert result == files
        assert result["empty.txt"] == ""
        assert result["whitespace.txt"] == "   \n\t   "
        assert result["normal.txt"] == "Normal content"

    def test_build_docs_from_files_preserves_types(self):
        """Test that build_docs_from_files preserves data types."""
        files = {
            "file1.txt": "String content",
            "file2.txt": "Another string"
        }

        result = build_docs_from_files(files)

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_chunk_text_realistic_document(self):
        """Test chunking with realistic document content."""
        text = """
        Introduction

        This is a sample document that contains multiple paragraphs and sections.
        It represents typical documentation content that might be chunked for
        processing in a RAG system.

        Main Content

        The main content section contains detailed information about the topic.
        This section is longer and might span multiple chunks depending on the
        chunk size settings. It includes examples, code snippets, and explanations.

        Conclusion

        The conclusion summarizes the key points and provides final thoughts.
        """

        chunks = chunk_text(text, chunk_size=200, overlap=50)

        assert len(chunks) > 1
        # Check that section headers are preserved
        combined_text = " ".join(chunks)
        assert "Introduction" in combined_text
        assert "Main Content" in combined_text
        assert "Conclusion" in combined_text

    def test_chunk_text_unicode_content(self):
        """Test chunking with Unicode content."""
        text = "English text. Español: ¡Hola! 中文：你好 العربية: مرحبا Русский: Привет"
        chunks = chunk_text(text, chunk_size=30, overlap=10)

        assert len(chunks) >= 1
        # Unicode characters should be preserved
        combined = "".join(chunks)
        assert "¡Hola!" in combined
        assert "你好" in combined
        assert "مرحبا" in combined
        assert "Привет" in combined

    def test_chunk_text_boundary_conditions(self):
        """Test edge cases for chunk boundaries."""
        # Test when overlap equals chunk size
        text = "A" * 50
        chunks = chunk_text(text, chunk_size=10, overlap=10)
        assert len(chunks) > 0

        # Test when overlap is larger than remaining text
        text = "Short"
        chunks = chunk_text(text, chunk_size=10, overlap=20)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_consistency(self):
        """Test that chunking is consistent across multiple calls."""
        text = "Consistent text for testing chunking behavior"

        chunks1 = chunk_text(text, chunk_size=20, overlap=5)
        chunks2 = chunk_text(text, chunk_size=20, overlap=5)

        assert chunks1 == chunks2