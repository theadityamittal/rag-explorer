"""Unified Document Processor for RAG Explorer engine."""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from rag_explorer.core.registry import ProviderRegistry, ProviderNotConfiguredError, ProviderNotAvailableError
from .database import simple_add_documents_with_embeddings, simple_reset_collection, simple_get_collection_count
from data.chunker import chunk_text
from rag_explorer.utils.settings import CHROMA_COLLECTION

logger = logging.getLogger(__name__)


class UnifiedDocumentProcessor:
    """Unified document processor that handles indexing local documents into vector database."""

    def __init__(self):
        """Initialize the document processor."""
        self.registry = ProviderRegistry()
        logger.debug("Initialized UnifiedDocumentProcessor")

    def process_local_directory(
        self,
        directory: str,
        chunk_size: int = 1000,
        overlap: int = 150,
        reset_collection: bool = False
    ) -> Dict[str, Any]:
        """Process local directory and index documents into vector database.

        Args:
            directory: Path to directory containing documents
            chunk_size: Size of text chunks in characters
            overlap: Overlap between chunks in characters
            reset_collection: Whether to reset the collection before adding documents

        Returns:
            Dictionary with operation results:
                - status: Operation status
                - count: Number of document chunks indexed
                - files_processed: Number of files processed
                - collection: Collection name used

        Raises:
            ConnectionError: If providers are not configured properly
            RuntimeError: If processing operation fails
            ValueError: If directory doesn't exist or parameters are invalid
        """
        if not directory or not directory.strip():
            raise ValueError("Directory path cannot be empty")

        directory_path = Path(directory).resolve()
        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")

        if overlap < 0:
            raise ValueError("Overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk size")

        try:
            # Get embedding provider
            embedding_provider = self._get_embedding_provider()

            # Reset collection if requested
            if reset_collection:
                logger.info(f"Resetting collection: {CHROMA_COLLECTION}")
                simple_reset_collection(CHROMA_COLLECTION)

            # Read documents from directory
            documents = self._read_documents_from_directory(str(directory_path))

            if not documents:
                logger.warning(f"No supported documents found in: {directory}")
                return {
                    "status": "success",
                    "count": 0,
                    "files_processed": 0,
                    "collection": CHROMA_COLLECTION,
                    "message": "No supported documents found"
                }

            # Process documents into chunks
            chunk_texts = []
            metadatas = []
            total_chunks = 0

            for file_path, content in documents.items():
                if not content.strip():
                    logger.warning(f"Skipping empty file: {file_path}")
                    continue

                # Chunk the document
                chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)

                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Only add non-empty chunks
                        chunk_texts.append(chunk)
                        metadatas.append({
                            "source": file_path,
                            "path": file_path,
                            "chunk_id": i,
                            "chunk_size": len(chunk),
                            "total_chunks": len(chunks)
                        })

                total_chunks += len(chunks)

            if not chunk_texts:
                logger.warning("No valid text chunks found after processing")
                return {
                    "status": "success",
                    "count": 0,
                    "files_processed": len(documents),
                    "collection": CHROMA_COLLECTION,
                    "message": "No valid text chunks found"
                }

            # Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
            embeddings = embedding_provider.embed_texts(chunk_texts, batch_size=10)

            if len(embeddings) != len(chunk_texts):
                raise RuntimeError(f"Embedding count mismatch: {len(embeddings)} vs {len(chunk_texts)}")

            # Store in database
            logger.info(f"Storing {len(chunk_texts)} chunks in database...")
            simple_add_documents_with_embeddings(
                texts=chunk_texts,
                embeddings=embeddings,
                metadatas=metadatas,
                collection_name=CHROMA_COLLECTION
            )

            # Get final count
            final_count = simple_get_collection_count(CHROMA_COLLECTION)

            logger.info(f"Successfully processed {len(documents)} files into {len(chunk_texts)} chunks")

            return {
                "status": "success",
                "count": len(chunk_texts),
                "files_processed": len(documents),
                "collection": CHROMA_COLLECTION,
                "total_documents": final_count
            }

        except (ProviderNotConfiguredError, ProviderNotAvailableError) as e:
            raise ConnectionError(str(e))
        except ValueError as e:
            # Re-raise ValueError (input validation)
            raise
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise RuntimeError(f"Document processing operation failed: {e}")

    def _read_documents_from_directory(self, directory: str) -> Dict[str, str]:
        """Read supported documents from directory recursively.

        Args:
            directory: Directory path to read from

        Returns:
            Dictionary mapping file paths to content

        Raises:
            RuntimeError: If file reading fails
        """
        supported_extensions = {'.md', '.txt', '.py', '.js', '.json', '.yaml', '.yml', '.rst'}
        documents = {}
        files_read = 0
        files_skipped = 0

        try:
            for root, dirs, files in os.walk(directory):
                # Skip hidden directories and common non-document directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {
                    '__pycache__', 'node_modules', '.git', '.venv', 'venv', 'env'
                }]

                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = Path(file).suffix.lower()

                    # Skip hidden files and unsupported extensions
                    if file.startswith('.') or file_ext not in supported_extensions:
                        files_skipped += 1
                        continue

                    try:
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        if content.strip():  # Only add files with content
                            documents[file_path] = content
                            files_read += 1
                        else:
                            logger.warning(f"Skipping empty file: {file_path}")
                            files_skipped += 1

                    except Exception as e:
                        logger.warning(f"Failed to read file {file_path}: {e}")
                        files_skipped += 1
                        continue

            logger.info(f"Read {files_read} files, skipped {files_skipped} files from: {directory}")
            return documents

        except Exception as e:
            raise RuntimeError(f"Failed to read documents from directory: {e}")

    def _get_embedding_provider(self):
        """Get embedding provider with clear error messages."""
        try:
            return self.registry.get_embedding_provider()
        except ProviderNotConfiguredError as e:
            # Extract provider name from error and provide clear guidance
            provider_name = str(e).split()[0].lower() if str(e) else "unknown"
            if "openai" in provider_name:
                raise ConnectionError("OpenAI provider not configured. Please set OPENAI_API_KEY environment variable or change PRIMARY_EMBEDDING_PROVIDER setting.")
            elif "google" in provider_name:
                raise ConnectionError("Google provider not configured. Please set GEMINI_API_KEY environment variable or change PRIMARY_EMBEDDING_PROVIDER setting.")
            elif "ollama" in provider_name:
                raise ConnectionError("Ollama provider not configured. Please ensure Ollama is running or change PRIMARY_EMBEDDING_PROVIDER setting.")
            else:
                raise ConnectionError(f"Embedding provider not configured: {e}")
        except ProviderNotAvailableError as e:
            raise ConnectionError(f"Embedding provider is not available: {e}")

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported file extensions
        """
        return ['.md', '.txt', '.py', '.js', '.json', '.yaml', '.yml', '.rst']

    def get_processing_stats(self, directory: str) -> Dict[str, Any]:
        """Get statistics about documents in directory without processing them.

        Args:
            directory: Directory path to analyze

        Returns:
            Dictionary with statistics about the directory

        Raises:
            ValueError: If directory doesn't exist
        """
        if not os.path.exists(directory):
            raise ValueError(f"Directory does not exist: {directory}")

        if not os.path.isdir(directory):
            raise ValueError(f"Path is not a directory: {directory}")

        try:
            supported_extensions = set(self.get_supported_extensions())
            stats = {
                "directory": directory,
                "total_files": 0,
                "supported_files": 0,
                "unsupported_files": 0,
                "empty_files": 0,
                "total_size_bytes": 0,
                "file_types": {},
                "largest_file": None,
                "largest_file_size": 0
            }

            for root, dirs, files in os.walk(directory):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]

                for file in files:
                    if file.startswith('.'):
                        continue

                    file_path = os.path.join(root, file)
                    file_ext = Path(file).suffix.lower()
                    stats["total_files"] += 1

                    try:
                        file_size = os.path.getsize(file_path)
                        stats["total_size_bytes"] += file_size

                        if file_size > stats["largest_file_size"]:
                            stats["largest_file_size"] = file_size
                            stats["largest_file"] = file_path

                        if file_ext in supported_extensions:
                            stats["supported_files"] += 1

                            # Check if file is empty
                            if file_size == 0:
                                stats["empty_files"] += 1

                            # Track file types
                            if file_ext in stats["file_types"]:
                                stats["file_types"][file_ext] += 1
                            else:
                                stats["file_types"][file_ext] = 1
                        else:
                            stats["unsupported_files"] += 1

                    except Exception as e:
                        logger.warning(f"Failed to stat file {file_path}: {e}")
                        continue

            return stats

        except Exception as e:
            raise RuntimeError(f"Failed to analyze directory: {e}")