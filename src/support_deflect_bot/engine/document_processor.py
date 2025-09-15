"""Unified Document Processor for Support Deflect Bot."""

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.robotparser as robotparser
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

from ..core.providers import get_default_registry, ProviderType, ProviderError, ProviderUnavailableError
from ..utils.settings import (
    ALLOW_HOSTS,
    CRAWL_CACHE_PATH,
    TRUSTED_DOMAINS,
    USER_AGENT,
    CHROMA_COLLECTION
)

class UnifiedDocumentProcessor:
    """
    Unified document processor that handles local files, web content, and batch URL processing.
    Provides document ingestion, web crawling, and source validation capabilities.
    """
    
    def __init__(self, provider_registry=None):
        """Initialize document processor with provider registry and configuration."""
        self.provider_registry = provider_registry or get_default_registry()
        self.cache_path = CRAWL_CACHE_PATH
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        
        self.processing_stats = {
            "files_processed": 0,
            "urls_processed": 0,
            "chunks_created": 0,
            "errors": 0,
            "last_processing_time": None
        }

    def process_local_directory(
        self, 
        directory: str = "./docs", 
        file_extensions: Optional[List[str]] = None,
        chunk_size: int = 900, 
        overlap: int = 150,
        reset_collection: bool = True
    ) -> Dict[str, int]:
        """
        Process local directory and ingest documents into vector database.
        
        Args:
            directory: Path to directory containing documents
            file_extensions: List of file extensions to process (default: .md, .txt)
            chunk_size: Size of text chunks in characters
            overlap: Overlap between chunks in characters
            reset_collection: Whether to reset the collection before ingestion
            
        Returns:
            Dictionary with processing statistics
        """
        if file_extensions is None:
            file_extensions = [".md", ".txt", ".rst", ".py", ".js", ".json", ".yaml", ".yml"]
        
        try:
            # Read all documents from directory
            documents = self._read_documents_from_directory(directory, file_extensions)
            
            if not documents:
                return {"files_processed": 0, "chunks_created": 0, "errors": 0}
            
            # Reset collection if requested
            if reset_collection:
                from data.store import reset_collection
                reset_collection()
            
            # Process documents into chunks
            chunk_texts = []
            metadatas = []
            
            for file_path, content in documents.items():
                chunks = self._chunk_text(content, chunk_size=chunk_size, overlap=overlap)
                for i, chunk in enumerate(chunks):
                    chunk_texts.append(chunk)
                    metadatas.append({
                        "path": file_path,
                        "chunk_id": i,
                        "source_type": "local_file",
                        "host": "localhost"
                    })
            
            # Generate embeddings and store
            chunks_created = self._store_chunks(chunk_texts, metadatas)
            
            self.processing_stats["files_processed"] += len(documents)
            self.processing_stats["chunks_created"] += chunks_created
            
            return {
                "files_processed": len(documents),
                "chunks_created": chunks_created,
                "errors": 0
            }
            
        except Exception as e:
            logging.error(f"Error processing local directory {directory}: {e}")
            self.processing_stats["errors"] += 1
            return {"files_processed": 0, "chunks_created": 0, "errors": 1, "error": str(e)}

    def process_web_content(
        self, 
        urls: List[str],
        force: bool = False,
        chunk_size: int = 900,
        overlap: int = 150
    ) -> Dict[str, Dict[str, int]]:
        """
        Process and index web content from provided URLs.
        
        Args:
            urls: List of URLs to process
            force: Whether to force re-processing (ignore cache)
            chunk_size: Size of text chunks in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            Dictionary mapping URLs to processing statistics
        """
        cache = self._load_cache()
        results = {}
        
        for url in urls:
            stats = {
                "fetched": 0,
                "chunks": 0,
                "replaced": 0,
                "errors": 0,
                "skipped_304": 0,
                "skipped_samehash": 0,
            }
            
            try:
                # Check cache and fetch conditionally
                entry = cache.get(url, {})
                response = self._fetch_html(
                    url,
                    etag=None if force else entry.get("etag"),
                    last_modified=None if force else entry.get("last_modified")
                )
                
                if not force and response.status_code == 304:
                    stats["skipped_304"] = 1
                    results[url] = stats
                    continue
                
                response.raise_for_status()
                html = response.text
                title, text = self._html_to_text(html)
                content_hash = self._sha256(text)
                
                # Check content hash to avoid reprocessing
                if not force and entry.get("content_hash") == content_hash:
                    stats["skipped_samehash"] = 1
                    results[url] = stats
                    continue
                
                # Process and index the content
                chunks_created = self._index_single_url(url, html, title, text, chunk_size, overlap)
                stats["fetched"] = 1
                stats["chunks"] = chunks_created
                stats["replaced"] = 1
                
                # Update cache
                cache[url] = {
                    "etag": response.headers.get("ETag") or response.headers.get("Etag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "content_hash": content_hash,
                    "updated_at": int(time.time()),
                }
                
                self.processing_stats["urls_processed"] += 1
                self.processing_stats["chunks_created"] += chunks_created
                
            except Exception as e:
                logging.error(f"Error processing URL {url}: {e}")
                stats["errors"] += 1
                self.processing_stats["errors"] += 1
            
            results[url] = stats
            time.sleep(0.3)  # Polite crawling delay
        
        self._save_cache(cache)
        return results

    def process_batch_urls(
        self,
        seed_urls: List[str],
        depth: int = 1,
        max_pages: int = 40,
        same_domain: bool = True,
        force: bool = False,
        chunk_size: int = 900,
        overlap: int = 150
    ) -> Dict[str, Dict[str, int]]:
        """
        Process URLs with breadth-first crawling.
        
        Args:
            seed_urls: Initial URLs to start crawling from
            depth: Maximum crawl depth
            max_pages: Maximum number of pages to process
            same_domain: Whether to restrict crawling to same domain
            force: Whether to force re-processing (ignore cache)
            chunk_size: Size of text chunks in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            Dictionary mapping URLs to processing statistics
        """
        cache = self._load_cache()
        visited: Set[str] = set()
        results: Dict[str, Dict[str, int]] = {}
        queue = deque()
        
        # Initialize queue with seed URLs
        for seed_url in seed_urls:
            normalized = self._normalize_url(seed_url)
            if normalized:
                queue.append((normalized, 0))
        
        while queue and len(results) < max_pages:
            url, current_depth = queue.popleft()
            
            if url in visited:
                continue
            visited.add(url)
            
            # Check if URL is allowed
            host = urlparse(url).netloc
            if host not in ALLOW_HOSTS:
                continue
            
            # Check robots.txt
            if not self._robots_ok(url):
                results[url] = {
                    "fetched": 0, "chunks": 0, "replaced": 0, "errors": 0,
                    "skipped_304": 0, "skipped_samehash": 0, "robots_blocked": 1
                }
                continue
            
            stats = {
                "fetched": 0, "chunks": 0, "replaced": 0, "errors": 0,
                "skipped_304": 0, "skipped_samehash": 0
            }
            
            try:
                # Process the URL
                entry = cache.get(url, {})
                response = self._fetch_html(
                    url,
                    etag=None if force else entry.get("etag"),
                    last_modified=None if force else entry.get("last_modified")
                )
                
                if not force and response.status_code == 304:
                    stats["skipped_304"] = 1
                    results[url] = stats
                    continue
                
                response.raise_for_status()
                html = response.text
                title, text = self._html_to_text(html)
                content_hash = self._sha256(text)
                
                # Check content hash
                if not force and entry.get("content_hash") == content_hash:
                    stats["skipped_samehash"] = 1
                    results[url] = stats
                    # Continue to extract links even if content unchanged
                else:
                    chunks_created = self._index_single_url(url, html, title, text, chunk_size, overlap)
                    stats["fetched"] = 1
                    stats["chunks"] = chunks_created
                    stats["replaced"] = 1
                    
                    cache[url] = {
                        "etag": response.headers.get("ETag") or response.headers.get("Etag"),
                        "last_modified": response.headers.get("Last-Modified"),
                        "content_hash": content_hash,
                        "updated_at": int(time.time()),
                    }
                
                results[url] = stats
                
                # Extract and queue links for next depth level
                if current_depth < depth:
                    links = self._extract_links(html, url)
                    for link in links:
                        if same_domain and not self._same_host(url, link):
                            continue
                        if link not in visited:
                            queue.append((link, current_depth + 1))
                
            except Exception as e:
                logging.error(f"Error crawling URL {url}: {e}")
                stats["errors"] += 1
                results[url] = stats
            
            time.sleep(0.3)  # Polite crawling delay
        
        self._save_cache(cache)
        return results

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the current document collection.
        
        Returns:
            Dictionary with collection statistics and metadata
        """
        try:
            from data.store import get_client, get_collection
            
            client = get_client()
            collection = get_collection(client)
            
            # Get basic collection info
            count = collection.count()
            
            # Try to get some sample documents to analyze
            sample_size = min(10, count)
            if sample_size > 0:
                sample = collection.get(limit=sample_size, include=["metadatas"])
                
                # Analyze metadata to get insights
                sources = set()
                hosts = set()
                source_types = set()
                
                for metadata in sample["metadatas"]:
                    if "path" in metadata:
                        sources.add(metadata["path"])
                    if "host" in metadata:
                        hosts.add(metadata["host"])
                    if "source_type" in metadata:
                        source_types.add(metadata["source_type"])
                
                return {
                    "connected": True,
                    "collection_name": CHROMA_COLLECTION,
                    "total_chunks": count,
                    "sample_sources": list(sources),
                    "unique_hosts": list(hosts),
                    "source_types": list(source_types),
                    "processing_stats": self.processing_stats
                }
            else:
                return {
                    "connected": True,
                    "collection_name": CHROMA_COLLECTION,
                    "total_chunks": 0,
                    "processing_stats": self.processing_stats
                }
                
        except Exception as e:
            logging.error(f"Error getting collection stats: {e}")
            return {
                "connected": False,
                "error": str(e),
                "collection_name": CHROMA_COLLECTION,
                "processing_stats": self.processing_stats
            }

    def validate_sources(self, sources: List[str]) -> Dict[str, bool]:
        """
        Validate that sources are accessible and processable.
        
        Args:
            sources: List of file paths or URLs to validate
            
        Returns:
            Dictionary mapping sources to their validity status
        """
        results = {}
        
        for source in sources:
            try:
                if source.startswith(('http://', 'https://')):
                    # Validate URL
                    response = requests.head(source, timeout=10, headers={"User-Agent": USER_AGENT})
                    results[source] = response.status_code < 400
                else:
                    # Validate local file/directory
                    results[source] = os.path.exists(source) and os.access(source, os.R_OK)
            except Exception as e:
                logging.warning(f"Error validating source {source}: {e}")
                results[source] = False
        
        return results

    # Private helper methods
    
    def _read_documents_from_directory(self, directory: str, extensions: List[str]) -> Dict[str, str]:
        """Read all documents from directory with specified extensions."""
        documents = {}
        
        for root, _, files in os.walk(directory):
            for filename in files:
                if any(filename.lower().endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            documents[file_path] = f.read()
                    except Exception as e:
                        logging.warning(f"Error reading file {file_path}: {e}")
                        continue
        
        return documents
    
    def _chunk_text(self, text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
        """Split text into overlapping chunks."""
        text = text.strip()
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end == text_length:
                break
                
            start = end - overlap
            if start < 0:
                start = 0
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def _store_chunks(self, chunk_texts: List[str], metadatas: List[Dict]) -> int:
        """Store chunks in vector database with embeddings."""
        if not chunk_texts:
            return 0
        
        try:
            # Generate embeddings using provider chain
            embedding_chain = self.provider_registry.build_fallback_chain(ProviderType.EMBEDDING)
            embeddings = None
            
            for provider in embedding_chain:
                try:
                    embeddings = provider.embed_texts(chunk_texts, batch_size=10)
                    break
                except (ProviderError, ProviderUnavailableError, Exception) as e:
                    logging.warning(f"Embedding provider {provider.get_config().name} failed: {e}")
                    continue
            
            if embeddings is None:
                logging.error("All embedding providers failed")
                return 0
            
            # Store in ChromaDB
            from data.store import get_collection
            collection = get_collection()
            
            ids = [f"{meta['path']}#{meta['chunk_id']}" for meta in metadatas]
            collection.add(
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            return len(chunk_texts)
            
        except Exception as e:
            logging.error(f"Error storing chunks: {e}")
            return 0
    
    def _index_single_url(self, url: str, html: str, title: str, text: str, chunk_size: int, overlap: int) -> int:
        """Index a single URL by chunking and storing its content."""
        try:
            # Remove existing content for this URL
            from data.store import get_collection
            collection = get_collection()
            
            try:
                collection.delete(where={"path": url})
            except Exception as e:
                logging.debug(f"Failed to delete existing content for {url}: {e}")
            
            # Create chunks
            chunks = self._chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                return 0
            
            # Generate embeddings
            embedding_chain = self.provider_registry.build_fallback_chain(ProviderType.EMBEDDING)
            embeddings = None
            
            for provider in embedding_chain:
                try:
                    embeddings = provider.embed_texts(chunks, batch_size=10)
                    break
                except (ProviderError, ProviderUnavailableError, Exception) as e:
                    logging.warning(f"Embedding provider {provider.get_config().name} failed: {e}")
                    continue
            
            if embeddings is None:
                logging.error("All embedding providers failed")
                return 0
            
            # Prepare metadata
            host = urlparse(url).netloc
            metadatas = [
                {
                    "path": url,
                    "title": title,
                    "chunk_id": i,
                    "host": host,
                    "source_type": "web_content"
                }
                for i in range(len(chunks))
            ]
            
            # Store in database
            ids = [f"{url}#{i}" for i in range(len(chunks))]
            collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            return len(chunks)
            
        except Exception as e:
            logging.error(f"Error indexing URL {url}: {e}")
            return 0
    
    def _fetch_html(self, url: str, etag: Optional[str] = None, last_modified: Optional[str] = None, timeout: int = 20):
        """Fetch HTML content with conditional headers."""
        headers = {"User-Agent": USER_AGENT}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        return requests.get(url, headers=headers, timeout=timeout)
    
    def _html_to_text(self, html: str) -> Tuple[str, str]:
        """Convert HTML to clean text and extract title."""
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        
        # Remove noise elements
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        for tag in soup.find_all(["nav", "aside", "footer"]):
            tag.decompose()
        noisy = soup.select(".sidebar, .toc, .sphinxsidebar, .related, .breadcrumbs")
        for element in noisy:
            element.decompose()
        
        # Extract structured text
        lines = []
        main = soup.find("main") or soup.find("div", {"role": "main"}) or soup.body or soup
        
        def append_text(text: str):
            text = (text or "").strip()
            if text:
                lines.append(text)
        
        for element in main.descendants:
            if isinstance(element, Tag):
                tag_name = element.name.lower()
                if tag_name in {"h1", "h2", "h3", "h4"}:
                    append_text(f"# {element.get_text(' ', strip=True)}")
                elif tag_name in {"p", "li"}:
                    append_text(element.get_text(" ", strip=True))
                elif tag_name == "pre":
                    text = element.get_text("\n", strip=True)
                    if text:
                        append_text("```")
                        append_text(text)
                        append_text("```")
                elif tag_name == "code":
                    text = element.get_text("", strip=True)
                    if text:
                        append_text(f"`{text}`")
        
        return title, "\n\n".join(lines)
    
    def _extract_links(self, html: str, base_url: str) -> Set[str]:
        """Extract and normalize links from HTML."""
        soup = BeautifulSoup(html, "lxml")
        links = set()
        
        for anchor in soup.find_all("a", href=True):
            url = self._normalize_url(anchor["href"], base=base_url)
            if not url:
                continue
            host = urlparse(url).netloc
            if host in ALLOW_HOSTS:
                links.add(url)
        
        return links
    
    def _normalize_url(self, href: str, base: Optional[str] = None) -> Optional[str]:
        """Normalize URL by resolving relative paths and cleaning."""
        if not href:
            return None
        
        absolute_url = urljoin(base, href) if base else href
        parsed = urlparse(absolute_url)
        
        if parsed.scheme not in ("http", "https"):
            return None
        
        # Remove fragment and query for stability
        cleaned = parsed._replace(fragment="", query="")
        return cleaned.geturl()
    
    def _same_host(self, url1: str, url2: str) -> bool:
        """Check if two URLs have the same host."""
        return urlparse(url1).netloc == urlparse(url2).netloc
    
    def _robots_ok(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        parsed_url = urlparse(url)
        host = parsed_url.netloc
        
        # Bypass robots.txt for trusted domains
        if host in TRUSTED_DOMAINS:
            return True
        
        try:
            host_url = f"{parsed_url.scheme}://{host}"
            robots_url = urljoin(host_url, "/robots.txt")
            
            response = self._fetch_html(robots_url, timeout=10)
            response.raise_for_status()
            
            # Parse robots.txt content
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(response.text)
                temp_path = f.name
            
            try:
                rp = robotparser.RobotFileParser()
                rp.set_url("file://" + temp_path)
                rp.read()
                return rp.can_fetch(USER_AGENT, url)
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            # If robots.txt check fails, allow by default
            logging.info(f"Robots.txt check failed for {url}: {e}. Allowing crawl.")
            return True
    
    def _sha256(self, text: str) -> str:
        """Generate SHA256 hash of text."""
        return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
    
    def _load_cache(self) -> Dict[str, Dict]:
        """Load processing cache from file."""
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_cache(self, cache: Dict[str, Dict]):
        """Save processing cache to file."""
        temp_path = self.cache_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.cache_path)