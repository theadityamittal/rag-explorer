"""Data module compatibility bridge - bridges old data functions to new or existing implementations."""

from typing import List, Dict, Any
from ._path_helper import ensure_src_path

def ingest_folder(folder_path: str) -> int:
    """
    Ingest folder using old data.ingest module.
    
    Args:
        folder_path: Path to folder to ingest
        
    Returns:
        Number of chunks ingested
    """
    try:
        ensure_src_path()
        from src.data.ingest import ingest_folder as old_ingest_folder
        return old_ingest_folder(folder_path)
    except ImportError as e:
        raise ImportError(f"Old data.ingest module not available: {e}")


def crawl_urls(seeds: List[str], depth: int = 1, max_pages: int = 30, 
              same_domain: bool = True, force: bool = False) -> Dict[str, Any]:
    """
    Crawl URLs using old data.web_ingest module.
    
    Args:
        seeds: List of seed URLs
        depth: Crawl depth
        max_pages: Maximum pages to crawl
        same_domain: Restrict to same domain
        force: Force re-index
        
    Returns:
        Crawl result dictionary
    """
    try:
        ensure_src_path()
        from src.data.web_ingest import crawl_urls as old_crawl_urls
        return old_crawl_urls(seeds=seeds, depth=depth, max_pages=max_pages,
                             same_domain=same_domain, force=force)
    except ImportError as e:
        raise ImportError(f"Old data.web_ingest module not available: {e}")


def index_urls(urls: List[str], force: bool = False) -> Dict[str, Any]:
    """
    Index URLs using old data.web_ingest module.
    
    Args:
        urls: List of URLs to index
        force: Force re-index
        
    Returns:
        Index result dictionary
    """
    try:
        ensure_src_path()
        from src.data.web_ingest import index_urls as old_index_urls
        return old_index_urls(urls=urls, force=force)
    except ImportError as e:
        raise ImportError(f"Old data.web_ingest module not available: {e}")