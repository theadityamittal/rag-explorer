"""Retrieve compatibility bridge - bridges old retrieve function to new system."""

from typing import Dict, List, Optional
import logging
from ._path_helper import ensure_src_path

logger = logging.getLogger(__name__)


def retrieve(query: str, k: int = 5, domains: Optional[List[str]] = None) -> List[Dict]:
    """
    Search through indexed documents using the old interface.
    
    This function provides the same interface as src.core.retrieve.retrieve()
    but can potentially use new provider systems for embeddings.
    
    Args:
        query: Search query string
        k: Number of results to return
        domains: Optional domain filtering
        
    Returns:
        List of search hit dictionaries
    """
    try:
        # For now, fallback to old system since it's complex to migrate
        # TODO: Eventually use new embedding providers here
        ensure_src_path()
        from src.core.retrieve import retrieve as old_retrieve
        return old_retrieve(query, k=k, domains=domains)
        
    except ImportError as e:
        logger.error(f"Old retrieve system not available: {e}")
        raise RuntimeError(
            "Search system not available. Please ensure the system is properly configured."
        )
    except Exception as e:
        logger.error(f"Retrieve operation failed: {e}")
        raise RuntimeError(f"Search failed: {e}")