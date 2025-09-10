"""Compatibility layer for bridging old and new provider systems."""

# Re-export key functions for easy importing
from .llm_bridge import llm_echo, llm_chat
from .rag_bridge import answer_question
from .retrieve_bridge import retrieve
from .data_bridge import ingest_folder, crawl_urls, index_urls
from .utils_bridge import Meter, init_clean_cli

__all__ = [
    'llm_echo',
    'llm_chat', 
    'answer_question',
    'retrieve',
    'ingest_folder',
    'crawl_urls', 
    'index_urls',
    'Meter',
    'init_clean_cli',
]