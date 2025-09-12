"""Request models for Support Deflect Bot API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

class AskRequest(BaseModel):
    """Request model for ask endpoint."""
    question: str = Field(..., min_length=1, max_length=1000, description="Question to ask")
    domains: Optional[List[str]] = Field(None, description="Domain filtering")
    max_chunks: int = Field(5, ge=1, le=20, description="Maximum chunks to retrieve")
    min_confidence: float = Field(0.25, ge=0.0, le=1.0, description="Minimum confidence threshold")
    use_context: bool = Field(True, description="Whether to use retrieved context")
    
    @validator('question')
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    k: int = Field(5, ge=1, le=50, description="Number of results to return")
    domains: Optional[List[str]] = Field(None, description="Domain filtering")
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class IndexRequest(BaseModel):
    """Request model for index endpoint."""
    directory: Optional[str] = Field(None, description="Directory to index")
    force: bool = Field(False, description="Force reindexing")
    recursive: bool = Field(True, description="Process subdirectories")
    file_patterns: Optional[List[str]] = Field(None, description="File patterns to include")

class CrawlRequest(BaseModel):
    """Request model for crawl endpoint."""
    urls: List[str] = Field(..., min_items=1, max_items=10, description="URLs to crawl")
    depth: int = Field(1, ge=1, le=3, description="Crawl depth")
    max_pages: int = Field(40, ge=1, le=100, description="Maximum pages to crawl")
    same_domain: bool = Field(True, description="Stay within same domain")
    
    @validator('urls')
    def validate_urls(cls, v):
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        for url in v:
            if not url_pattern.match(url):
                raise ValueError(f'Invalid URL: {url}')
        return v

class BatchAskRequest(BaseModel):
    """Request model for batch ask endpoint."""
    questions: List[str] = Field(..., min_items=1, max_items=10, description="Questions to ask")
    domains: Optional[List[str]] = Field(None, description="Domain filtering")
    max_chunks: int = Field(5, ge=1, le=20, description="Maximum chunks to retrieve")
    min_confidence: float = Field(0.25, ge=0.0, le=1.0, description="Minimum confidence threshold")
    
    @validator('questions')
    def validate_questions(cls, v):
        for q in v:
            if not q.strip():
                raise ValueError('Questions cannot be empty')
        return [q.strip() for q in v]