"""Response models for Support Deflect Bot API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Source(BaseModel):
    """Source document information."""
    id: Optional[str] = Field(None, description="Document ID")
    content: str = Field(..., description="Document content excerpt")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    distance: float = Field(..., ge=0.0, le=2.0, description="Similarity distance")

class AskResponse(BaseModel):
    """Response model for ask endpoint."""
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    sources: List[Source] = Field(default_factory=list, description="Source documents")
    chunks_used: int = Field(..., ge=0, description="Number of chunks used")
    response_time: float = Field(..., ge=0.0, description="Response time in seconds")
    provider_used: str = Field(..., description="LLM provider used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class SearchResult(BaseModel):
    """Search result item."""
    id: Optional[str] = Field(None, description="Document ID")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    distance: float = Field(..., ge=0.0, le=2.0, description="Similarity distance")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")

class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_count: int = Field(..., ge=0, description="Total number of results")
    query: str = Field(..., description="Original query")
    response_time: float = Field(..., ge=0.0, description="Response time in seconds")

class ProcessingDetail(BaseModel):
    """Processing detail item."""
    status: str = Field(..., description="Processing status")
    path: Optional[str] = Field(None, description="File path")
    url: Optional[str] = Field(None, description="URL")
    error: Optional[str] = Field(None, description="Error message if failed")

class IndexResponse(BaseModel):
    """Response model for index endpoint."""
    success: bool = Field(..., description="Success status")
    processed_count: int = Field(..., ge=0, description="Number of documents processed")
    failed_count: int = Field(..., ge=0, description="Number of documents failed")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    directory: Optional[str] = Field(None, description="Directory processed")
    details: List[ProcessingDetail] = Field(default_factory=list, description="Processing details")
    error_messages: List[str] = Field(default_factory=list, description="Error messages")

class CrawlResponse(BaseModel):
    """Response model for crawl endpoint."""
    success: bool = Field(..., description="Success status")
    processed_count: int = Field(..., ge=0, description="Number of URLs processed")
    failed_count: int = Field(..., ge=0, description="Number of URLs failed")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    urls: List[str] = Field(default_factory=list, description="URLs processed")
    crawl_details: Dict[str, Any] = Field(default_factory=dict, description="Crawl configuration")
    details: List[ProcessingDetail] = Field(default_factory=list, description="Processing details")
    error_messages: List[str] = Field(default_factory=list, description="Error messages")

class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    providers: Dict[str, Any] = Field(default_factory=dict, description="Provider status")
    database: Dict[str, Any] = Field(default_factory=dict, description="Database status")

class BatchAskResponse(BaseModel):
    """Response model for batch ask endpoint."""
    results: List[AskResponse] = Field(default_factory=list, description="Individual ask results")
    total_questions: int = Field(..., ge=0, description="Total questions processed")
    successful_answers: int = Field(..., ge=0, description="Successful answers")
    processing_time: float = Field(..., ge=0.0, description="Total processing time in seconds")
    
class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")