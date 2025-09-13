"""Indexing endpoints for Support Deflect Bot API."""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
import time
import os

from ..models.requests import IndexRequest, CrawlRequest
from ..models.responses import IndexResponse, CrawlResponse, ProcessingDetail
from ..dependencies.engine import get_document_processor
from ...engine.document_processor import UnifiedDocumentProcessor

router = APIRouter(prefix="/api/v1", tags=["indexing"])

@router.post("/reindex", response_model=IndexResponse)
async def reindex_documents(
    request: IndexRequest,
    processor: UnifiedDocumentProcessor = Depends(get_document_processor)
) -> IndexResponse:
    """Reindex documents from a directory."""
    try:
        start_time = time.time()
        
        # Use current directory if none specified
        directory = request.directory or os.getcwd()
        
        if not os.path.exists(directory):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Directory does not exist: {directory}"
            )
        
        if not os.path.isdir(directory):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path is not a directory: {directory}"
            )
        
        # Process local directory using unified document processor
        result = await processor.process_local_directory(
            directory_path=directory,
            recursive=request.recursive,
            force_reprocess=request.force,
            file_patterns=request.file_patterns
        )
        
        processing_time = time.time() - start_time
        
        # Convert processing results to response format
        details = []
        error_messages = []
        
        for item in result.get("processed_files", []):
            details.append(ProcessingDetail(
                status="success",
                path=item.get("path"),
                error=None
            ))
        
        for item in result.get("failed_files", []):
            error_msg = item.get("error", "Unknown error")
            error_messages.append(f"{item.get('path', 'Unknown file')}: {error_msg}")
            details.append(ProcessingDetail(
                status="failed",
                path=item.get("path"),
                error=error_msg
            ))
        
        return IndexResponse(
            success=result.get("success", True),
            processed_count=result.get("processed_count", 0),
            failed_count=result.get("failed_count", 0),
            processing_time=processing_time,
            directory=directory,
            details=details,
            error_messages=error_messages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )

@router.post("/crawl", response_model=CrawlResponse)
async def crawl_urls(
    request: CrawlRequest,
    processor: UnifiedDocumentProcessor = Depends(get_document_processor)
) -> CrawlResponse:
    """Crawl and index web content."""
    try:
        start_time = time.time()
        
        # Process web content using unified document processor
        result = await processor.process_web_content(
            urls=request.urls,
            crawl_depth=request.depth,
            max_pages=request.max_pages,
            same_domain_only=request.same_domain
        )
        
        processing_time = time.time() - start_time
        
        # Convert processing results to response format
        details = []
        error_messages = []
        
        for item in result.get("processed_urls", []):
            details.append(ProcessingDetail(
                status="success",
                url=item.get("url"),
                error=None
            ))
        
        for item in result.get("failed_urls", []):
            error_msg = item.get("error", "Unknown error")
            error_messages.append(f"{item.get('url', 'Unknown URL')}: {error_msg}")
            details.append(ProcessingDetail(
                status="failed",
                url=item.get("url"),
                error=error_msg
            ))
        
        return CrawlResponse(
            success=result.get("success", True),
            processed_count=result.get("processed_count", 0),
            failed_count=result.get("failed_count", 0),
            processing_time=processing_time,
            urls=request.urls,
            crawl_details={
                "depth": request.depth,
                "max_pages": request.max_pages,
                "same_domain": request.same_domain
            },
            details=details,
            error_messages=error_messages
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Crawling failed: {str(e)}"
        )