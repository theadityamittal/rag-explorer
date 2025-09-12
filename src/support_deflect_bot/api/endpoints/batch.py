"""Batch processing endpoints for Support Deflect Bot API."""

import time
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from ..models.requests import BatchAskRequest, CrawlRequest
from ..models.responses import BatchAskResponse, CrawlResponse, AskResponse, Source
from ..dependencies.engine import get_rag_engine, get_document_processor
from ...engine import UnifiedRAGEngine, UnifiedDocumentProcessor

router = APIRouter(prefix="/api/v1/batch", tags=["batch"])

@router.post("/ask", response_model=BatchAskResponse)
async def batch_ask_questions(
    request: BatchAskRequest,
    engine: UnifiedRAGEngine = Depends(get_rag_engine)
) -> BatchAskResponse:
    """Process multiple questions in batch (duplicate of /api/v1/batch_ask for organization)."""
    try:
        start_time = time.time()
        
        results = []
        successful_count = 0
        
        for question in request.questions:
            try:
                question_start = time.time()
                result = engine.answer_question(
                    question=question,
                    domains=request.domains,
                    k=request.max_chunks,
                    min_confidence=request.min_confidence
                )
                question_time = time.time() - question_start
                
                # Convert sources to response model format
                sources = []
                for citation in result.get("citations", []):
                    sources.append(Source(
                        id=citation.get("id"),
                        content=citation.get("text", ""),
                        metadata=citation.get("metadata", {}),
                        distance=citation.get("distance", 0.0)
                    ))
                
                ask_response = AskResponse(
                    answer=result.get("answer", ""),
                    confidence=result.get("confidence", 0.0),
                    sources=sources,
                    chunks_used=len(result.get("citations", [])),
                    response_time=question_time,
                    provider_used=result.get("provider_used", "unknown"),
                    metadata=result.get("metadata", {})
                )
                
                results.append(ask_response)
                successful_count += 1
                
            except Exception as e:
                # Create error response for failed question
                error_response = AskResponse(
                    answer=f"Error processing question: {str(e)}",
                    confidence=0.0,
                    sources=[],
                    chunks_used=0,
                    response_time=0.0,
                    provider_used="error",
                    metadata={"error": str(e)}
                )
                results.append(error_response)
        
        total_time = time.time() - start_time
        
        return BatchAskResponse(
            results=results,
            total_questions=len(request.questions),
            successful_answers=successful_count,
            processing_time=total_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )

@router.post("/crawl", response_model=CrawlResponse)
async def batch_crawl_urls(
    request: CrawlRequest,
    processor: UnifiedDocumentProcessor = Depends(get_document_processor)
) -> CrawlResponse:
    """Batch crawl multiple URL groups (extended crawling functionality)."""
    try:
        start_time = time.time()
        
        # This could be enhanced to handle more sophisticated batch crawling
        # For now, it delegates to the regular crawl endpoint logic
        all_results = {}
        processed_count = 0
        failed_count = 0
        
        for url in request.urls:
            try:
                result = processor.process_web_content(
                    url=url,
                    max_depth=request.depth,
                    max_pages=request.max_pages,
                    same_domain_only=request.same_domain
                )
                
                if result:
                    processed_count += 1
                    all_results[url] = result
                else:
                    failed_count += 1
                    
            except Exception as url_error:
                failed_count += 1
                all_results[url] = {"error": str(url_error)}
        
        processing_time = time.time() - start_time
        success = processed_count > 0
        
        return CrawlResponse(
            success=success,
            processed_count=processed_count,
            failed_count=failed_count,
            processing_time=processing_time,
            urls=request.urls,
            crawl_details={
                "depth": request.depth,
                "max_pages": request.max_pages,
                "same_domain": request.same_domain,
                "batch_mode": True
            },
            details=[],  # Could be enhanced with detailed results
            error_messages=[]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch crawling failed: {str(e)}"
        )

@router.get("/status")
async def get_batch_status() -> dict:
    """Get batch processing status and queue information."""
    # This is a placeholder for more sophisticated batch job tracking
    return {
        "batch_processing": "available",
        "queue_status": "not_implemented",
        "active_jobs": 0,
        "completed_jobs": "not_tracked",
        "timestamp": time.time()
    }