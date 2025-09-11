"""New unified CLI using shared engine services."""

import time
from typing import List, Optional

import click
from rich.console import Console

from ..engine import UnifiedRAGEngine, UnifiedDocumentProcessor, UnifiedQueryService, UnifiedEmbeddingService

# Import settings with fallback
try:
    from ..utils.settings import (
        ANSWER_MIN_CONF,
        CRAWL_DEPTH,
        CRAWL_MAX_PAGES,
        CRAWL_SAME_DOMAIN,
        DEFAULT_SEEDS,
        DOCS_FOLDER,
        MAX_CHUNKS,
    )
    APP_NAME = "Support Deflect Bot"
    APP_VERSION = "2.0.0"
except ImportError:
    # Fallback to old settings
    from support_deflect_bot_old.utils.settings import (
        ANSWER_MIN_CONF,
        APP_NAME,
        APP_VERSION,
        CRAWL_DEPTH,
        CRAWL_MAX_PAGES,
        CRAWL_SAME_DOMAIN,
        DEFAULT_SEEDS,
        DOCS_FOLDER,
        MAX_CHUNKS,
    )

from .output import format_answer, format_search_results
from .ask_session import UnifiedAskSession

console = Console()

# Initialize engine services
_rag_engine = None
_doc_processor = None
_query_service = None
_embedding_service = None


def get_rag_engine() -> UnifiedRAGEngine:
    """Get or create RAG engine instance."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = UnifiedRAGEngine()
    return _rag_engine


def get_doc_processor() -> UnifiedDocumentProcessor:
    """Get or create document processor instance."""
    global _doc_processor
    if _doc_processor is None:
        _doc_processor = UnifiedDocumentProcessor()
    return _doc_processor


def get_query_service() -> UnifiedQueryService:
    """Get or create query service instance."""
    global _query_service
    if _query_service is None:
        _query_service = UnifiedQueryService()
    return _query_service


def get_embedding_service() -> UnifiedEmbeddingService:
    """Get or create embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = UnifiedEmbeddingService()
    return _embedding_service


@click.group()
@click.version_option(version=APP_VERSION, prog_name=APP_NAME)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
@click.pass_context
def cli(ctx, verbose, quiet):
    """Support Deflect Bot - Unified CLI with shared engine services."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


@cli.command()
@click.option("--docs-path", help="Override documentation folder path")
@click.option("--reset", is_flag=True, help="Reset collection before indexing")
@click.pass_context
def index(ctx, docs_path, reset):
    """Index local documentation using unified document processor."""
    try:
        folder_path = docs_path or DOCS_FOLDER
        
        if not ctx.obj["quiet"]:
            console.print(f"🔍 Indexing local documentation from {folder_path}...", style="cyan")
        
        doc_processor = get_doc_processor()
        result = doc_processor.process_local_directory(
            directory=folder_path,
            reset_collection=reset
        )
        
        if ctx.obj["quiet"]:
            console.print(f"{result['chunks_created']}")
        else:
            console.print(
                f" Successfully indexed {result['files_processed']} files "
                f"({result['chunks_created']} chunks) from {folder_path}", 
                style="green"
            )
            
            if result.get('errors', 0) > 0:
                console.print(f"⚠️  {result['errors']} files had errors", style="yellow")
    
    except Exception as e:
        console.print(f"❌ Indexing failed: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()


@cli.command()
@click.argument("query")
@click.option("--limit", "-l", default=5, help="Number of results to return (1-20)")
@click.option("--domains", help="Comma-separated list of domains to filter")
@click.option("--output", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.pass_context
def search(ctx, query, limit, domains, output):
    """Search through indexed documents using unified query service."""
    if limit < 1 or limit > 20:
        console.print("❌ Limit must be between 1 and 20", style="red")
        raise click.Abort()
    
    domain_list = [d.strip() for d in domains.split(",")] if domains else None
    
    try:
        if not ctx.obj["quiet"]:
            console.print(f"🔍 Searching for: '{query}'", style="cyan")
        
        query_service = get_query_service()
        
        # Preprocess query
        processed_query = query_service.preprocess_query(query)
        if not processed_query["valid"]:
            console.print(f"❌ Invalid query: {processed_query['reason']}", style="red")
            raise click.Abort()
        
        # Retrieve documents
        results = query_service.retrieve_documents(
            processed_query,
            k=limit,
            domains=domain_list
        )
        
        if output == "json":
            import json
            output_data = {
                "query": query,
                "processed_query": processed_query,
                "results": [
                    {
                        "text": r["text"][:400],
                        "path": r["meta"].get("path"),
                        "distance": r.get("distance"),
                        "relevance_score": r.get("relevance_score")
                    }
                    for r in results
                ]
            }
            console.print(json.dumps(output_data, indent=2))
        else:
            format_search_results(console, query, results, ctx.obj["quiet"])
    
    except Exception as e:
        console.print(f"L Search failed: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()


@cli.command()
@click.option("--domains", help="Comma-separated list of domains to filter")
@click.option("--confidence", type=float, help="Override confidence threshold (0.0-1.0)")
@click.option("--max-chunks", type=int, help="Override max chunks to retrieve")
@click.pass_context
def ask(ctx, domains, confidence, max_chunks):
    """Start interactive Q&A session using unified RAG engine."""
    domain_list = [d.strip() for d in domains.split(",")] if domains else None
    
    try:
        rag_engine = get_rag_engine()
        query_service = get_query_service()
        
        session = UnifiedAskSession(
            rag_engine=rag_engine,
            query_service=query_service,
            console=console,
            domain_filter=domain_list,
            confidence_override=confidence,
            max_chunks_override=max_chunks or MAX_CHUNKS,
            verbose=ctx.obj["verbose"],
            quiet=ctx.obj["quiet"]
        )
        
        session.start()
        
    except KeyboardInterrupt:
        console.print("\n=K Session interrupted. Goodbye!", style="yellow")
    except Exception as e:
        console.print(f"L Session failed: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()


@cli.command()
@click.argument("urls", nargs=-1, required=False)
@click.option("--force", is_flag=True, help="Force re-index even if content unchanged")
@click.option("--depth", type=int, help="Crawl depth (1-3)")
@click.option("--max-pages", type=int, help="Maximum pages to crawl (1-100)")
@click.option("--same-domain", is_flag=True, help="Restrict to same domain")
@click.option("--default", "use_default", is_flag=True, help="Use configured default seeds")
@click.pass_context
def crawl(ctx, urls, force, depth, max_pages, same_domain, use_default):
    """Crawl and index web pages using unified document processor."""
    try:
        doc_processor = get_doc_processor()
        
        if use_default:
            if not ctx.obj["quiet"]:
                console.print(f"🕷️ Crawling default seeds: {', '.join(DEFAULT_SEEDS)}", style="cyan")
            
            result = doc_processor.process_batch_urls(
                seed_urls=DEFAULT_SEEDS,
                depth=CRAWL_DEPTH,
                max_pages=CRAWL_MAX_PAGES,
                same_domain=CRAWL_SAME_DOMAIN,
                force=force
            )
            
        elif depth or max_pages or same_domain:
            if not urls:
                console.print("L URLs required for depth crawling", style="red")
                raise click.Abort()
            
            url_list = list(urls)
            depth = depth or 1
            max_pages = max_pages or 30
            
            if depth < 1 or depth > 3:
                console.print("L Depth must be between 1 and 3", style="red")
                raise click.Abort()
            if max_pages < 1 or max_pages > 100:
                console.print("L Max pages must be between 1 and 100", style="red")
                raise click.Abort()
            
            if not ctx.obj["quiet"]:
                console.print(f"🕷️ Deep crawling {len(url_list)} seeds (depth: {depth}, max: {max_pages})", style="cyan")
            
            result = doc_processor.process_batch_urls(
                seed_urls=url_list,
                depth=depth,
                max_pages=max_pages,
                same_domain=same_domain,
                force=force
            )
            
        elif urls:
            url_list = list(urls)
            if not ctx.obj["quiet"]:
                console.print(f"🕷️ Processing {len(url_list)} URLs", style="cyan")
            
            result = doc_processor.process_web_content(url_list, force=force)
            
        else:
            console.print("L Provide URLs, use --default, or specify crawl options", style="red")
            raise click.Abort()
        
        # Format results
        total_processed = len(result)
        total_chunks = sum(stats.get("chunks", 0) for stats in result.values())
        errors = sum(1 for stats in result.values() if stats.get("errors", 0) > 0)
        
        if not ctx.obj["quiet"]:
            console.print(f" Crawl completed: {total_processed} URLs, {total_chunks} chunks", style="green")
            if errors > 0:
                console.print(f"⚠️  {errors} URLs had errors", style="yellow")
        else:
            console.print(f"{total_processed},{total_chunks},{errors}")
    
    except Exception as e:
        console.print(f"❌ Crawl operation failed: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()


@cli.command()
@click.option("--output", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.pass_context
def status(ctx, output):
    """System health check using all engine services."""
    try:
        if not ctx.obj["quiet"]:
            console.print("🏥 Checking system health...", style="cyan")
        
        # Check all engine services
        rag_engine = get_rag_engine()
        doc_processor = get_doc_processor()
        query_service = get_query_service()
        embedding_service = get_embedding_service()
        
        # Gather status information
        status_data = {
            "rag_engine": rag_engine.get_metrics(),
            "document_processor": doc_processor.get_collection_stats(),
            "query_service": query_service.get_query_analytics(),
            "embedding_service": embedding_service.get_analytics(),
            "provider_validation": rag_engine.validate_providers()
        }
        
        # Determine overall health
        overall_health = "ok"
        if not status_data["document_processor"]["connected"]:
            overall_health = "database_error"
        elif not any(status_data["provider_validation"].values()):
            overall_health = "no_providers"
        
        if output == "json":
            import json
            status_data["overall_health"] = overall_health
            console.print(json.dumps(status_data, indent=2))
        else:
            if overall_health == "ok":
                console.print(" System status: OK", style="green")
            else:
                console.print(f"L System status: {overall_health.upper()}", style="red")
            
            if ctx.obj["verbose"]:
                from rich.table import Table
                table = Table(title="Engine Status")
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="green")
                
                table.add_row("RAG Engine", f" {status_data['rag_engine']['queries_processed']} queries processed")
                table.add_row("Document DB", " Connected" if status_data["document_processor"]["connected"] else "L Disconnected")
                table.add_row("Query Service", f" {status_data['query_service']['total_queries']} queries")
                table.add_row("Embedding Service", f" {status_data['embedding_service']['total_embeddings_generated']} embeddings")
                
                console.print(table)
        
        if overall_health != "ok":
            raise click.Abort()
    
    except Exception as e:
        console.print(f"L Health check failed: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()


@cli.command()
@click.option("--output", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.pass_context
def metrics(ctx, output):
    """Show comprehensive performance metrics from all engine services."""
    try:
        # Gather metrics from all services
        rag_engine = get_rag_engine()
        doc_processor = get_doc_processor()
        query_service = get_query_service()
        embedding_service = get_embedding_service()
        
        metrics_data = {
            "version": APP_VERSION,
            "rag_engine": rag_engine.get_metrics(),
            "document_processor": doc_processor.get_collection_stats(),
            "query_service": query_service.get_query_analytics(),
            "embedding_service": embedding_service.get_analytics()
        }
        
        if output == "json":
            import json
            console.print(json.dumps(metrics_data, indent=2))
        else:
            from rich.table import Table
            
            # Summary table
            table = Table(title="Performance Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            # RAG Engine metrics
            rag_metrics = metrics_data["rag_engine"]
            table.add_row("Queries Processed", str(rag_metrics.get("queries_processed", 0)))
            table.add_row("Successful Answers", str(rag_metrics.get("successful_answers", 0)))
            table.add_row("Average Confidence", str(rag_metrics.get("average_confidence", 0)))
            
            # Document metrics
            doc_metrics = metrics_data["document_processor"]
            table.add_row("Total Documents", str(doc_metrics.get("total_chunks", 0)))
            table.add_row("Files Processed", str(doc_metrics.get("processing_stats", {}).get("files_processed", 0)))
            
            # Embedding metrics
            embed_metrics = metrics_data["embedding_service"]
            table.add_row("Embeddings Generated", str(embed_metrics.get("total_embeddings_generated", 0)))
            table.add_row("Cache Hit Rate", f"{embed_metrics.get('cache_hit_rate', 0)*100:.1f}%")
            
            console.print(table)
    
    except Exception as e:
        console.print(f"L Failed to get metrics: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()


@cli.command()
@click.pass_context
def ping(ctx):
    """Test provider connectivity through RAG engine."""
    try:
        if not ctx.obj["quiet"]:
            console.print("🏓 Testing provider connectivity...", style="cyan")
        
        rag_engine = get_rag_engine()
        provider_status = rag_engine.validate_providers()
        
        working_providers = [name for name, status in provider_status.items() if status]
        
        if working_providers:
            if ctx.obj["quiet"]:
                console.print("ok")
            else:
                console.print(f" Providers available: {', '.join(working_providers)}", style="green")
        else:
            if ctx.obj["quiet"]:
                console.print("failed")
            else:
                console.print("L No providers available", style="red")
            raise click.Abort()
    
    except Exception as e:
        if ctx.obj["quiet"]:
            console.print("failed")
        else:
            console.print(f"L Provider test failed: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()


if __name__ == "__main__":
    cli()