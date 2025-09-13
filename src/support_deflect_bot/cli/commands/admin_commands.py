"""Admin commands for health checks, metrics, and configuration management."""

import json
import click
from rich.console import Console
from rich.table import Table

from ...utils.settings import (
    ANSWER_MIN_CONF,
    CRAWL_DEPTH, 
    CRAWL_MAX_PAGES,
    DOCS_FOLDER,
    MAX_CHUNKS
)


def get_rag_engine():
    """Get or create RAG engine instance."""
    from ...engine import UnifiedRAGEngine
    global _rag_engine
    if "_rag_engine" not in globals() or _rag_engine is None:
        _rag_engine = UnifiedRAGEngine()
    return _rag_engine


def get_doc_processor():
    """Get or create document processor instance."""
    from ...engine import UnifiedDocumentProcessor
    global _doc_processor
    if "_doc_processor" not in globals() or _doc_processor is None:
        _doc_processor = UnifiedDocumentProcessor()
    return _doc_processor


def get_query_service():
    """Get or create query service instance."""
    from ...engine import UnifiedQueryService
    global _query_service
    if "_query_service" not in globals() or _query_service is None:
        _query_service = UnifiedQueryService()
    return _query_service


def get_embedding_service():
    """Get or create embedding service instance."""
    from ...engine import UnifiedEmbeddingService
    global _embedding_service
    if "_embedding_service" not in globals() or _embedding_service is None:
        _embedding_service = UnifiedEmbeddingService()
    return _embedding_service


@click.command()
@click.option("--output", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.pass_context
def status(ctx, output):
    """System health check using all engine services."""
    console = Console()
    
    try:
        if not ctx.obj["quiet"]:
            console.print("[STATUS] Checking system health...", style="cyan")
        
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
            status_data["overall_health"] = overall_health
            console.print(json.dumps(status_data, indent=2))
        else:
            if overall_health == "ok":
                console.print("SUCCESS: System status: OK", style="green")
            else:
                console.print(f"ERROR: System status: {overall_health.upper()}", style="red")
            
            if ctx.obj["verbose"]:
                table = Table(title="Engine Status")
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="green")
                
                table.add_row("RAG Engine", f"OK: {status_data['rag_engine']['queries_processed']} queries processed")
                table.add_row("Document DB", "Connected" if status_data["document_processor"]["connected"] else "Disconnected")
                table.add_row("Query Service", f"OK: {status_data['query_service']['total_queries']} queries")
                table.add_row("Embedding Service", f"OK: {status_data['embedding_service']['total_embeddings_generated']} embeddings")
                
                console.print(table)
        
        if overall_health != "ok":
            raise click.Abort()
    
    except Exception as e:
        console.print(f"ERROR: Health check failed: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()


@click.command()
@click.option("--output", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.pass_context
def metrics(ctx, output):
    """Show comprehensive performance metrics from all engine services."""
    console = Console()
    
    try:
        # Gather metrics from all services
        rag_engine = get_rag_engine()
        doc_processor = get_doc_processor()
        query_service = get_query_service()
        embedding_service = get_embedding_service()
        
        metrics_data = {
            "version": "2.0.0",  # APP_VERSION
            "rag_engine": rag_engine.get_metrics(),
            "document_processor": doc_processor.get_collection_stats(),
            "query_service": query_service.get_query_analytics(),
            "embedding_service": embedding_service.get_analytics()
        }
        
        if output == "json":
            console.print(json.dumps(metrics_data, indent=2))
        else:
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
        console.print(f"ERROR: Failed to get metrics: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()


@click.command()
@click.pass_context
def ping(ctx):
    """Test provider connectivity through RAG engine."""
    console = Console()
    
    try:
        if not ctx.obj["quiet"]:
            console.print("[PING] Testing provider connectivity...", style="cyan")
        
        rag_engine = get_rag_engine()
        provider_status = rag_engine.validate_providers()
        
        working_providers = [name for name, status in provider_status.items() if status]
        
        if working_providers:
            if ctx.obj["quiet"]:
                console.print("ok")
            else:
                console.print(f"SUCCESS: Providers available: {', '.join(working_providers)}", style="green")
        else:
            if ctx.obj["quiet"]:
                console.print("failed")
            else:
                console.print("ERROR: No providers available", style="red")
            raise click.Abort()
    
    except Exception as e:
        if ctx.obj["quiet"]:
            console.print("failed")
        else:
            console.print(f"ERROR: Provider test failed: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()


@click.command()
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--validate", is_flag=True, help="Validate configuration settings")
@click.pass_context
def config(ctx, show, validate):
    """Configuration management for Support Deflect Bot."""
    console = Console()
    
    try:
        from ...config import ConfigurationManager
        
        config_manager = ConfigurationManager()
        
        if show:
            if not ctx.obj["quiet"]:
                console.print("[CONFIG] Current Configuration", style="cyan")
            
            config_table = Table(title="Configuration Settings")
            config_table.add_column("Setting", style="cyan")
            config_table.add_column("Value", style="green")
            config_table.add_column("Source", style="yellow")
            
            # Add key configuration values
            config_items = [
                ("ANSWER_MIN_CONF", ANSWER_MIN_CONF, "settings"),
                ("MAX_CHUNKS", MAX_CHUNKS, "settings"),
                ("CRAWL_DEPTH", CRAWL_DEPTH, "settings"),
                ("CRAWL_MAX_PAGES", CRAWL_MAX_PAGES, "settings"),
                ("DOCS_FOLDER", DOCS_FOLDER, "settings"),
                ("CHROMA_DB_PATH", getattr(config_manager, 'chroma_db_path', 'default'), "config"),
            ]
            
            for name, value, source in config_items:
                config_table.add_row(name, str(value), source)
            
            console.print(config_table)
            
        elif validate:
            if not ctx.obj["quiet"]:
                console.print("[CONFIG] Validating configuration...", style="cyan")
            
            # Validate provider configurations
            rag_engine = get_rag_engine()
            provider_status = rag_engine.validate_providers()
            
            validation_table = Table(title="Configuration Validation")
            validation_table.add_column("Component", style="cyan")
            validation_table.add_column("Status", style="green")
            validation_table.add_column("Details")
            
            # Provider validation
            for provider_name, is_valid in provider_status.items():
                status = "OK: Valid" if is_valid else "ERROR: Invalid"
                details = "Configuration OK" if is_valid else "Check API keys/settings"
                validation_table.add_row(f"Provider: {provider_name}", status, details)
            
            # Database validation
            try:
                doc_processor = get_doc_processor()
                db_stats = doc_processor.get_collection_stats()
                db_status = "OK: Valid" if db_stats.get("connected", False) else "ERROR: Invalid"
                db_details = f"Connected, {db_stats.get('total_chunks', 0)} documents" if db_stats.get("connected") else "Connection failed"
                validation_table.add_row("Vector Database", db_status, db_details)
            except Exception as e:
                validation_table.add_row("Vector Database", "ERROR: Invalid", str(e))
            
            console.print(validation_table)
            
        else:
            console.print("[CONFIG] Configuration Management", style="cyan")
            console.print("Use --show to display current settings")
            console.print("Use --validate to check configuration validity")
            console.print("Environment variables and .env files are used for configuration")
            
    except Exception as e:
        console.print(f"ERROR: Configuration command failed: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()