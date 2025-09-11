"""Output formatting utilities for the unified CLI."""

from typing import Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def format_search_results(console: Console, query: str, results: List[Dict], quiet: bool = False):
    """Format search results for display."""
    if quiet:
        # Quiet mode: just show count
        console.print(len(results))
        return
    
    if not results:
        console.print("No results found.", style="yellow")
        return
    
    # Create results table
    table = Table(title=f"Search Results for: '{query}'")
    table.add_column("Rank", style="cyan", width=4)
    table.add_column("Source", style="blue", width=40)
    table.add_column("Relevance", style="green", width=8)
    table.add_column("Preview", style="white", width=60)
    
    for i, result in enumerate(results, 1):
        path = result.get("meta", {}).get("path", "unknown")
        relevance_score = result.get("relevance_score")
        distance = result.get("distance")
        
        # Format relevance score
        if relevance_score is not None:
            relevance = f"{relevance_score:.3f}"
        elif distance is not None:
            # Convert distance to similarity percentage
            similarity = 1.0 / (1.0 + max(0.0, distance))
            relevance = f"{similarity:.3f}"
        else:
            relevance = "N/A"
        
        # Format preview text
        preview = result.get("text", "")[:150]
        if len(result.get("text", "")) > 150:
            preview += "..."
        
        # Truncate path for display
        if len(path) > 37:
            display_path = "..." + path[-34:]
        else:
            display_path = path
        
        table.add_row(
            str(i),
            display_path,
            relevance,
            preview
        )
    
    console.print(table)


def format_answer(console: Console, response: Dict, verbose: bool = False):
    """Format RAG answer response for display."""
    answer = response.get("answer", "")
    confidence = response.get("confidence", 0.0)
    citations = response.get("citations", [])
    metadata = response.get("metadata", {})
    
    # Main answer panel
    confidence_color = "green" if confidence >= 0.7 else "yellow" if confidence >= 0.4 else "red"
    title = f"Answer (Confidence: {confidence:.3f})"
    
    answer_panel = Panel(
        answer,
        title=title,
        title_align="left",
        border_style=confidence_color
    )
    console.print(answer_panel)
    
    # Citations if available
    if citations and verbose:
        console.print("\n📚 Sources:", style="cyan bold")
        for i, citation in enumerate(citations, 1):
            path = citation.get("path", "unknown")
            preview = citation.get("preview", "")
            
            citation_text = Text()
            citation_text.append(f"[{i}] ", style="cyan bold")
            citation_text.append(f"{path}\n", style="blue")
            citation_text.append(f"    {preview}", style="dim")
            
            console.print(citation_text)
    
    # Metadata if verbose
    if verbose and metadata:
        console.print(f"\n📋 Metadata:", style="dim")
        for key, value in metadata.items():
            console.print(f"  {key}: {value}", style="dim")


def format_metrics_table(console: Console, metrics_data: Dict[str, Any]):
    """Format comprehensive metrics in a table."""
    table = Table(title="Performance Metrics")
    table.add_column("Component", style="cyan", width=20)
    table.add_column("Metric", style="blue", width=25)
    table.add_column("Value", style="green", width=15)
    
    # RAG Engine metrics
    if "rag_engine" in metrics_data:
        rag = metrics_data["rag_engine"]
        table.add_row("RAG Engine", "Queries Processed", str(rag.get("queries_processed", 0)))
        table.add_row("", "Successful Answers", str(rag.get("successful_answers", 0)))
        table.add_row("", "Average Confidence", f"{rag.get('average_confidence', 0):.3f}")
        table.add_row("", "Refusals", str(rag.get("refusals", 0)))
    
    # Document Processor metrics
    if "document_processor" in metrics_data:
        doc = metrics_data["document_processor"]
        table.add_row("Documents", "Total Chunks", str(doc.get("total_chunks", 0)))
        table.add_row("", "Connected", "Yes" if doc.get("connected") else "No")
        processing_stats = doc.get("processing_stats", {})
        table.add_row("", "Files Processed", str(processing_stats.get("files_processed", 0)))
    
    # Query Service metrics
    if "query_service" in metrics_data:
        query = metrics_data["query_service"]
        table.add_row("Query Service", "Total Queries", str(query.get("total_queries", 0)))
        table.add_row("", "Success Rate", f"{query.get('success_rate', 0)*100:.1f}%")
    
    # Embedding Service metrics
    if "embedding_service" in metrics_data:
        embed = metrics_data["embedding_service"]
        table.add_row("Embeddings", "Generated", str(embed.get("total_embeddings_generated", 0)))
        table.add_row("", "Cache Hit Rate", f"{embed.get('cache_hit_rate', 0)*100:.1f}%")
        table.add_row("", "Avg Time/Embed", f"{embed.get('average_time_per_embedding', 0):.4f}s")
    
    console.print(table)


def format_status_summary(console: Console, status_data: Dict[str, Any], verbose: bool = False):
    """Format system status summary."""
    overall_health = status_data.get("overall_health", "unknown")
    
    if overall_health == "ok":
        console.print("✅ System Status: [green]HEALTHY[/green]")
    elif overall_health == "database_error":
        console.print("❌ System Status: [red]DATABASE ERROR[/red]")
    elif overall_health == "no_providers":
        console.print("⚠️ System Status: [yellow]NO PROVIDERS[/yellow]")
    else:
        console.print(f"❓ System Status: [dim]{overall_health.upper()}[/dim]")
    
    if verbose:
        # Provider status
        providers = status_data.get("provider_validation", {})
        if providers:
            console.print("\n🔧 Provider Status:", style="cyan")
            for name, available in providers.items():
                status_icon = "✅" if available else "❌"
                console.print(f"  {status_icon} {name}")
        
        # Database info
        doc_status = status_data.get("document_processor", {})
        if doc_status.get("connected"):
            total_docs = doc_status.get("total_chunks", 0)
            console.print(f"\n📊 Database: Connected ({total_docs} documents)", style="green")
        else:
            console.print(f"\n📊 Database: Disconnected", style="red")


def format_crawl_results(console: Console, results: Dict[str, Dict], quiet: bool = False):
    """Format web crawling results."""
    if quiet:
        total_processed = len(results)
        total_chunks = sum(stats.get("chunks", 0) for stats in results.values())
        errors = sum(1 for stats in results.values() if stats.get("errors", 0) > 0)
        console.print(f"{total_processed},{total_chunks},{errors}")
        return
    
    # Create detailed results table
    table = Table(title="Crawl Results")
    table.add_column("URL", style="blue", width=50)
    table.add_column("Status", style="green", width=12)
    table.add_column("Chunks", style="cyan", width=8)
    table.add_column("Notes", style="yellow", width=30)
    
    for url, stats in results.items():
        # Determine status
        if stats.get("errors", 0) > 0:
            status = "❌ Error"
            status_style = "red"
        elif stats.get("robots_blocked", 0) > 0:
            status = "🚫 Blocked"
            status_style = "yellow"
        elif stats.get("skipped_304", 0) > 0:
            status = "💾 Cached"
            status_style = "blue"
        elif stats.get("skipped_samehash", 0) > 0:
            status = "↔️ Same"
            status_style = "blue"
        elif stats.get("fetched", 0) > 0:
            status = "✅ Success"
            status_style = "green"
        else:
            status = "❓ Unknown"
            status_style = "dim"
        
        chunks = str(stats.get("chunks", 0))
        
        # Create notes
        notes = []
        if stats.get("replaced", 0) > 0:
            notes.append("updated")
        if stats.get("robots_blocked", 0) > 0:
            notes.append("robots.txt")
        notes_text = ", ".join(notes) if notes else ""
        
        # Truncate URL for display
        display_url = url if len(url) <= 47 else url[:44] + "..."
        
        table.add_row(display_url, status, chunks, notes_text)
    
    console.print(table)
    
    # Summary
    total_processed = len(results)
    total_chunks = sum(stats.get("chunks", 0) for stats in results.values())
    successful = sum(1 for stats in results.values() if stats.get("fetched", 0) > 0)
    errors = sum(1 for stats in results.values() if stats.get("errors", 0) > 0)
    
    summary_text = f"📋 Summary: {total_processed} URLs processed, {successful} successful, {total_chunks} chunks created"
    if errors > 0:
        summary_text += f", {errors} errors"
    
    console.print(f"\n{summary_text}", style="cyan")


def format_provider_validation(console: Console, validation_results: Dict[str, Dict]):
    """Format provider validation results."""
    if not validation_results:
        console.print("No providers to validate.", style="yellow")
        return
    
    table = Table(title="Provider Validation")
    table.add_column("Provider", style="cyan", width=20)
    table.add_column("Status", style="green", width=12)
    table.add_column("Dimension", style="blue", width=10)
    table.add_column("Response Time", style="yellow", width=15)
    table.add_column("Error", style="red", width=30)
    
    for provider_name, result in validation_results.items():
        status = "✅ Available" if result.get("available") else "❌ Failed"
        dimension = str(result.get("dimension", "N/A"))
        response_time = f"{result.get('response_time', 0):.3f}s" if result.get("response_time") else "N/A"
        error = result.get("error", "")[:27] + "..." if len(result.get("error", "")) > 30 else result.get("error", "")
        
        table.add_row(provider_name, status, dimension, response_time, error)
    
    console.print(table)