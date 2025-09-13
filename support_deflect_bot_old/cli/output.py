"""Rich output formatting utilities."""

import json
from typing import Any, Dict, List

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def format_search_results(
    console: Console, query: str, hits: List[Dict], quiet: bool = False
):
    """Format and display search results."""
    if not hits:
        if not quiet:
            console.print("❌ No results found", style="yellow")
        return

    if quiet:
        for hit in hits:
            console.print(
                f"{hit['meta'].get('path', 'unknown')}:{hit['meta'].get('chunk_id', 0)}"
            )
        return

    # Create results table
    table = Table(title=f"Search Results for: '{query}'")
    table.add_column("Rank", width=4, style="dim")
    table.add_column("Source", style="cyan")
    table.add_column("Distance", width=8, style="yellow")
    table.add_column("Preview", style="white")

    for i, hit in enumerate(hits, 1):
        path = hit["meta"].get("path", "unknown")
        chunk_id = hit["meta"].get("chunk_id", 0)
        distance = f"{hit.get('distance', 0):.3f}"
        preview = hit["text"][:150] + ("..." if len(hit["text"]) > 150 else "")

        source = f"{path}:{chunk_id}"
        table.add_row(str(i), source, distance, preview)

    console.print(table)


def format_answer(console: Console, result: Dict, quiet: bool = False):
    """Format and display a Q&A answer with citations."""
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    confidence = result.get("confidence", 0.0)

    if quiet:
        console.print(answer)
        return

    # Display the answer
    console.print("🤖 [bold cyan]Bot:[/bold cyan]", end=" ")

    # Check if it's a refusal
    if "I don't have enough information" in answer:
        console.print(answer, style="yellow")
    else:
        console.print(answer, style="white")

    # Display citations if available
    if citations:
        console.print("\n📚 [dim]Sources:[/dim]")
        for i, citation in enumerate(citations, 1):
            path = citation.get("path", "unknown")
            chunk_id = citation.get("chunk_id", 0)
            rank = citation.get("rank", i)
            preview = citation.get("preview", "")[:100]

            source_text = f"  [{rank}] {path}:{chunk_id}"
            if preview:
                source_text += f" - {preview}..."

            console.print(source_text, style="dim blue")

    # Display confidence if available
    if confidence > 0:
        conf_style = (
            "green" if confidence >= 0.5 else "yellow" if confidence >= 0.3 else "red"
        )
        console.print(
            f"\n🎯 [dim]Confidence: [/dim][{conf_style}]{confidence:.2f}[/{conf_style}]"
        )


def format_metrics(console: Console, metrics: Dict[str, Any]):
    """Format and display performance metrics."""
    version = metrics.get("version", "unknown")

    console.print(f"\n📊 Performance Metrics (v{version})")

    # Create metrics table
    table = Table()
    table.add_column("Operation", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("P50 (ms)", style="yellow")
    table.add_column("P95 (ms)", style="yellow")

    for operation, data in metrics.items():
        if operation == "version":
            continue

        if isinstance(data, dict):
            count = data.get("count", 0)
            p50 = data.get("p50_ms", 0.0)
            p95 = data.get("p95_ms", 0.0)

            table.add_row(operation.title(), str(count), f"{p50:.1f}", f"{p95:.1f}")

    console.print(table)


def format_config(console: Console, config: Dict[str, Any]):
    """Format and display configuration."""
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    for key, value in config.items():
        table.add_row(key, str(value))

    console.print(table)
