import click
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rag_explorer.engine import UnifiedRAGEngine

console = Console()

global _rag_engine
if '_rag_engine' not in globals():
    _rag_engine = UnifiedRAGEngine()

@click.command()
@click.argument('query')
@click.option('--count', '-c', default=5, help='Number of results to return')
def search(query, count):
    """Search vector database and show top results"""
    console.print(Panel(f"[blue]Searching for: {query}[/blue]", title="Vector Search"))

    try:
        # Use UnifiedRAGEngine to search documents
        results = _rag_engine.search_documents(query=query, count=count)

        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        # Create results table
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Score", style="green", width=8)
        table.add_column("Source", style="yellow", width=20)
        table.add_column("Content", style="white")

        for i, result in enumerate(results, 1):
            score = result.get('similarity_score', 0.0)
            metadata = result.get('metadata', {})
            source = metadata.get('source') or metadata.get('path') or 'Unknown'
            content = result.get('text', '')
            content = content[:100] + "..." if len(content) > 100 else content

            table.add_row(
                str(i),
                f"{score:.3f}",
                source,
                content
            )

        console.print(table)
        console.print(f"[dim]Found {len(results)} results[/dim]")

    except Exception as e:
        console.print(f"[red]✗ Error searching: {str(e)}[/red]")
        sys.exit(1)
