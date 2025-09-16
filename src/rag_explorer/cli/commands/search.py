import click  
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rag_explorer.engine import UnifiedRAGEngine, UnifiedQueryService, UnifiedEmbeddingService

console = Console()

global _query_service, _embedding_service
if _query_service not in globals():
    _query_service = UnifiedQueryService()
if _embedding_service not in globals():
    _embedding_service = UnifiedEmbeddingService()

@click.command()
@click.argument('query')
def search(query):
    """Search vector database and show top results"""
    console.print(Panel(f"[blue]Searching for: {query}[/blue]", title="Vector Search"))

    try:

        processed_query = _query_service.preprocess_query(query)
        query_embedding = _embedding_service.generate_embeddings(processed_query)
        

        results = query_by_embedding(
            query_embedding=query_embedding,
            k=settings.SEARCH_MAX_RESULTS
        )

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
            score = 1 - result.get('distance', 0)  # Convert distance to similarity score
            meta = result.get('meta', {})
            source = meta.get('source') or meta.get('path') or 'Unknown'
            content = result.get('text', '')
            content = content[:100] + "..." if len(content) > 100 else content

            table.add_row(
                str(i),
                f"{score:.3f}",
                source,
                content
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]✗ Error searching: {str(e)}[/red]")
        sys.exit(1)
