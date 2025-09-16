import click  
import sys
from rich.console import Console
from rich.panel import Panel
from rag_explorer.engine import UnifiedDocumentProcessor
from rag_explorer.utils.settings import (DOCS_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DB_PATH)

console = Console()

global _doc_processor
if '_doc_processor' not in globals():
    _doc_processor = UnifiedDocumentProcessor()

@click.command()
@click.option('--reset', is_flag=True, help='Reset the vector database before indexing')
def index(reset):
    """Index documents from a directory"""
    docs_path = DOCS_FOLDER
    console.print(Panel(f"[blue]Indexing documents from: {docs_path}[/blue]", title="RAG Explorer"))

    try:
        if reset:
            console.print("[yellow]Resetting vector database...[/yellow]")

        result = _doc_processor.process_local_directory(
            directory=DOCS_FOLDER,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            reset_collection=reset
        )

        console.print(f"[green]✓ Successfully indexed {result['count']} document chunks[/green]")
        console.print(f"[dim]Files processed: {result.get('files_processed', 'N/A')}[/dim]")
        console.print(f"[dim]Storage: {CHROMA_DB_PATH}[/dim]")

    except Exception as e:
        console.print(f"[red]✗ Error indexing documents: {str(e)}[/red]")
        sys.exit(1)
