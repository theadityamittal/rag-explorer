"""Index command for document processing using unified document processor."""

import click
from rich.console import Console

from ...utils.settings import DOCS_FOLDER


def get_doc_processor():
    """Get or create document processor instance."""
    from ...engine import UnifiedDocumentProcessor
    global _doc_processor
    if "_doc_processor" not in globals() or _doc_processor is None:
        _doc_processor = UnifiedDocumentProcessor()
    return _doc_processor


@click.command()
@click.option("--docs-path", help="Override documentation folder path")
@click.option("--reset", is_flag=True, help="Reset collection before indexing")
@click.pass_context
def index(ctx, docs_path, reset):
    """Index local documentation using unified document processor."""
    console = Console()
    
    try:
        folder_path = docs_path or DOCS_FOLDER
        
        if not ctx.obj["quiet"]:
            console.print(f"[INDEXING] Processing local documentation from {folder_path}...", style="cyan")
        
        doc_processor = get_doc_processor()
        result = doc_processor.process_local_directory(
            directory=folder_path,
            reset_collection=reset
        )
        
        if ctx.obj["quiet"]:
            console.print(f"{result['chunks_created']}")
        else:
            console.print(
                f"SUCCESS: Indexed {result['files_processed']} files "
                f"({result['chunks_created']} chunks) from {folder_path}", 
                style="green"
            )
            
            if result.get('errors', 0) > 0:
                console.print(f"WARNING: {result['errors']} files had errors", style="yellow")
    
    except Exception as e:
        console.print(f"ERROR: Indexing failed: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()