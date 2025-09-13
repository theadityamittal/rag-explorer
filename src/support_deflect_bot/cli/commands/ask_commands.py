"""Ask command for interactive Q&A sessions using unified RAG engine."""

import click
from rich.console import Console

from ..ask_session import UnifiedAskSession
from ...utils.settings import MAX_CHUNKS


def get_rag_engine():
    """Get or create RAG engine instance."""
    from ...engine import UnifiedRAGEngine
    global _rag_engine
    if "_rag_engine" not in globals() or _rag_engine is None:
        _rag_engine = UnifiedRAGEngine()
    return _rag_engine


def get_query_service():
    """Get or create query service instance."""
    from ...engine import UnifiedQueryService
    global _query_service
    if "_query_service" not in globals() or _query_service is None:
        _query_service = UnifiedQueryService()
    return _query_service


@click.command()
@click.option("--domains", help="Comma-separated list of domains to filter")
@click.option("--confidence", type=float, help="Override confidence threshold (0.0-1.0)")
@click.option("--max-chunks", type=int, help="Override max chunks to retrieve")
@click.pass_context
def ask(ctx, domains, confidence, max_chunks):
    """Start interactive Q&A session using unified RAG engine."""
    console = Console()
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
        console.print("\n>> Session interrupted. Goodbye!", style="yellow")
    except Exception as e:
        console.print(f"ERROR: Session failed: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()