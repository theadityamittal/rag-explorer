"""Search command for document retrieval using unified query service."""

import json
import click
from rich.console import Console

from ..output import format_search_results

# Global variable for singleton query service instance
_query_service = None


def get_query_service():
    """Get or create query service instance."""
    from ...engine import UnifiedQueryService
    global _query_service
    if _query_service is None:
        _query_service = UnifiedQueryService()
    return _query_service


@click.command()
@click.argument("query")
@click.option("--limit", "-l", default=5, help="Number of results to return (1-20)")
@click.option("--domains", help="Comma-separated list of domains to filter")
@click.option("--output", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.pass_context
def search(ctx, query, limit, domains, output):
    """Search through indexed documents using unified query service."""
    console = Console()
    
    if limit < 1 or limit > 20:
        console.print("ERROR: Limit must be between 1 and 20", style="red")
        raise click.Abort()
    
    domain_list = [d.strip() for d in domains.split(",")] if domains else None
    
    try:
        if not ctx.obj["quiet"]:
            console.print(f"[SEARCHING] Looking for: '{query}'", style="cyan")
        
        query_service = get_query_service()
        
        # Preprocess query
        processed_query = query_service.preprocess_query(query)
        if not processed_query["valid"]:
            console.print(f"ERROR: Invalid query: {processed_query['reason']}", style="red")
            raise click.Abort()
        
        # Retrieve documents
        results = query_service.retrieve_documents(
            processed_query,
            k=limit,
            domains=domain_list
        )
        
        if output == "json":
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
        console.print(f"ERROR: Search failed: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()