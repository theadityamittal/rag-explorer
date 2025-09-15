"""Crawl commands for web content indexing using unified document processor."""

import click
from rich.console import Console

from ...utils.settings import (
    CRAWL_DEPTH,
    CRAWL_MAX_PAGES,
    CRAWL_SAME_DOMAIN,
    DEFAULT_SEEDS,
)

# Global variable for singleton document processor instance
_doc_processor = None


def get_doc_processor():
    """Get or create document processor instance."""
    from ...engine import UnifiedDocumentProcessor
    global _doc_processor
    if _doc_processor is None:
        _doc_processor = UnifiedDocumentProcessor()
    return _doc_processor


@click.command()
@click.argument("urls", nargs=-1, required=False)
@click.option("--force", is_flag=True, help="Force re-index even if content unchanged")
@click.option("--depth", type=int, help="Crawl depth (1-3)")
@click.option("--max-pages", type=int, help="Maximum pages to crawl (1-100)")
@click.option("--same-domain", is_flag=True, help="Restrict to same domain")
@click.option("--default", "use_default", is_flag=True, help="Use configured default seeds")
@click.pass_context
def crawl(ctx, urls, force, depth, max_pages, same_domain, use_default):
    """Crawl and index web pages using unified document processor."""
    console = Console()
    
    try:
        doc_processor = get_doc_processor()
        
        if use_default:
            if not ctx.obj["quiet"]:
                console.print(f"[CRAWLING] Using default seeds: {', '.join(DEFAULT_SEEDS)}", style="cyan")
            
            result = doc_processor.process_batch_urls(
                seed_urls=DEFAULT_SEEDS,
                depth=CRAWL_DEPTH,
                max_pages=CRAWL_MAX_PAGES,
                same_domain=CRAWL_SAME_DOMAIN,
                force=force
            )
            
        elif depth or max_pages or same_domain:
            if not urls:
                console.print("ERROR: URLs required for depth crawling", style="red")
                raise click.Abort()
            
            url_list = list(urls)
            depth = depth or 1
            max_pages = max_pages or 30
            
            if depth < 1 or depth > 3:
                console.print("ERROR: Depth must be between 1 and 3", style="red")
                raise click.Abort()
            if max_pages < 1 or max_pages > 100:
                console.print("ERROR: Max pages must be between 1 and 100", style="red")
                raise click.Abort()
            
            if not ctx.obj["quiet"]:
                console.print(f"[CRAWLING] Deep crawling {len(url_list)} seeds (depth: {depth}, max: {max_pages})", style="cyan")
            
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
                console.print(f"[CRAWLING] Processing {len(url_list)} URLs", style="cyan")
            
            result = doc_processor.process_web_content(url_list, force=force)
            
        else:
            console.print("ERROR: Provide URLs, use --default, or specify crawl options", style="red")
            raise click.Abort()
        
        # Format results
        total_processed = len(result)
        total_chunks = sum(stats.get("chunks", 0) for stats in result.values())
        errors = sum(1 for stats in result.values() if stats.get("errors", 0) > 0)
        
        if not ctx.obj["quiet"]:
            console.print(f"SUCCESS: Crawl completed: {total_processed} URLs, {total_chunks} chunks", style="green")
            if errors > 0:
                console.print(f"WARNING: {errors} URLs had errors", style="yellow")
        else:
            console.print(f"{total_processed},{total_chunks},{errors}")
    
    except Exception as e:
        console.print(f"ERROR: Crawl operation failed: {e}", style="red")
        if ctx.obj["verbose"]:
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()