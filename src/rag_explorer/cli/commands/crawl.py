import click  
import sys
from rich.console import Console
from rich.panel import Panel

console = Console()

@click.command()
def crawl():
    """Crawl websites from CRAWL_SOURCES environment variable"""
    if not settings.CRAWL_SOURCES:
        console.print("[red]No crawl sources configured. Set CRAWL_SOURCES environment variable.[/red]")
        sys.exit(1)

    console.print(Panel(f"[blue]Crawling {len(settings.CRAWL_SOURCES)} sources[/blue]", title="Web Crawler"))

    try:
        from rag_explorer.data.web_ingest import crawl_urls

        # Filter out empty sources
        sources = [source.strip() for source in settings.CRAWL_SOURCES if source.strip()]

        if not sources:
            console.print("[red]No valid crawl sources found.[/red]")
            sys.exit(1)

        console.print(f"[dim]Starting crawl with depth={settings.CRAWL_DEPTH}, max_pages={settings.CRAWL_MAX_PAGES}[/dim]")

        result = crawl_urls(
            seeds=sources,
            depth=settings.CRAWL_DEPTH,
            max_pages=settings.CRAWL_MAX_PAGES,
            same_domain=settings.CRAWL_SAME_DOMAIN
        )

        total_pages = 0
        for source, stats in result.items():
            pages_crawled = stats.get('indexed', 0)
            total_pages += pages_crawled
            console.print(f"[green]✓ Crawled {pages_crawled} pages from {source}[/green]")

        console.print(f"[green]✓ Total pages crawled and indexed: {total_pages}[/green]")
        if total_pages > 0:
            console.print("[dim]Pages have been automatically indexed to the vector database[/dim]")

    except Exception as e:
        console.print(f"[red]✗ Error crawling: {str(e)}[/red]")
        sys.exit(1)
