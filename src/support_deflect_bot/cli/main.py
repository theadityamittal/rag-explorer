"""Main CLI entry point for Support Deflect Bot."""

# Initialize clean CLI environment FIRST (suppress warnings)  
from ..compat import init_clean_cli
init_clean_cli()

import time
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table

# Use compatibility bridges for core functionality
from ..compat import llm_echo, answer_question, retrieve, ingest_folder, crawl_urls, index_urls, Meter

# Use new settings from support_deflect_bot structure
from ..utils.settings import (
    APP_NAME,
    APP_VERSION,
    DOCS_FOLDER,
    ANSWER_MIN_CONF,
    MAX_CHUNKS,
    MAX_CHARS_PER_CHUNK,
    CRAWL_DEPTH,
    CRAWL_MAX_PAGES,
    CRAWL_SAME_DOMAIN,
    DEFAULT_SEEDS,
)

from .ask_session import start_interactive_session
from .output import format_search_results, format_answer, format_metrics

console = Console()

# Global meters for tracking performance
ASK_METER = Meter()
SEARCH_METER = Meter()


@click.group()
@click.version_option(version=APP_VERSION, prog_name=APP_NAME)
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Minimal output')
@click.pass_context
def cli(ctx, verbose, quiet):
    """Support Deflect Bot - Intelligent document Q&A with confidence-based refusal."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet


@cli.command()
@click.pass_context
def index(ctx):
    """Index local documentation from ./docs folder."""
    try:
        if not ctx.obj['quiet']:
            console.print("🔄 Indexing local documentation...", style="cyan")
        
        n = ingest_folder(DOCS_FOLDER)
        
        if ctx.obj['quiet']:
            console.print(f"{n}")
        else:
            console.print(f"✅ Successfully indexed {n} chunks from {DOCS_FOLDER}", style="green")
            
    except ConnectionError:
        console.print("❌ Database connection failed", style="red")
        raise click.Abort()
    except FileNotFoundError:
        console.print(f"❌ Documentation folder '{DOCS_FOLDER}' not found", style="red")
        raise click.Abort()
    except Exception as e:
        console.print(f"❌ Indexing failed: {e}", style="red")
        raise click.Abort()


@cli.command()
@click.argument('query')
@click.option('--limit', '-l', default=5, help='Number of results to return (1-20)')
@click.option('--output', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def search(ctx, query, limit, output):
    """Search through indexed documents."""
    if limit < 1 or limit > 20:
        console.print("❌ Limit must be between 1 and 20", style="red")
        raise click.Abort()
        
    t0 = time.perf_counter()
    
    try:
        if not ctx.obj['quiet']:
            console.print(f"🔍 Searching for: '{query}'", style="cyan")
            
        hits = retrieve(query, k=limit)
        
        if output == 'json':
            import json
            results = [
                {
                    "text": h["text"][:400],
                    "path": h["meta"].get("path"),
                    "chunk_id": h["meta"].get("chunk_id"),
                    "distance": h.get("distance"),
                }
                for h in hits
            ]
            console.print(json.dumps({"query": query, "results": results}, indent=2))
        else:
            format_search_results(console, query, hits, ctx.obj['quiet'])
            
    except ConnectionError:
        console.print("❌ Database connection failed", style="red")
        raise click.Abort()
    except Exception as e:
        console.print(f"❌ Search failed: {e}", style="red")
        raise click.Abort()
    finally:
        SEARCH_METER.observe(time.perf_counter() - t0)


@cli.command()
@click.option('--domains', help='Comma-separated list of domains to filter')
@click.option('--confidence', type=float, help='Override confidence threshold')
@click.pass_context
def ask(ctx, domains, confidence):
    """Start interactive Q&A session (exit with 'end')."""
    domain_list = [d.strip() for d in domains.split(',')] if domains else None
    
    try:
        start_interactive_session(console, domain_list, ASK_METER, ctx.obj)
    except KeyboardInterrupt:
        console.print("\n👋 Session interrupted. Goodbye!", style="yellow")
    except Exception as e:
        console.print(f"❌ Session failed: {e}", style="red")
        raise click.Abort()


@cli.command()
@click.argument('urls', nargs=-1, required=False)
@click.option('--force', is_flag=True, help='Force re-index even if content unchanged')
@click.option('--depth', type=int, help='Crawl depth (1-3)')
@click.option('--max-pages', type=int, help='Maximum pages to crawl (1-100)')
@click.option('--same-domain', is_flag=True, help='Restrict to same domain')
@click.option('--default', 'use_default', is_flag=True, help='Use configured default seeds')
@click.pass_context
def crawl(ctx, urls, force, depth, max_pages, same_domain, use_default):
    """Crawl and index web pages with various options."""
    try:
        if use_default:
            # Use default crawl settings
            if not ctx.obj['quiet']:
                console.print(f"🕷️ Crawling default seeds: {', '.join(DEFAULT_SEEDS)}", style="cyan")
            
            result = crawl_urls(
                seeds=DEFAULT_SEEDS,
                depth=CRAWL_DEPTH,
                max_pages=CRAWL_MAX_PAGES,
                same_domain=CRAWL_SAME_DOMAIN,
                force=force,
            )
            
        elif depth or max_pages or same_domain:
            # Depth crawling mode
            if not urls:
                console.print("❌ URLs required for depth crawling", style="red")
                raise click.Abort()
                
            url_list = list(urls)
            depth = depth or 1
            max_pages = max_pages or 30
            
            if depth < 1 or depth > 3:
                console.print("❌ Depth must be between 1 and 3", style="red")
                raise click.Abort()
            if max_pages < 1 or max_pages > 100:
                console.print("❌ Max pages must be between 1 and 100", style="red")
                raise click.Abort()
                
            if not ctx.obj['quiet']:
                console.print(f"🕷️ Deep crawling {len(url_list)} seeds (depth: {depth}, max: {max_pages})", style="cyan")
                
            result = crawl_urls(
                seeds=url_list,
                depth=depth,
                max_pages=max_pages,
                same_domain=same_domain,
                force=force,
            )
            
        elif urls:
            # Simple URL crawling
            url_list = list(urls)
            if not ctx.obj['quiet']:
                console.print(f"🕷️ Crawling {len(url_list)} URLs", style="cyan")
                
            result = index_urls(url_list, force=force)
            
        else:
            console.print("❌ Provide URLs, use --default, or specify crawl options", style="red")
            raise click.Abort()
            
        if not ctx.obj['quiet']:
            console.print(f"✅ Crawl completed: {result}", style="green")
        else:
            console.print(str(result))
            
    except ConnectionError:
        console.print("❌ Database connection failed", style="red")
        raise click.Abort()
    except Exception as e:
        console.print(f"❌ Crawl operation failed: {e}", style="red")
        raise click.Abort()


@cli.command()
@click.pass_context
def status(ctx):
    """System health check."""
    try:
        # Check basic health
        health_ok = True
        
        if not ctx.obj['quiet']:
            console.print("🏥 Checking system health...", style="cyan")
        
        if health_ok:
            if ctx.obj['quiet']:
                console.print("ok")
            else:
                console.print("✅ System status: OK", style="green")
        else:
            console.print("❌ System status: FAILED", style="red")
            raise click.Abort()
            
    except Exception as e:
        console.print(f"❌ Health check failed: {e}", style="red")
        raise click.Abort()


@cli.command()
@click.pass_context
def ping(ctx):
    """Test LLM connectivity."""
    try:
        if not ctx.obj['quiet']:
            console.print("🏓 Pinging LLM service...", style="cyan")
            
        text = llm_echo("Wake up!")
        
        if ctx.obj['quiet']:
            console.print("ok")
        else:
            console.print(f"✅ LLM responded: {text}", style="green")
            
    except Exception as e:
        if ctx.obj['quiet']:
            console.print("failed")
        else:
            console.print(f"❌ LLM service unavailable: {e}", style="red")
        raise click.Abort()


@cli.command()
@click.option('--output', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def metrics(ctx, output):
    """Show performance metrics."""
    try:
        metrics_data = {
            "ask": ASK_METER.summary(),
            "search": SEARCH_METER.summary(),
            "version": APP_VERSION,
        }
        
        if output == 'json':
            import json
            console.print(json.dumps(metrics_data, indent=2))
        else:
            format_metrics(console, metrics_data)
            
    except Exception as e:
        console.print(f"❌ Failed to get metrics: {e}", style="red")
        raise click.Abort()


@cli.command()
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--set', 'set_config', help='Set configuration KEY=VALUE')
@click.pass_context
def config(ctx, show, set_config):
    """Show or modify configuration."""
    if set_config:
        console.print("❌ Configuration modification not yet implemented", style="red")
        return
        
    # Show current configuration
    config_data = {
        "APP_NAME": APP_NAME,
        "APP_VERSION": APP_VERSION,
        "DOCS_FOLDER": DOCS_FOLDER,
        "ANSWER_MIN_CONF": ANSWER_MIN_CONF,
        "MAX_CHUNKS": MAX_CHUNKS,
        "MAX_CHARS_PER_CHUNK": MAX_CHARS_PER_CHUNK,
        "CRAWL_DEPTH": CRAWL_DEPTH,
        "CRAWL_MAX_PAGES": CRAWL_MAX_PAGES,
        "CRAWL_SAME_DOMAIN": CRAWL_SAME_DOMAIN,
    }
    
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in config_data.items():
        table.add_row(key, str(value))
    
    console.print(table)


if __name__ == '__main__':
    cli()