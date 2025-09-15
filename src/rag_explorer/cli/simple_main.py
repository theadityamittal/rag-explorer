"""Simple CLI interface for RAG Explorer."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any

from rag_explorer.engine.simple_rag_engine import SimpleRAGEngine
from rag_explorer.utils import simple_settings
from rag_explorer.data.store import get_client, get_collection, query_by_embedding, reset_collection, return_client

console = Console()

@click.group()
@click.version_option()
def cli():
    """RAG Explorer - Simple document Q&A tool for personal use"""
    pass

@cli.command()
@click.option('--reset', is_flag=True, help='Reset the vector database before indexing')
def index(reset):
    """Index documents from a directory"""
    docs_path = simple_settings.DOCS_FOLDER
    console.print(Panel(f"[blue]Indexing documents from: {docs_path}[/blue]", title="RAG Explorer"))

    try:
        engine = SimpleRAGEngine()

        if reset:
            console.print("[yellow]Resetting vector database...[/yellow]")
            reset_collection()

        result = engine.index_documents(
            docs_path,
            chunk_size=simple_settings.CHUNK_SIZE,
            chunk_overlap=simple_settings.CHUNK_OVERLAP
        )

        console.print(f"[green]✓ Successfully indexed {result['count']} document chunks[/green]")
        console.print(f"[dim]Files processed: {result.get('files_processed', 'N/A')}[/dim]")
        console.print(f"[dim]Storage: {simple_settings.CHROMA_DB_PATH}[/dim]")

    except Exception as e:
        console.print(f"[red]✗ Error indexing documents: {str(e)}[/red]")
        sys.exit(1)

@cli.command()
@click.argument('question')
def ask(question):
    """Ask a question about your documents"""
    console.print(Panel(f"[blue]Question: {question}[/blue]", title="RAG Explorer"))

    try:
        engine = SimpleRAGEngine()
        result = engine.answer_question(
            question,
            min_confidence=simple_settings.MIN_CONFIDENCE,
            max_chunks=simple_settings.MAX_CHUNKS
        )

        # Display answer
        console.print(f"[green]Answer:[/green] {result['answer']}")
        console.print(f"[dim]Confidence: {result['confidence']:.2f}[/dim]")

        # Display sources if available
        if result.get('sources'):
            console.print("\n[blue]Sources:[/blue]")
            for i, source in enumerate(result['sources'][:3], 1):
                console.print(f"  {i}. {source}")

    except Exception as e:
        console.print(f"[red]✗ Error answering question: {str(e)}[/red]")
        sys.exit(1)

@cli.command()
@click.argument('query')
def search(query):
    """Search vector database and show top results"""
    console.print(Panel(f"[blue]Searching for: {query}[/blue]", title="Vector Search"))

    try:
        engine = SimpleRAGEngine()
        query_embedding = engine.generate_query_embedding(query)

        results = query_by_embedding(
            query_embedding=query_embedding,
            k=simple_settings.SEARCH_MAX_RESULTS
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

@cli.command()
def crawl():
    """Crawl websites from CRAWL_SOURCES environment variable"""
    if not simple_settings.CRAWL_SOURCES:
        console.print("[red]No crawl sources configured. Set CRAWL_SOURCES environment variable.[/red]")
        sys.exit(1)

    console.print(Panel(f"[blue]Crawling {len(simple_settings.CRAWL_SOURCES)} sources[/blue]", title="Web Crawler"))

    try:
        from rag_explorer.data.web_ingest import crawl_urls

        # Filter out empty sources
        sources = [source.strip() for source in simple_settings.CRAWL_SOURCES if source.strip()]

        if not sources:
            console.print("[red]No valid crawl sources found.[/red]")
            sys.exit(1)

        console.print(f"[dim]Starting crawl with depth={simple_settings.CRAWL_DEPTH}, max_pages={simple_settings.CRAWL_MAX_PAGES}[/dim]")

        result = crawl_urls(
            seeds=sources,
            depth=simple_settings.CRAWL_DEPTH,
            max_pages=simple_settings.CRAWL_MAX_PAGES,
            same_domain=simple_settings.CRAWL_SAME_DOMAIN
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

@cli.command()
def ping():
    """Test connectivity to LLM and embedding providers"""
    console.print(Panel("[blue]Testing Provider Connectivity[/blue]", title="Ping Test"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        results = {}

        # Test LLM provider with retry logic
        llm_task = progress.add_task(f"Testing {simple_settings.PRIMARY_LLM_PROVIDER} LLM...", total=None)
        results['llm'] = _test_provider_with_retry('llm', simple_settings.PRIMARY_LLM_PROVIDER, progress, llm_task)
        progress.remove_task(llm_task)

        # Test embedding provider with retry logic
        emb_task = progress.add_task(f"Testing {simple_settings.PRIMARY_EMBEDDING_PROVIDER} embeddings...", total=None)
        results['embedding'] = _test_provider_with_retry('embedding', simple_settings.PRIMARY_EMBEDDING_PROVIDER, progress, emb_task)
        progress.remove_task(emb_task)

    # Display results
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Provider", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    table.add_row(
        simple_settings.PRIMARY_LLM_PROVIDER,
        "LLM",
        results['llm']['status'],
        results['llm']['details']
    )

    table.add_row(
        simple_settings.PRIMARY_EMBEDDING_PROVIDER,
        "Embedding",
        results['embedding']['status'],
        results['embedding']['details']
    )

    console.print(table)

def _test_provider_with_retry(provider_type: str, provider_name: str, progress, task_id):
    """Test provider connectivity with retry logic"""
    # Early checks: if provider_name is openai and not simple_settings.OPENAI_API_KEY, return {'status':'Not Configured','details':'OPENAI_API_KEY not set'}
    if provider_name == 'openai' and not simple_settings.OPENAI_API_KEY:
        return {'status': 'Not Configured', 'details': 'OPENAI_API_KEY not set'}
    
    # analogous for anthropic and google
    if provider_name == 'anthropic' and not simple_settings.ANTHROPIC_API_KEY:
        return {'status': 'Not Configured', 'details': 'ANTHROPIC_API_KEY not set'}
    
    if provider_name == 'google' and not simple_settings.GOOGLE_API_KEY:
        return {'status': 'Not Configured', 'details': 'GOOGLE_API_KEY not set'}
    
    # Keep retry loop only for configured providers
    for attempt in range(simple_settings.PING_RETRY_COUNT):
        try:
            if attempt > 0:
                progress.update(task_id, description=f"Testing {provider_name} {provider_type} (attempt {attempt + 1}/{simple_settings.PING_RETRY_COUNT})...")
                time.sleep(1)  # Small delay between retries
            
            if provider_name == 'ollama':
                if provider_type == 'llm':
                    response = requests.get(f"{simple_settings.OLLAMA_HOST}/api/tags", timeout=simple_settings.PING_TIMEOUT)
                    return {'status': 'Connected', 'details': f"Ollama running at {simple_settings.OLLAMA_HOST}"}
                else:  # embedding
                    response = requests.get(f"{simple_settings.OLLAMA_HOST}/api/tags", timeout=simple_settings.PING_TIMEOUT)
                    if response.status_code == 200:
                        models = response.json().get('models', [])
                        model_names = [m.get('name', '') for m in models]
                        if any(simple_settings.OLLAMA_EMBEDDING_MODEL in name for name in model_names):
                            return {'status': 'Connected', 'details': f"Embedding model {simple_settings.OLLAMA_EMBEDDING_MODEL} available"}
                        else:
                            return {'status': 'Warning', 'details': f"Model {simple_settings.OLLAMA_EMBEDDING_MODEL} not found in available models"}
            
            elif provider_name == 'openai':
                import openai
                client = openai.OpenAI(api_key=simple_settings.OPENAI_API_KEY)
                models = client.models.list()
                if provider_type == 'llm':
                    return {'status': 'Connected', 'details': f"OpenAI API accessible, {len(models.data)} models available"}
                else:  # embedding
                    return {'status': 'Connected', 'details': f"OpenAI embedding API accessible"}
            
            elif provider_name == 'anthropic':
                if provider_type == 'llm':
                    import anthropic
                    client = anthropic.Anthropic(api_key=simple_settings.ANTHROPIC_API_KEY)
                    models = client.models.list()
                    return {'status': 'Connected', 'details': f"Anthropic API accessible, {len(models.data)} models available"}
                else:  # embedding
                    return {'status': 'Not Supported', 'details': 'Anthropic does not provide embedding models'}
            
            elif provider_name == 'google':
                import google.generativeai as genai
                genai.configure(api_key=simple_settings.GOOGLE_API_KEY)
                models = list(genai.list_models())
                return {'status': 'Connected', 'details': f"Google API accessible, {len(models)} models available"}
            
            else:
                return {'status': 'Unknown', 'details': f"Unknown provider: {provider_name}"}
                
        except Exception as e:
            if attempt == simple_settings.PING_RETRY_COUNT - 1:  # Last attempt
                return {'status': 'Failed', 'details': f"Failed after {simple_settings.PING_RETRY_COUNT} attempts: {str(e)}"}
            # Continue to next attempt
            continue
    
    return {'status': 'Failed', 'details': 'All retry attempts failed'}

@cli.command()
def configure():
    """Interactive configuration editor for environment variables"""
    console.print(Panel("[blue]RAG Explorer Configuration[/blue]", title="Configuration Editor"))

    if not simple_settings.CONFIG_INTERACTIVE_MODE:
        console.print("[yellow]Interactive mode disabled. Set CONFIG_INTERACTIVE_MODE=true to enable.[/yellow]")
        _show_current_config()
        return

    console.print("[dim]Current configuration:[/dim]")
    _show_current_config()

    console.print("\n[blue]Available settings to modify:[/blue]")
    console.print("1. PRIMARY_LLM_PROVIDER")
    console.print("2. PRIMARY_EMBEDDING_PROVIDER")
    console.print("3. CHUNK_SIZE")
    console.print("4. CHUNK_OVERLAP")
    console.print("5. MIN_CONFIDENCE")
    console.print("6. MAX_CHUNKS")
    console.print("7. CRAWL_SOURCES")
    console.print("8. API Keys")
    console.print("9. Model Settings")
    console.print("10. Paths")
    console.print("11. Show all settings")
    console.print("12. Validate current configuration")

    try:
        choice = click.prompt("Select option (1-12)", type=int)

        if choice == 1:
            valid_providers = simple_settings.get_valid_llm_providers()
            new_value = click.prompt("LLM Provider", type=click.Choice(valid_providers), default=simple_settings.PRIMARY_LLM_PROVIDER)
            _update_env_setting("PRIMARY_LLM_PROVIDER", new_value)
        elif choice == 2:
            valid_providers = simple_settings.get_valid_embedding_providers()
            new_value = click.prompt("Embedding Provider", type=click.Choice(valid_providers), default=simple_settings.PRIMARY_EMBEDDING_PROVIDER)
            _update_env_setting("PRIMARY_EMBEDDING_PROVIDER", new_value)
        elif choice == 3:
            new_value = click.prompt("Chunk Size", type=int, default=simple_settings.CHUNK_SIZE)
            _update_env_setting("CHUNK_SIZE", str(new_value))
        elif choice == 4:
            new_value = click.prompt("Chunk Overlap", type=int, default=simple_settings.CHUNK_OVERLAP)
            _update_env_setting("CHUNK_OVERLAP", str(new_value))
        elif choice == 5:
            new_value = click.prompt("Min Confidence", type=float, default=simple_settings.MIN_CONFIDENCE)
            _update_env_setting("MIN_CONFIDENCE", str(new_value))
        elif choice == 6:
            new_value = click.prompt("Max Chunks", type=int, default=simple_settings.MAX_CHUNKS)
            _update_env_setting("MAX_CHUNKS", str(new_value))
        elif choice == 7:
            current_sources = ",".join(simple_settings.CRAWL_SOURCES) if simple_settings.CRAWL_SOURCES else ""
            new_value = click.prompt("Crawl Sources (comma-separated URLs)", default=current_sources)
            _update_env_setting("CRAWL_SOURCES", new_value)
        elif choice == 8:
            _configure_api_keys()
        elif choice == 9:
            _configure_models()
        elif choice == 10:
            _configure_paths()
        elif choice == 11:
            _show_all_settings()
        elif choice == 12:
            _validate_configuration()
        else:
            console.print("[red]Invalid choice[/red]")

    except click.Abort:
        console.print("\n[dim]Configuration cancelled[/dim]")

def _update_env_setting(key: str, value: str):
    """Update or add a setting in the .env file"""
    env_file = Path(".env")

    if env_file.exists():
        lines = env_file.read_text().splitlines()
    else:
        lines = []

    # Find and update existing line or add new one
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            if simple_settings.CONFIG_SHOW_SENSITIVE or not _is_sensitive_key(key):
                console.print(f"[dim]Updating {key}[/dim]")
                lines[i] = f"{key}={value}"
            else:
                console.print(f"[dim]Updating {key} (value hidden)[/dim]")
                lines[i] = f"{key}={value}"
            updated = True
            break

    if not updated:
        if simple_settings.CONFIG_SHOW_SENSITIVE or not _is_sensitive_key(key):
            console.print(f"[dim]Adding {key}={value}[/dim]")
        else:
            console.print(f"[dim]Adding {key} (value hidden)[/dim]")
        lines.append(f"{key}={value}")

    # Write back to file
    env_file.write_text("\n".join(lines) + "\n")
    console.print(f"[green]✓ Updated .env file[/green]")
    console.print("[yellow]Restart required for changes to take effect[/yellow]")

def _is_sensitive_key(key: str) -> bool:
    """Check if a key contains sensitive information"""
    sensitive_keys = ["API_KEY", "TOKEN", "SECRET", "PASSWORD"]
    return any(sensitive in key for sensitive in sensitive_keys)

def _validate_configuration():
    """Validate current configuration and show warnings"""
    console.print("\n[blue]Configuration Validation:[/blue]")

    warnings = []

    # Check provider/API key combinations
    if simple_settings.PRIMARY_LLM_PROVIDER == 'openai' and not simple_settings.OPENAI_API_KEY:
        warnings.append("OpenAI LLM provider selected but OPENAI_API_KEY not set")

    if simple_settings.PRIMARY_LLM_PROVIDER == 'anthropic' and not simple_settings.ANTHROPIC_API_KEY:
        warnings.append("Anthropic LLM provider selected but ANTHROPIC_API_KEY not set")

    if simple_settings.PRIMARY_LLM_PROVIDER == 'google' and not simple_settings.GOOGLE_API_KEY:
        warnings.append("Google LLM provider selected but GOOGLE_API_KEY not set")

    if simple_settings.PRIMARY_EMBEDDING_PROVIDER == 'openai' and not simple_settings.OPENAI_API_KEY:
        warnings.append("OpenAI embedding provider selected but OPENAI_API_KEY not set")

    if simple_settings.PRIMARY_EMBEDDING_PROVIDER == 'google' and not simple_settings.GOOGLE_API_KEY:
        warnings.append("Google embedding provider selected but GOOGLE_API_KEY not set")

    # Check for anthropic embedding (not supported)
    if simple_settings.PRIMARY_EMBEDDING_PROVIDER == 'anthropic':
        warnings.append("Anthropic does not provide embedding models, consider using OpenAI or Google for embeddings")

    # Check chunk settings
    if simple_settings.CHUNK_SIZE < 100:
        warnings.append("CHUNK_SIZE is very small, may result in poor context")

    if simple_settings.CHUNK_OVERLAP >= simple_settings.CHUNK_SIZE:
        warnings.append("CHUNK_OVERLAP should be smaller than CHUNK_SIZE")

    # Check crawl sources
    if simple_settings.CRAWL_SOURCES:
        for source in simple_settings.CRAWL_SOURCES:
            if source.strip() and not (source.startswith('http://') or source.startswith('https://')):
                warnings.append(f"Crawl source should start with http:// or https://: {source}")

    if warnings:
        for warning in warnings:
            console.print(f"[yellow]⚠ {warning}[/yellow]")
    else:
        console.print("[green]✓ Configuration looks good![/green]")

def _show_current_config():
    """Show current configuration in a table"""
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("LLM Provider", simple_settings.PRIMARY_LLM_PROVIDER)
    table.add_row("Embedding Provider", simple_settings.PRIMARY_EMBEDDING_PROVIDER)
    table.add_row("Chunk Size", str(simple_settings.CHUNK_SIZE))
    table.add_row("Chunk Overlap", str(simple_settings.CHUNK_OVERLAP))
    table.add_row("Min Confidence", str(simple_settings.MIN_CONFIDENCE))
    table.add_row("Max Chunks", str(simple_settings.MAX_CHUNKS))
    table.add_row("Docs Folder", simple_settings.DOCS_FOLDER)
    table.add_row("Vector DB Path", simple_settings.CHROMA_DB_PATH)

    console.print(table)

def _configure_api_keys():
    """Configure API keys"""
    console.print("\n[blue]API Key Configuration:[/blue]")
    console.print("1. OPENAI_API_KEY")
    console.print("2. ANTHROPIC_API_KEY")
    console.print("3. GOOGLE_API_KEY")
    
    try:
        choice = click.prompt("Select API key to configure (1-3)", type=int)
        
        if choice == 1:
            current_value = "***" if simple_settings.OPENAI_API_KEY else "Not set"
            console.print(f"Current value: {current_value}")
            new_value = click.prompt("OpenAI API Key", hide_input=True, default="", show_default=False)
            if new_value:
                _update_env_setting("OPENAI_API_KEY", new_value)
        elif choice == 2:
            current_value = "***" if simple_settings.ANTHROPIC_API_KEY else "Not set"
            console.print(f"Current value: {current_value}")
            new_value = click.prompt("Anthropic API Key", hide_input=True, default="", show_default=False)
            if new_value:
                _update_env_setting("ANTHROPIC_API_KEY", new_value)
        elif choice == 3:
            current_value = "***" if simple_settings.GOOGLE_API_KEY else "Not set"
            console.print(f"Current value: {current_value}")
            new_value = click.prompt("Google API Key", hide_input=True, default="", show_default=False)
            if new_value:
                _update_env_setting("GOOGLE_API_KEY", new_value)
        else:
            console.print("[red]Invalid choice[/red]")
    except click.Abort:
        console.print("\n[dim]API key configuration cancelled[/dim]")

def _configure_models():
    """Configure model settings"""
    console.print("\n[blue]Model Configuration:[/blue]")
    console.print("1. OLLAMA_LLM_MODEL")
    console.print("2. OLLAMA_EMBEDDING_MODEL")
    console.print("3. OPENAI_LLM_MODEL")
    console.print("4. OPENAI_EMBEDDING_MODEL")
    console.print("5. GOOGLE_LLM_MODEL")
    console.print("6. GOOGLE_EMBEDDING_MODEL")
    
    try:
        choice = click.prompt("Select model to configure (1-6)", type=int)
        
        if choice == 1:
            new_value = click.prompt("Ollama LLM Model", default=simple_settings.OLLAMA_LLM_MODEL)
            _update_env_setting("OLLAMA_LLM_MODEL", new_value)
        elif choice == 2:
            new_value = click.prompt("Ollama Embedding Model", default=simple_settings.OLLAMA_EMBEDDING_MODEL)
            _update_env_setting("OLLAMA_EMBEDDING_MODEL", new_value)
        elif choice == 3:
            new_value = click.prompt("OpenAI LLM Model", default=simple_settings.OPENAI_LLM_MODEL)
            _update_env_setting("OPENAI_LLM_MODEL", new_value)
        elif choice == 4:
            new_value = click.prompt("OpenAI Embedding Model", default=simple_settings.OPENAI_EMBEDDING_MODEL)
            _update_env_setting("OPENAI_EMBEDDING_MODEL", new_value)
        elif choice == 5:
            new_value = click.prompt("Google LLM Model", default=simple_settings.GOOGLE_LLM_MODEL)
            _update_env_setting("GOOGLE_LLM_MODEL", new_value)
        elif choice == 6:
            new_value = click.prompt("Google Embedding Model", default=simple_settings.GOOGLE_EMBEDDING_MODEL)
            _update_env_setting("GOOGLE_EMBEDDING_MODEL", new_value)
        else:
            console.print("[red]Invalid choice[/red]")
    except click.Abort:
        console.print("\n[dim]Model configuration cancelled[/dim]")

def _configure_paths():
    """Configure path settings"""
    console.print("\n[blue]Path Configuration:[/blue]")
    console.print("1. DOCS_FOLDER")
    console.print("2. CHROMA_DB_PATH")
    
    try:
        choice = click.prompt("Select path to configure (1-2)", type=int)
        
        if choice == 1:
            new_value = click.prompt("Documents Folder", default=simple_settings.DOCS_FOLDER)
            _update_env_setting("DOCS_FOLDER", new_value)
        elif choice == 2:
            new_value = click.prompt("ChromaDB Path", default=simple_settings.CHROMA_DB_PATH)
            _update_env_setting("CHROMA_DB_PATH", new_value)
        else:
            console.print("[red]Invalid choice[/red]")
    except click.Abort:
        console.print("\n[dim]Path configuration cancelled[/dim]")

def _show_all_settings():
    """Show all available environment variables"""
    settings_info = {
        "Providers": ["PRIMARY_LLM_PROVIDER", "PRIMARY_EMBEDDING_PROVIDER"],
        "API Keys": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"],
        "Models": ["OLLAMA_LLM_MODEL", "OPENAI_LLM_MODEL", "ANTHROPIC_LLM_MODEL"],
        "RAG Settings": ["CHUNK_SIZE", "CHUNK_OVERLAP", "MIN_CONFIDENCE", "MAX_CHUNKS"],
        "Crawl Settings": ["CRAWL_SOURCES", "CRAWL_DEPTH", "CRAWL_MAX_PAGES"],
        "Paths": ["DOCS_FOLDER", "CHROMA_DB_PATH"]
    }

    for category, variables in settings_info.items():
        console.print(f"\n[blue]{category}:[/blue]")
        for var in variables:
            console.print(f"  {var}")

@cli.command()
def metrics():
    """Show vector database storage information and metrics"""
    console.print(Panel("[blue]RAG Explorer Metrics[/blue]", title="Storage Metrics"))

    try:
        engine = SimpleRAGEngine()

        if simple_settings.METRICS_OUTPUT_FORMAT == "json":
            metrics_data = _get_metrics_data(engine)
            console.print(json.dumps(metrics_data, indent=2))
            return

        # Table format (default)
        client = get_client()
        try:
            collection = get_collection(client)

            # Get collection info
            collection_count = collection.count()
        finally:
            return_client(client)

        # Storage metrics table
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Details", style="dim")

        table.add_row("Total Chunks", str(collection_count), "Indexed document chunks")

        # Check docs folder
        docs_path = Path(simple_settings.DOCS_FOLDER)
        if docs_path.exists():
            doc_files = list(docs_path.rglob("*.txt")) + list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.pdf"))
            table.add_row("Local Files", str(len(doc_files)), f"Files in {simple_settings.DOCS_FOLDER}")

        # Database size
        db_path = Path(simple_settings.CHROMA_DB_PATH)
        if db_path.exists():
            db_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
            db_size_mb = db_size / (1024 * 1024)
            table.add_row("Database Size", f"{db_size_mb:.1f} MB", simple_settings.CHROMA_DB_PATH)

        # Provider info
        table.add_row("LLM Provider", simple_settings.PRIMARY_LLM_PROVIDER, "Active language model")
        table.add_row("Embedding Provider", simple_settings.PRIMARY_EMBEDDING_PROVIDER, "Active embedding model")

        console.print(table)

        # Source breakdown if available
        if collection_count > 0:
            console.print("\n[blue]Sources Breakdown:[/blue]")
            try:
                # Use peek() to get sample documents without needing embeddings
                sample_results = collection.peek(limit=min(100, collection_count))

                sources = {}
                if sample_results.get('metadatas'):
                    for meta in sample_results['metadatas']:
                        source = meta.get('source', 'Unknown')
                        sources[source] = sources.get(source, 0) + 1

                for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]:
                    console.print(f"  {source}: {count} chunks")

            except Exception as e:
                console.print(f"[dim]Could not analyze sources: {e}[/dim]")

    except Exception as e:
        console.print(f"[red]✗ Error getting metrics: {str(e)}[/red]")
        sys.exit(1)

def _get_metrics_data(engine) -> Dict[str, Any]:
    """Get metrics data as a dictionary for JSON output"""
    try:
        client = get_client()
        try:
            collection = get_collection(client)
            collection_count = collection.count()
        finally:
            return_client(client)

        docs_path = Path(simple_settings.DOCS_FOLDER)
        local_files = 0
        if docs_path.exists():
            doc_files = list(docs_path.rglob("*.txt")) + list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.pdf"))
            local_files = len(doc_files)

        db_path = Path(simple_settings.CHROMA_DB_PATH)
        db_size_mb = 0
        if db_path.exists():
            db_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
            db_size_mb = db_size / (1024 * 1024)

        return {
            "total_chunks": collection_count,
            "local_files": local_files,
            "database_size_mb": round(db_size_mb, 1),
            "llm_provider": simple_settings.PRIMARY_LLM_PROVIDER,
            "embedding_provider": simple_settings.PRIMARY_EMBEDDING_PROVIDER,
            "docs_folder": simple_settings.DOCS_FOLDER,
            "vector_db_path": simple_settings.CHROMA_DB_PATH
        }
    except Exception as e:
        return {"error": str(e)}

@cli.command()
def status():
    """Show system status and configuration"""
    console.print(Panel("[blue]RAG Explorer Status[/blue]", title="System Status"))

    # Get DB stats directly without requiring engine initialization
    try:
        client = get_client()
        try:
            collection = get_collection(client)
            doc_count = collection.count()
            db_connected = True
        finally:
            return_client(client)
    except Exception as e:
        doc_count = 0
        db_connected = False
        console.print(f"[yellow]⚠ Database connection failed: {e}[/yellow]")

    # Create status table
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    # Provider information from settings
    table.add_row(
        "LLM Provider",
        simple_settings.PRIMARY_LLM_PROVIDER,
        f"Configured: {simple_settings.PRIMARY_LLM_PROVIDER}"
    )
    table.add_row(
        "Embedding Provider",
        simple_settings.PRIMARY_EMBEDDING_PROVIDER,
        f"Configured: {simple_settings.PRIMARY_EMBEDDING_PROVIDER}"
    )

    # Database information
    table.add_row(
        "Vector Database",
        "Connected" if db_connected else "Disconnected",
        f"Documents: {doc_count}"
    )

    # Storage information
    table.add_row(
        "Storage Path",
        "Available" if os.path.exists(simple_settings.CHROMA_DB_PATH) else "Not Found",
        simple_settings.CHROMA_DB_PATH
    )

    console.print(table)

    # Try to initialize engine for additional info, but handle failures gracefully
    try:
        engine = SimpleRAGEngine()
        status_info = engine.get_status()
        
        # Provider availability
        providers = status_info.get('available_providers', {})
        if providers:
            console.print("\n[blue]Available Providers:[/blue]")
            for provider_type, provider_list in providers.items():
                console.print(f"  {provider_type.title()}: {', '.join(provider_list)}")
        
        console.print(f"\n[green]✓ RAG Engine initialized successfully[/green]")
        
    except Exception as e:
        console.print(f"\n[yellow]⚠ RAG Engine initialization failed: {e}[/yellow]")
        console.print("[dim]This may indicate missing API keys or provider configuration issues.[/dim]")
        console.print("[dim]Use 'configure' command to set up providers or 'ping' to test connectivity.[/dim]")

if __name__ == "__main__":
    cli()
