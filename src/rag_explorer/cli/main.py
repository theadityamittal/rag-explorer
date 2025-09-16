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

from .commands import index, ask, search, crawl, configure
from rag_explorer.utils import settings
from rag_explorer.engine.database import get_simple_client, get_or_create_collection

console = Console()
from rag_explorer.engine import UnifiedDocumentProcessor, UnifiedQueryService, UnifiedEmbeddingService, UnifiedRAGEngine

global _doc_processor # Singleton document processor instance
_doc_processor = UnifiedDocumentProcessor() # Initialize once   
global _query_service # Singleton query service instance
_query_service = UnifiedQueryService() # Initialize once
global _embedding_service # Singleton embedding service instance
_embedding_service = UnifiedEmbeddingService() # Initialize once
global _rag_engine # Singleton RAG engine instance
_rag_engine = UnifiedRAGEngine() # Initialize once

@click.group()
@click.version_option()
def cli():
    """RAG Explorer - Simple document Q&A tool for personal use"""
    pass

cli.add_command(index)
cli.add_command(ask)
cli.add_command(search)
cli.add_command(crawl)
cli.add_command(configure)


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
        llm_task = progress.add_task(f"Testing {settings.PRIMARY_LLM_PROVIDER} LLM...", total=None)
        results['llm'] = _test_provider_with_retry('llm', settings.PRIMARY_LLM_PROVIDER, progress, llm_task)
        progress.remove_task(llm_task)

        # Test embedding provider with retry logic
        emb_task = progress.add_task(f"Testing {settings.PRIMARY_EMBEDDING_PROVIDER} embeddings...", total=None)
        results['embedding'] = _test_provider_with_retry('embedding', settings.PRIMARY_EMBEDDING_PROVIDER, progress, emb_task)
        progress.remove_task(emb_task)

    # Display results
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Provider", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    table.add_row(
        settings.PRIMARY_LLM_PROVIDER,
        "LLM",
        results['llm']['status'],
        results['llm']['details']
    )

    table.add_row(
        settings.PRIMARY_EMBEDDING_PROVIDER,
        "Embedding",
        results['embedding']['status'],
        results['embedding']['details']
    )

    console.print(table)

def _test_provider_with_retry(provider_type: str, provider_name: str, progress, task_id):
    """Test provider connectivity with retry logic"""
    # Early checks: if provider_name is openai and not simple_settings.OPENAI_API_KEY, return {'status':'Not Configured','details':'OPENAI_API_KEY not set'}
    if provider_name == 'openai' and not settings.OPENAI_API_KEY:
        return {'status': 'Not Configured', 'details': 'OPENAI_API_KEY not set'}
    
    # analogous for anthropic and google
    if provider_name == 'anthropic' and not settings.ANTHROPIC_API_KEY:
        return {'status': 'Not Configured', 'details': 'ANTHROPIC_API_KEY not set'}
    
    if provider_name == 'google' and not settings.GEMINI_API_KEY:
        return {'status': 'Not Configured', 'details': 'GEMINI_API_KEY not set'}
    
    # Keep retry loop only for configured providers
    for attempt in range(settings.PING_RETRY_COUNT):
        try:
            if attempt > 0:
                progress.update(task_id, description=f"Testing {provider_name} {provider_type} (attempt {attempt + 1}/{settings.PING_RETRY_COUNT})...")
                time.sleep(1)  # Small delay between retries
            
            if provider_name == 'ollama':
                if provider_type == 'llm':
                    response = requests.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=settings.PING_TIMEOUT)
                    return {'status': 'Connected', 'details': f"Ollama running at {settings.OLLAMA_HOST}"}
                else:  # embedding
                    response = requests.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=settings.PING_TIMEOUT)
                    if response.status_code == 200:
                        models = response.json().get('models', [])
                        model_names = [m.get('name', '') for m in models]
                        if any(settings.OLLAMA_EMBEDDING_MODEL in name for name in model_names):
                            return {'status': 'Connected', 'details': f"Embedding model {settings.OLLAMA_EMBEDDING_MODEL} available"}
                        else:
                            return {'status': 'Warning', 'details': f"Model {settings.OLLAMA_EMBEDDING_MODEL} not found in available models"}
            
            elif provider_name == 'openai':
                import openai
                client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                models = client.models.list()
                if provider_type == 'llm':
                    return {'status': 'Connected', 'details': f"OpenAI API accessible, {len(models.data)} models available"}
                else:  # embedding
                    return {'status': 'Connected', 'details': f"OpenAI embedding API accessible"}
            
            elif provider_name == 'anthropic':
                if provider_type == 'llm':
                    import anthropic
                    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
                    models = client.models.list()
                    return {'status': 'Connected', 'details': f"Anthropic API accessible, {len(models.data)} models available"}
                else:  # embedding
                    return {'status': 'Not Supported', 'details': 'Anthropic does not provide embedding models'}
            
            elif provider_name == 'google':
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                models = list(genai.list_models())
                return {'status': 'Connected', 'details': f"Google API accessible, {len(models)} models available"}
            
            else:
                return {'status': 'Unknown', 'details': f"Unknown provider: {provider_name}"}
                
        except Exception as e:
            if attempt == settings.PING_RETRY_COUNT - 1:  # Last attempt
                return {'status': 'Failed', 'details': f"Failed after {settings.PING_RETRY_COUNT} attempts: {str(e)}"}
            # Continue to next attempt
            continue
    
    return {'status': 'Failed', 'details': 'All retry attempts failed'}


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
            if settings.CONFIG_SHOW_SENSITIVE or not _is_sensitive_key(key):
                console.print(f"[dim]Updating {key}[/dim]")
                lines[i] = f"{key}={value}"
            else:
                console.print(f"[dim]Updating {key} (value hidden)[/dim]")
                lines[i] = f"{key}={value}"
            updated = True
            break

    if not updated:
        if settings.CONFIG_SHOW_SENSITIVE or not _is_sensitive_key(key):
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
    if settings.PRIMARY_LLM_PROVIDER == 'openai' and not settings.OPENAI_API_KEY:
        warnings.append("OpenAI LLM provider selected but OPENAI_API_KEY not set")

    if settings.PRIMARY_LLM_PROVIDER == 'anthropic' and not settings.ANTHROPIC_API_KEY:
        warnings.append("Anthropic LLM provider selected but ANTHROPIC_API_KEY not set")

    if settings.PRIMARY_LLM_PROVIDER == 'google' and not settings.GEMINI_API_KEY:
        warnings.append("Google LLM provider selected but GEMINI_API_KEY not set")

    if settings.PRIMARY_EMBEDDING_PROVIDER == 'openai' and not settings.OPENAI_API_KEY:
        warnings.append("OpenAI embedding provider selected but OPENAI_API_KEY not set")

    if settings.PRIMARY_EMBEDDING_PROVIDER == 'google' and not settings.GEMINI_API_KEY:
        warnings.append("Google embedding provider selected but GEMINI_API_KEY not set")

    # Check for anthropic embedding (not supported)
    if settings.PRIMARY_EMBEDDING_PROVIDER == 'anthropic':
        warnings.append("Anthropic does not provide embedding models, consider using OpenAI or Google for embeddings")

    # Check chunk settings
    if settings.CHUNK_SIZE < 100:
        warnings.append("CHUNK_SIZE is very small, may result in poor context")

    if settings.CHUNK_OVERLAP >= settings.CHUNK_SIZE:
        warnings.append("CHUNK_OVERLAP should be smaller than CHUNK_SIZE")

    # Check crawl sources
    if settings.CRAWL_SOURCES:
        for source in settings.CRAWL_SOURCES:
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
    table.add_column("Status", style="yellow")

    # Provider information with API key status
    llm_api_status = _get_api_key_status(settings.PRIMARY_LLM_PROVIDER)
    table.add_row("LLM Provider", settings.PRIMARY_LLM_PROVIDER, llm_api_status)

    embedding_api_status = _get_api_key_status(settings.PRIMARY_EMBEDDING_PROVIDER)
    table.add_row("Embedding Provider", settings.PRIMARY_EMBEDDING_PROVIDER, embedding_api_status)

    # Model settings for primary providers
    llm_model = _get_model_for_provider(settings.PRIMARY_LLM_PROVIDER, "llm")
    table.add_row("LLM Model", llm_model, "")

    embedding_model = _get_model_for_provider(settings.PRIMARY_EMBEDDING_PROVIDER, "embedding")
    table.add_row("Embedding Model", embedding_model, "")

    # RAG settings
    table.add_row("Chunk Size", str(settings.CHUNK_SIZE), "")
    table.add_row("Chunk Overlap", str(settings.CHUNK_OVERLAP), "")
    table.add_row("Min Confidence", str(settings.MIN_CONFIDENCE), "")
    table.add_row("Max Chunks", str(settings.MAX_CHUNKS), "")

    # Paths
    table.add_row("Docs Folder", settings.DOCS_FOLDER, "")
    table.add_row("Vector DB Path", settings.CHROMA_DB_PATH, "")

    console.print(table)

def _get_api_key_status(provider: str) -> str:
    """Get API key status for a provider"""
    if provider == "openai":
        return "✓ Set" if settings.OPENAI_API_KEY else "✗ Missing"
    elif provider == "anthropic":
        return "✓ Set" if settings.ANTHROPIC_API_KEY else "✗ Missing"
    elif provider == "google":
        return "✓ Set" if settings.GEMINI_API_KEY else "✗ Missing"
    elif provider == "ollama":
        return "N/A (Local)"
    else:
        return "Unknown"

def _get_model_for_provider(provider: str, model_type: str) -> str:
    """Get the model setting for a specific provider and type"""
    if provider == "openai":
        return settings.OPENAI_LLM_MODEL if model_type == "llm" else settings.OPENAI_EMBEDDING_MODEL
    elif provider == "anthropic":
        return settings.ANTHROPIC_LLM_MODEL if model_type == "llm" else "N/A (No embeddings)"
    elif provider == "google":
        return settings.GEMINI_LLM_MODEL if model_type == "llm" else settings.GEMINI_EMBEDDING_MODEL
    elif provider == "ollama":
        return settings.OLLAMA_LLM_MODEL if model_type == "llm" else settings.OLLAMA_EMBEDDING_MODEL
    else:
        return "Unknown"

def _configure_api_keys():
    """Configure API keys"""
    console.print("\n[blue]API Key Configuration:[/blue]")
    console.print("1. OPENAI_API_KEY")
    console.print("2. ANTHROPIC_API_KEY")
    console.print("3. GEMINI_API_KEY")
    console.print("4. Back to main menu")
    console.print("\n[dim]Press Ctrl+C (or Cmd+C on Mac) to quit anytime[/dim]")

    try:
        choice = click.prompt("Select option (1-4)", type=int)

        if choice == 1:
            current_value = "***" if settings.OPENAI_API_KEY else "Not set"
            console.print(f"Current value: {current_value}")
            new_value = click.prompt("OpenAI API Key", hide_input=True, default="", show_default=False)
            if new_value:
                _update_env_setting("OPENAI_API_KEY", new_value)
        elif choice == 2:
            current_value = "***" if settings.ANTHROPIC_API_KEY else "Not set"
            console.print(f"Current value: {current_value}")
            new_value = click.prompt("Anthropic API Key", hide_input=True, default="", show_default=False)
            if new_value:
                _update_env_setting("ANTHROPIC_API_KEY", new_value)
        elif choice == 3:
            current_value = "***" if settings.GEMINI_API_KEY else "Not set"
            console.print(f"Current value: {current_value}")
            new_value = click.prompt("Google API Key", hide_input=True, default="", show_default=False)
            if new_value:
                _update_env_setting("GEMINI_API_KEY", new_value)
        elif choice == 4:
            return "back"
        else:
            console.print("[red]Invalid choice[/red]")
    except click.Abort:
        console.print("\n[dim]API key configuration cancelled[/dim]")

    return None

def _configure_models():
    """Configure model settings"""
    console.print("\n[blue]Model Configuration:[/blue]")
    console.print("1. OLLAMA_LLM_MODEL")
    console.print("2. OLLAMA_EMBEDDING_MODEL")
    console.print("3. OPENAI_LLM_MODEL")
    console.print("4. OPENAI_EMBEDDING_MODEL")
    console.print("5. GEMINI_LLM_MODEL")
    console.print("6. GEMINI_EMBEDDING_MODEL")
    console.print("7. Back to main menu")
    console.print("\n[dim]Press Ctrl+C (or Cmd+C on Mac) to quit anytime[/dim]")

    try:
        choice = click.prompt("Select option (1-7)", type=int)

        if choice == 1:
            new_value = click.prompt("Ollama LLM Model", default=settings.OLLAMA_LLM_MODEL)
            _update_env_setting("OLLAMA_LLM_MODEL", new_value)
        elif choice == 2:
            new_value = click.prompt("Ollama Embedding Model", default=settings.OLLAMA_EMBEDDING_MODEL)
            _update_env_setting("OLLAMA_EMBEDDING_MODEL", new_value)
        elif choice == 3:
            new_value = click.prompt("OpenAI LLM Model", default=settings.OPENAI_LLM_MODEL)
            _update_env_setting("OPENAI_LLM_MODEL", new_value)
        elif choice == 4:
            new_value = click.prompt("OpenAI Embedding Model", default=settings.OPENAI_EMBEDDING_MODEL)
            _update_env_setting("OPENAI_EMBEDDING_MODEL", new_value)
        elif choice == 5:
            new_value = click.prompt("Google LLM Model", default=settings.GEMINI_LLM_MODEL)
            _update_env_setting("GEMINI_LLM_MODEL", new_value)
        elif choice == 6:
            new_value = click.prompt("Google Embedding Model", default=settings.GEMINI_EMBEDDING_MODEL)
            _update_env_setting("GEMINI_EMBEDDING_MODEL", new_value)
        elif choice == 7:
            return "back"
        else:
            console.print("[red]Invalid choice[/red]")
    except click.Abort:
        console.print("\n[dim]Model configuration cancelled[/dim]")

    return None

def _configure_paths():
    """Configure path settings"""
    console.print("\n[blue]Path Configuration:[/blue]")
    console.print("1. DOCS_FOLDER")
    console.print("2. CHROMA_DB_PATH")
    console.print("3. Back to main menu")
    console.print("\n[dim]Press Ctrl+C (or Cmd+C on Mac) to quit anytime[/dim]")

    try:
        choice = click.prompt("Select option (1-3)", type=int)

        if choice == 1:
            new_value = click.prompt("Documents Folder", default=settings.DOCS_FOLDER)
            _update_env_setting("DOCS_FOLDER", new_value)
        elif choice == 2:
            new_value = click.prompt("ChromaDB Path", default=settings.CHROMA_DB_PATH)
            _update_env_setting("CHROMA_DB_PATH", new_value)
        elif choice == 3:
            return "back"
        else:
            console.print("[red]Invalid choice[/red]")
    except click.Abort:
        console.print("\n[dim]Path configuration cancelled[/dim]")

    return None

def _show_all_settings():
    """Show all available environment variables"""
    settings_info = {
        "Providers": ["PRIMARY_LLM_PROVIDER", "PRIMARY_EMBEDDING_PROVIDER"],
        "API Keys": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"],
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
        engine = UnifiedRAGEngine()

        if settings.METRICS_OUTPUT_FORMAT == "json":
            metrics_data = _get_metrics_data(engine)
            console.print(json.dumps(metrics_data, indent=2))
            return

        # Table format (default)
        client = get_simple_client()
        collection = get_or_create_collection(client)

        # Get collection info
        collection_count = collection.count()

        # Storage metrics table
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Details", style="dim")

        table.add_row("Total Chunks", str(collection_count), "Indexed document chunks")

        # Check docs folder
        docs_path = Path(settings.DOCS_FOLDER)
        if docs_path.exists():
            doc_files = list(docs_path.rglob("*.txt")) + list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.pdf"))
            table.add_row("Local Files", str(len(doc_files)), f"Files in {settings.DOCS_FOLDER}")

        # Database size
        db_path = Path(settings.CHROMA_DB_PATH)
        if db_path.exists():
            db_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
            db_size_mb = db_size / (1024 * 1024)
            table.add_row("Database Size", f"{db_size_mb:.1f} MB", settings.CHROMA_DB_PATH)

        # Provider info
        table.add_row("LLM Provider", settings.PRIMARY_LLM_PROVIDER, "Active language model")
        table.add_row("Embedding Provider", settings.PRIMARY_EMBEDDING_PROVIDER, "Active embedding model")

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
        client = get_simple_client()
        collection = get_or_create_collection(client)
        collection_count = collection.count()

        docs_path = Path(settings.DOCS_FOLDER)
        local_files = 0
        if docs_path.exists():
            doc_files = list(docs_path.rglob("*.txt")) + list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.pdf"))
            local_files = len(doc_files)

        db_path = Path(settings.CHROMA_DB_PATH)
        db_size_mb = 0
        if db_path.exists():
            db_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
            db_size_mb = db_size / (1024 * 1024)

        return {
            "total_chunks": collection_count,
            "local_files": local_files,
            "database_size_mb": round(db_size_mb, 1),
            "llm_provider": settings.PRIMARY_LLM_PROVIDER,
            "embedding_provider": settings.PRIMARY_EMBEDDING_PROVIDER,
            "docs_folder": settings.DOCS_FOLDER,
            "vector_db_path": settings.CHROMA_DB_PATH
        }
    except Exception as e:
        return {"error": str(e)}

@cli.command()
def status():
    """Show system status and configuration"""
    console.print(Panel("[blue]RAG Explorer Status[/blue]", title="System Status"))

    # Get DB stats directly without requiring engine initialization
    try:
        client = get_simple_client()
        collection = get_or_create_collection(client)
        doc_count = collection.count()
        db_connected = True
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
        settings.PRIMARY_LLM_PROVIDER,
        f"Configured: {settings.PRIMARY_LLM_PROVIDER}"
    )
    table.add_row(
        "Embedding Provider",
        settings.PRIMARY_EMBEDDING_PROVIDER,
        f"Configured: {settings.PRIMARY_EMBEDDING_PROVIDER}"
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
        "Available" if os.path.exists(settings.CHROMA_DB_PATH) else "Not Found",
        settings.CHROMA_DB_PATH
    )

    console.print(table)

    # Try to initialize engine for additional info, but handle failures gracefully
    try:
        engine = UnifiedRAGEngine()
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
