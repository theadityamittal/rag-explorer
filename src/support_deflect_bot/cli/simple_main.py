"""Simple CLI interface for RAG Explorer."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from support_deflect_bot.engine.simple_rag_engine import SimpleRAGEngine
from support_deflect_bot.utils.simple_settings import (
    PRIMARY_LLM_PROVIDER,
    PRIMARY_EMBEDDING_PROVIDER,
    DOCS_FOLDER,
    CHROMA_DB_PATH
)

console = Console()

@click.group()
@click.version_option()
def cli():
    """RAG Explorer - Simple document Q&A tool for personal use"""
    pass

@cli.command()
@click.argument('docs_path', default=DOCS_FOLDER)
@click.option('--chunk-size', default=800, help='Size of document chunks')
@click.option('--chunk-overlap', default=100, help='Overlap between chunks')
def index(docs_path, chunk_size, chunk_overlap):
    """Index documents from a directory"""
    console.print(Panel(f"[blue]Indexing documents from: {docs_path}[/blue]", title="RAG Explorer"))
    
    try:
        engine = SimpleRAGEngine()
        result = engine.index_documents(docs_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        console.print(f"[green]✓ Successfully indexed {result['count']} document chunks[/green]")
        console.print(f"[dim]Files processed: {result.get('files_processed', 'N/A')}[/dim]")
        console.print(f"[dim]Storage: {CHROMA_DB_PATH}[/dim]")
        
    except Exception as e:
        console.print(f"[red]✗ Error indexing documents: {str(e)}[/red]")
        sys.exit(1)

@cli.command()
@click.argument('question')
@click.option('--confidence', default=None, type=float, help='Minimum confidence threshold')
@click.option('--max-chunks', default=5, help='Maximum chunks to retrieve')
def ask(question, confidence, max_chunks):
    """Ask a question about your documents"""
    console.print(Panel(f"[blue]Question: {question}[/blue]", title="RAG Explorer"))
    
    try:
        engine = SimpleRAGEngine()
        result = engine.answer_question(question, min_confidence=confidence, max_chunks=max_chunks)
        
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
def status():
    """Show system status and configuration"""
    console.print(Panel("[blue]RAG Explorer Status[/blue]", title="System Status"))
    
    try:
        engine = SimpleRAGEngine()
        status_info = engine.get_status()
        
        # Create status table
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")
        
        # Provider information
        table.add_row(
            "LLM Provider", 
            status_info.get('llm_provider', 'Unknown'),
            f"Primary: {PRIMARY_LLM_PROVIDER}"
        )
        table.add_row(
            "Embedding Provider", 
            status_info.get('embedding_provider', 'Unknown'),
            f"Primary: {PRIMARY_EMBEDDING_PROVIDER}"
        )
        
        # Database information
        table.add_row(
            "Vector Database", 
            "Connected" if status_info.get('db_connected') else "Disconnected",
            f"Documents: {status_info.get('doc_count', 0)}"
        )
        
        # Storage information
        table.add_row(
            "Storage Path", 
            "Available" if os.path.exists(CHROMA_DB_PATH) else "Not Found",
            CHROMA_DB_PATH
        )
        
        console.print(table)
        
        # Provider availability
        providers = status_info.get('available_providers', {})
        if providers:
            console.print("\n[blue]Available Providers:[/blue]")
            for provider_type, provider_list in providers.items():
                console.print(f"  {provider_type.title()}: {', '.join(provider_list)}")
        
    except Exception as e:
        console.print(f"[red]✗ Error getting status: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    cli()
