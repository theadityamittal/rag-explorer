import click  
import sys
from rich.console import Console
from rich.panel import Panel

from rag_explorer.utils.settings import (MIN_CONFIDENCE, MAX_CHUNKS)

from rag_explorer.engine import UnifiedRAGEngine

console = Console()

global _rag_engine
if '_rag_engine' not in globals():
    _rag_engine = UnifiedRAGEngine()

@click.command()
@click.argument('question')
def ask(question):
    """Ask a question about your documents"""
    console.print(Panel(f"[blue]Question: {question}[/blue]", title="RAG Explorer"))

    try:
        result = _rag_engine.answer_question(
            question=question,
            k=MAX_CHUNKS,
            min_confidence=MIN_CONFIDENCE,
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
