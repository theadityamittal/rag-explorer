"""Interactive Q&A session handler."""

import time
from typing import List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Use RAG functionality from src module
from src.core.rag import answer_question

from .output import format_answer


def start_interactive_session(
    console: Console,
    domains: Optional[List[str]] = None,
    meter=None,
    ctx_obj: dict = None,
):
    """Start an interactive Q&A session."""
    question_count = 0

    # Welcome message
    console.print(
        "\n🤖 Support Deflect Bot - Interactive Q&A Session", style="bold cyan"
    )
    console.print("Ask me anything about your documentation. Type 'end' to exit.\n")

    if domains:
        console.print(
            f"🌐 Filtering responses to domains: {', '.join(domains)}", style="dim"
        )

    while True:
        try:
            # Get user input
            question = console.input("❓ [bold blue]You:[/bold blue] ")

            # Check for exit command
            if question.strip().lower() in ["end", "exit", "quit", "q"]:
                break

            # Skip empty questions
            if not question.strip():
                continue

            question_count += 1
            t0 = time.perf_counter()

            try:
                # Get answer from RAG system
                result = answer_question(question.strip(), domains=domains)

                # Format and display the answer
                format_answer(console, result, ctx_obj and ctx_obj.get("quiet", False))

                # Record metrics
                if meter:
                    meter.observe(time.perf_counter() - t0)

            except ConnectionError:
                console.print("❌ Database or LLM connection failed", style="red")
                continue
            except Exception as e:
                console.print(f"❌ Error processing question: {e}", style="red")
                continue

            console.print()  # Add spacing between Q&A pairs

        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+C or EOF gracefully
            break

    # Goodbye message
    if question_count > 0:
        console.print(
            f"\n👋 Goodbye! Asked {question_count} questions in this session.",
            style="green",
        )
    else:
        console.print("\n👋 Goodbye!", style="green")
