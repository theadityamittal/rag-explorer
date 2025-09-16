import click  
import sys
from rich.console import Console
from rich.panel import Panel

console = Console()

@click.command()
def configure():
    """Interactive configuration editor for environment variables"""
    console.print(Panel("[blue]RAG Explorer Configuration[/blue]", title="Configuration Editor"))

    if not settings.CONFIG_INTERACTIVE_MODE:
        console.print("[yellow]Interactive mode disabled. Set CONFIG_INTERACTIVE_MODE=true to enable.[/yellow]")
        _show_current_config()
        return

    while True:
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
        console.print("13. Exit configuration")
        console.print("\n[dim]Press Ctrl+C (or Cmd+C on Mac) to quit anytime[/dim]")

        try:
            choice = click.prompt("Select option (1-13)", type=int)

            if choice == 1:
                valid_providers = settings.get_valid_llm_providers()
                new_value = click.prompt("LLM Provider", type=click.Choice(valid_providers), default=settings.PRIMARY_LLM_PROVIDER)
                _update_env_setting("PRIMARY_LLM_PROVIDER", new_value)
            elif choice == 2:
                valid_providers = settings.get_valid_embedding_providers()
                new_value = click.prompt("Embedding Provider", type=click.Choice(valid_providers), default=settings.PRIMARY_EMBEDDING_PROVIDER)
                _update_env_setting("PRIMARY_EMBEDDING_PROVIDER", new_value)
            elif choice == 3:
                new_value = click.prompt("Chunk Size", type=int, default=settings.CHUNK_SIZE)
                _update_env_setting("CHUNK_SIZE", str(new_value))
            elif choice == 4:
                new_value = click.prompt("Chunk Overlap", type=int, default=settings.CHUNK_OVERLAP)
                _update_env_setting("CHUNK_OVERLAP", str(new_value))
            elif choice == 5:
                new_value = click.prompt("Min Confidence", type=float, default=settings.MIN_CONFIDENCE)
                _update_env_setting("MIN_CONFIDENCE", str(new_value))
            elif choice == 6:
                new_value = click.prompt("Max Chunks", type=int, default=settings.MAX_CHUNKS)
                _update_env_setting("MAX_CHUNKS", str(new_value))
            elif choice == 7:
                current_sources = ",".join(settings.CRAWL_SOURCES) if settings.CRAWL_SOURCES else ""
                new_value = click.prompt("Crawl Sources (comma-separated URLs)", default=current_sources)
                _update_env_setting("CRAWL_SOURCES", new_value)
            elif choice == 8:
                result = _configure_api_keys()
                if result == "back":
                    continue  # Return to main menu
            elif choice == 9:
                result = _configure_models()
                if result == "back":
                    continue  # Return to main menu
            elif choice == 10:
                result = _configure_paths()
                if result == "back":
                    continue  # Return to main menu
            elif choice == 11:
                _show_all_settings()
            elif choice == 12:
                _validate_configuration()
            elif choice == 13:
                console.print("[green]Configuration completed[/green]")
                break
            else:
                console.print("[red]Invalid choice[/red]")

        except click.Abort:
            console.print("\n[dim]Configuration cancelled[/dim]")
            break
