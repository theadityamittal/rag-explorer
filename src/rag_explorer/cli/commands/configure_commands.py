"""Interactive configure command for initial setup of Support Deflect Bot."""

import os
from pathlib import Path
from typing import Dict, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from ...utils.settings import validate_configuration, get_configured_providers


def get_current_env_value(key: str) -> Optional[str]:
    """Get current environment variable value."""
    return os.getenv(key)


def write_env_file(config_values: Dict[str, str], env_path: str = ".env") -> None:
    """Write configuration values to .env file."""
    existing_lines = []
    existing_keys = set()
    
    # Read existing .env file if it exists
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key = line.split('=')[0]
                    existing_keys.add(key)
                existing_lines.append(line)
    
    # Add new configuration values
    with open(env_path, 'w') as f:
        # Write existing lines first
        for line in existing_lines:
            if line.strip() and not line.startswith('#') and '=' in line:
                key = line.split('=')[0]
                if key in config_values:
                    # Update with new value
                    f.write(f"{key}={config_values[key]}\n")
                    del config_values[key]  # Remove from new values to add
                else:
                    f.write(line + "\n")
            else:
                f.write(line + "\n")
        
        # Add any remaining new values
        if config_values:
            f.write("\n# Configuration added by configure command\n")
            for key, value in config_values.items():
                f.write(f"{key}={value}\n")


@click.command()
@click.option("--reset", is_flag=True, help="Reset and reconfigure all settings")
@click.option("--quick", is_flag=True, help="Quick setup with minimal prompts")
@click.pass_context
def configure(ctx, reset: bool, quick: bool):
    """Interactive configuration setup for Support Deflect Bot."""
    console = Console()
    config_values = {}
    
    try:
        # Welcome message
        console.print(
            Panel.fit(
                "[bold cyan]Support Deflect Bot - Initial Setup[/bold cyan]\n\n"
                "This wizard will help you configure the essential settings for your bot.\n"
                "You can modify these later by editing the .env file or running this command again.",
                title="Configuration Wizard",
                border_style="cyan"
            )
        )
        
        # Check if already configured
        if not reset and not quick:
            current_providers = get_configured_providers()
            if current_providers:
                console.print(f"\n[yellow]Current providers configured: {', '.join(current_providers)}[/yellow]")
                if not Confirm.ask("Do you want to reconfigure settings?", default=False):
                    console.print("[green]Configuration unchanged.[/green]")
                    return
        
        console.print("\n[bold]Step 1: Provider Configuration[/bold]")
        
        # 1. Primary LLM Provider Selection
        current_llm = get_current_env_value("PRIMARY_LLM_PROVIDER") or "gemini"
        console.print(f"\nCurrent LLM provider: [cyan]{current_llm}[/cyan]")
        
        if quick:
            config_values["PRIMARY_LLM_PROVIDER"] = current_llm
        else:
            llm_options = ["gemini", "openai", "anthropic", "mistral", "groq", "ollama"]
            console.print("Available LLM providers:", ", ".join(llm_options))
            
            new_llm = Prompt.ask(
                "Primary LLM provider",
                default=current_llm,
                choices=llm_options
            )
            config_values["PRIMARY_LLM_PROVIDER"] = new_llm
        
        # 2. Primary Embedding Provider Selection  
        current_embed = get_current_env_value("PRIMARY_EMBEDDING_PROVIDER") or "gemini"
        console.print(f"Current embedding provider: [cyan]{current_embed}[/cyan]")
        
        if quick:
            config_values["PRIMARY_EMBEDDING_PROVIDER"] = current_embed
        else:
            embed_options = ["gemini", "openai", "ollama"]
            console.print("Available embedding providers:", ", ".join(embed_options))
            
            new_embed = Prompt.ask(
                "Primary embedding provider",
                default=current_embed,
                choices=embed_options
            )
            config_values["PRIMARY_EMBEDDING_PROVIDER"] = new_embed
        
        # 3. API Keys Configuration
        console.print("\n[bold]Step 2: API Keys[/bold]")
        
        selected_llm = config_values["PRIMARY_LLM_PROVIDER"]
        selected_embed = config_values["PRIMARY_EMBEDDING_PROVIDER"]
        
        # Determine which API keys are needed
        needed_keys = set()
        if selected_llm in ["gemini"]:
            needed_keys.add("GOOGLE_API_KEY")
        if selected_llm in ["openai"]:
            needed_keys.add("OPENAI_API_KEY")
        if selected_llm in ["anthropic"]:
            needed_keys.add("ANTHROPIC_API_KEY")
        if selected_llm in ["mistral"]:
            needed_keys.add("MISTRAL_API_KEY")
        if selected_llm in ["groq"]:
            needed_keys.add("GROQ_API_KEY")
            
        if selected_embed in ["gemini"]:
            needed_keys.add("GOOGLE_API_KEY")
        if selected_embed in ["openai"]:
            needed_keys.add("OPENAI_API_KEY")
        
        # Prompt for needed API keys
        for key in needed_keys:
            current_value = get_current_env_value(key)
            if current_value:
                console.print(f"{key}: [green]Already configured[/green]")
                if not quick and Confirm.ask(f"Update {key}?", default=False):
                    new_value = Prompt.ask(f"Enter new {key}", password=True)
                    if new_value.strip():
                        config_values[key] = new_value
            else:
                console.print(f"{key}: [red]Not configured[/red]")
                new_value = Prompt.ask(f"Enter {key} (required for {selected_llm}/{selected_embed})", password=True)
                if new_value.strip():
                    config_values[key] = new_value
        
        # 4. Ollama Configuration (if using ollama)
        if selected_llm == "ollama" or selected_embed == "ollama":
            console.print("\n[bold]Step 3: Ollama Configuration[/bold]")
            
            current_host = get_current_env_value("OLLAMA_HOST") or "http://localhost:11434"
            ollama_host = Prompt.ask("Ollama host URL", default=current_host)
            config_values["OLLAMA_HOST"] = ollama_host
            
            if not quick:
                current_model = get_current_env_value("OLLAMA_MODEL") or "llama3.1"
                ollama_model = Prompt.ask("Ollama LLM model", default=current_model)
                config_values["OLLAMA_MODEL"] = ollama_model
                
                current_embed_model = get_current_env_value("OLLAMA_EMBED_MODEL") or "nomic-embed-text"
                ollama_embed = Prompt.ask("Ollama embedding model", default=current_embed_model)
                config_values["OLLAMA_EMBED_MODEL"] = ollama_embed
        
        # 5. Basic Settings
        console.print("\n[bold]Step 4: Basic Settings[/bold]")
        
        # Documentation folder
        current_docs = get_current_env_value("DOCS_FOLDER") or "./docs"
        docs_folder = Prompt.ask("Documentation folder path", default=current_docs)
        config_values["DOCS_FOLDER"] = docs_folder
        
        if not quick:
            # RAG settings
            current_conf = get_current_env_value("ANSWER_MIN_CONF") or "0.25"
            min_conf = Prompt.ask("Minimum confidence threshold (0.0-1.0)", default=current_conf)
            config_values["ANSWER_MIN_CONF"] = min_conf
            
            current_chunks = get_current_env_value("MAX_CHUNKS") or "5"
            max_chunks = Prompt.ask("Maximum chunks to retrieve", default=current_chunks)
            config_values["MAX_CHUNKS"] = max_chunks
            
            # Crawling settings
            current_seeds = get_current_env_value("DEFAULT_SEEDS") or "https://docs.python.org/3/faq/index.html,https://docs.python.org/3/library/venv.html"
            default_seeds = Prompt.ask("Default crawl seeds (comma-separated URLs)", default=current_seeds)
            config_values["DEFAULT_SEEDS"] = default_seeds
            
            current_depth = get_current_env_value("CRAWL_DEPTH") or "1"
            crawl_depth = Prompt.ask("Crawl depth (1-3)", default=current_depth)
            config_values["CRAWL_DEPTH"] = crawl_depth
        
        # 6. Save Configuration
        console.print("\n[bold]Step 5: Save Configuration[/bold]")
        
        # Show summary
        summary_table = Table(title="Configuration Summary")
        summary_table.add_column("Setting", style="cyan")
        summary_table.add_column("Value", style="green")
        
        for key, value in config_values.items():
            # Mask API keys for display
            if "API_KEY" in key and value:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            summary_table.add_row(key, display_value)
        
        console.print(summary_table)
        
        if Confirm.ask("\nSave this configuration?", default=True):
            env_path = ".env"
            write_env_file(config_values, env_path)
            console.print(f"\n[green]Configuration saved to {env_path}[/green]")
            
            # Validate configuration
            console.print("\n[cyan]Validating configuration...[/cyan]")
            
            # Reload environment to pick up new values
            from dotenv import load_dotenv
            load_dotenv(override=True)
            
            warnings = validate_configuration()
            if warnings:
                console.print("[yellow]Configuration warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  - {warning}")
            else:
                console.print("[green]Configuration is valid![/green]")
            
            console.print(f"\n[cyan]Next steps:[/cyan]")
            console.print("1. Run '[bold]python -m src.rag_explorer.cli.main ping[/bold]' to test provider connectivity")
            console.print("2. Run '[bold]python -m src.rag_explorer.cli.main index[/bold]' to index your documentation") 
            console.print("3. Run '[bold]python -m src.rag_explorer.cli.main ask[/bold]' to start asking questions")
        else:
            console.print("[yellow]Configuration cancelled.[/yellow]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Configuration cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Configuration failed: {e}[/red]")
        if ctx.obj.get("verbose"):
            import traceback
            console.print(traceback.format_exc(), style="dim")
        raise click.Abort()