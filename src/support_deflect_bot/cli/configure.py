"""Interactive configuration command for Support Deflect Bot."""

import os
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel

from ..config import ConfigurationManager, AppConfig

console = Console()


@click.command()
@click.option("--reset", is_flag=True, help="Reset to default configuration")
@click.option("--file", "config_file", help="Use specific configuration file")
@click.pass_context
def configure(ctx, reset: bool, config_file: Optional[str]):
    """Interactive configuration setup for Support Deflect Bot."""

    try:
        # Initialize configuration manager
        manager = (
            ConfigurationManager(config_file) if config_file else ConfigurationManager()
        )

        if reset:
            if Confirm.ask("🔄 Reset to default configuration?", default=False):
                # Create new default config
                config = AppConfig()
                manager.save_config(config)
                console.print("✅ Configuration reset to defaults", style="green")
                return
            else:
                console.print("❌ Reset cancelled", style="yellow")
                return

        console.print(
            Panel.fit(
                "🛠️  Support Deflect Bot Configuration\n\n"
                "This interactive setup will configure your Support Deflect Bot.\n"
                "Press Enter to keep current values, or type new values to change them.",
                title="Configuration Setup",
            )
        )

        # Load current configuration
        current_config = manager.load_config()

        # Show current config file location
        console.print(f"📁 Configuration file: [cyan]{manager.config_file}[/cyan]\n")

        # Configure sections
        new_config_data = current_config.dict()

        # 1. API Keys
        _configure_api_keys(new_config_data)

        # 2. Documentation Settings
        _configure_docs(new_config_data)

        # 3. RAG Settings
        _configure_rag(new_config_data)

        # 4. Crawl Settings
        _configure_crawl(new_config_data)

        # 5. Model Overrides (optional)
        if Confirm.ask("\n🔧 Configure model overrides?", default=False):
            _configure_model_overrides(new_config_data)

        # Create and validate new configuration
        new_config = AppConfig(**new_config_data)

        # Show summary
        _show_configuration_summary(new_config)

        # Confirm and save
        if Confirm.ask("\n💾 Save this configuration?", default=True):
            manager.save_config(new_config)
            console.print("✅ Configuration saved successfully!", style="green")

            # Offer to export .env file
            if Confirm.ask("📄 Export as .env file?", default=False):
                env_path = Prompt.ask("Enter .env file path", default=".env")
                manager.export_env_file(env_path)
                console.print(f"✅ Configuration exported to {env_path}", style="green")

            # Validation check
            validation = manager.validate_config()
            if validation["warnings"]:
                console.print("\n⚠️  Configuration warnings:", style="yellow")
                for warning in validation["warnings"]:
                    console.print(f"  • {warning}", style="yellow")
        else:
            console.print("❌ Configuration not saved", style="yellow")

    except KeyboardInterrupt:
        console.print("\n\n⚠️  Configuration cancelled", style="yellow")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ Configuration failed: {e}", style="red")
        sys.exit(1)


def _configure_api_keys(config_data: dict):
    """Configure API keys section."""
    console.print("\n[bold cyan]🔑 API Keys Configuration[/bold cyan]")
    console.print("Configure API keys for different providers. Leave blank to skip.")

    api_keys = config_data.setdefault("api_keys", {})

    # Google Gemini (Primary)
    current_google = api_keys.get("google_api_key", "")
    display_google = (
        f"[green]✓ Configured[/green]" if current_google else "[red]✗ Not set[/red]"
    )
    console.print(f"Google Gemini (Primary): {display_google}")

    new_google = Prompt.ask(
        "Google API Key",
        default=current_google[:10] + "..." if current_google else "",
        password=True,
    )
    if new_google and new_google != current_google[:10] + "...":
        api_keys["google_api_key"] = new_google

    # OpenAI (Fallback)
    current_openai = api_keys.get("openai_api_key", "")
    display_openai = (
        f"[green]✓ Configured[/green]" if current_openai else "[red]✗ Not set[/red]"
    )
    console.print(f"OpenAI (Fallback): {display_openai}")

    new_openai = Prompt.ask(
        "OpenAI API Key",
        default=current_openai[:10] + "..." if current_openai else "",
        password=True,
    )
    if new_openai and new_openai != current_openai[:10] + "...":
        api_keys["openai_api_key"] = new_openai

    # Other providers (optional)
    if Confirm.ask(
        "Configure additional providers (Anthropic, Groq, Mistral)?", default=False
    ):
        providers = [
            ("anthropic_api_key", "Anthropic API Key"),
            ("groq_api_key", "Groq API Key"),
            ("mistral_api_key", "Mistral API Key"),
        ]

        for key, label in providers:
            current = api_keys.get(key, "")
            display = f"[green]✓ Set[/green]" if current else "[red]✗ Not set[/red]"
            console.print(f"{label}: {display}")

            new_key = Prompt.ask(
                label, default=current[:10] + "..." if current else "", password=True
            )
            if new_key and new_key != current[:10] + "...":
                api_keys[key] = new_key


def _configure_docs(config_data: dict):
    """Configure documentation settings."""
    console.print("\n[bold cyan]📚 Documentation Configuration[/bold cyan]")

    docs = config_data.setdefault("docs", {})

    current_path = docs.get("local_path", "./docs")
    console.print(f"Current docs path: [cyan]{current_path}[/cyan]")

    new_path = Prompt.ask("Documentation folder path", default=current_path)
    if new_path != current_path:
        # Validate path
        if not os.path.exists(new_path):
            if Confirm.ask(f"Path {new_path} doesn't exist. Create it?", default=True):
                Path(new_path).mkdir(parents=True, exist_ok=True)
                console.print(f"✅ Created directory: {new_path}", style="green")
            else:
                console.print("⚠️  Using non-existent path", style="yellow")

        docs["local_path"] = new_path

    auto_refresh = docs.get("auto_refresh", True)
    docs["auto_refresh"] = Confirm.ask(
        "Auto-refresh documentation", default=auto_refresh
    )


def _configure_rag(config_data: dict):
    """Configure RAG settings."""
    console.print("\n[bold cyan]🧠 RAG Configuration[/bold cyan]")
    console.print("Configure Retrieval-Augmented Generation parameters.")

    rag = config_data.setdefault("rag", {})

    # Confidence threshold
    current_conf = rag.get("confidence_threshold", 0.25)
    console.print(f"Current confidence threshold: [cyan]{current_conf}[/cyan]")
    console.print(
        "Lower values = more answers (less selective), Higher values = fewer answers (more selective)"
    )

    while True:
        conf_input = Prompt.ask(
            "Confidence threshold (0.0-1.0)", default=str(current_conf)
        )
        try:
            new_conf = float(conf_input)
            if 0.0 <= new_conf <= 1.0:
                rag["confidence_threshold"] = new_conf
                break
            else:
                console.print("⚠️  Must be between 0.0 and 1.0", style="yellow")
        except ValueError:
            console.print("⚠️  Invalid number", style="yellow")

    # Max chunks
    current_chunks = rag.get("max_chunks", 5)
    while True:
        chunks_input = Prompt.ask(
            "Maximum chunks to retrieve (1-20)", default=str(current_chunks)
        )
        try:
            new_chunks = int(chunks_input)
            if 1 <= new_chunks <= 20:
                rag["max_chunks"] = new_chunks
                break
            else:
                console.print("⚠️  Must be between 1 and 20", style="yellow")
        except ValueError:
            console.print("⚠️  Invalid number", style="yellow")

    # Max chars per chunk
    current_chars = rag.get("max_chars_per_chunk", 800)
    while True:
        chars_input = Prompt.ask(
            "Maximum characters per chunk (100-5000)", default=str(current_chars)
        )
        try:
            new_chars = int(chars_input)
            if 100 <= new_chars <= 5000:
                rag["max_chars_per_chunk"] = new_chars
                break
            else:
                console.print("⚠️  Must be between 100 and 5000", style="yellow")
        except ValueError:
            console.print("⚠️  Invalid number", style="yellow")


def _configure_crawl(config_data: dict):
    """Configure crawl settings."""
    console.print("\n[bold cyan]🕷️  Crawl Configuration[/bold cyan]")
    console.print("Configure web crawling parameters.")

    crawl = config_data.setdefault("crawl", {})

    # Allow hosts
    current_hosts = crawl.get("allow_hosts", ["docs.python.org"])
    console.print(f"Current allowed hosts: [cyan]{', '.join(current_hosts)}[/cyan]")

    hosts_input = Prompt.ask(
        "Allowed hosts (comma-separated)", default=",".join(current_hosts)
    )
    crawl["allow_hosts"] = [h.strip() for h in hosts_input.split(",") if h.strip()]

    # Trusted domains (subset of allowed hosts)
    current_trusted = crawl.get("trusted_domains", ["docs.python.org"])
    console.print(f"Current trusted domains: [cyan]{', '.join(current_trusted)}[/cyan]")

    trusted_input = Prompt.ask(
        "Trusted domains (subset of allowed hosts)", default=",".join(current_trusted)
    )
    crawl["trusted_domains"] = [
        d.strip() for d in trusted_input.split(",") if d.strip()
    ]

    # Default seeds
    current_seeds = crawl.get(
        "default_seeds", ["https://docs.python.org/3/faq/index.html"]
    )
    console.print("Configure default seed URLs:")

    seeds = []
    for i, seed in enumerate(current_seeds[:3]):  # Show first 3
        new_seed = Prompt.ask(f"Seed URL {i+1}", default=seed)
        if new_seed.strip():
            seeds.append(new_seed.strip())

    # Option to add more
    while len(seeds) < 5 and Confirm.ask("Add another seed URL?", default=False):
        seed = Prompt.ask("Seed URL")
        if seed.strip():
            seeds.append(seed.strip())

    crawl["default_seeds"] = seeds

    # Crawl depth
    current_depth = crawl.get("depth", 1)
    while True:
        depth_input = Prompt.ask("Crawl depth (1-5)", default=str(current_depth))
        try:
            new_depth = int(depth_input)
            if 1 <= new_depth <= 5:
                crawl["depth"] = new_depth
                break
            else:
                console.print("⚠️  Must be between 1 and 5", style="yellow")
        except ValueError:
            console.print("⚠️  Invalid number", style="yellow")

    # Max pages
    current_max = crawl.get("max_pages", 40)
    while True:
        max_input = Prompt.ask(
            "Maximum pages per crawl (1-500)", default=str(current_max)
        )
        try:
            new_max = int(max_input)
            if 1 <= new_max <= 500:
                crawl["max_pages"] = new_max
                break
            else:
                console.print("⚠️  Must be between 1 and 500", style="yellow")
        except ValueError:
            console.print("⚠️  Invalid number", style="yellow")

    # Same domain restriction
    current_same_domain = crawl.get("same_domain", True)
    crawl["same_domain"] = Confirm.ask(
        "Restrict crawling to same domain", default=current_same_domain
    )


def _configure_model_overrides(config_data: dict):
    """Configure model overrides."""
    console.print("\n[bold cyan]🤖 Model Overrides[/bold cyan]")
    console.print("Override default models (requires corresponding API keys).")

    overrides = config_data.setdefault("model_overrides", {})

    model_configs = [
        ("gemini_llm_model", "Gemini LLM Model", "gemini-2.5-flash-lite"),
        ("gemini_embedding_model", "Gemini Embedding Model", "gemini-embedding-001"),
        ("openai_llm_model", "OpenAI LLM Model", "gpt-4o-mini"),
        ("openai_embedding_model", "OpenAI Embedding Model", "text-embedding-3-small"),
    ]

    for key, label, default in model_configs:
        current = overrides.get(key, "")
        display_current = current if current else f"[dim]{default} (default)[/dim]"
        console.print(f"Current {label}: {display_current}")

        new_model = Prompt.ask(f"{label} (empty for default)", default=current)
        if new_model.strip():
            overrides[key] = new_model.strip()
        elif current:
            # Clear if user enters empty but had a value
            overrides[key] = None


def _show_configuration_summary(config: AppConfig):
    """Show configuration summary table."""
    console.print("\n[bold cyan]📋 Configuration Summary[/bold cyan]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", width=30)
    table.add_column("Value", style="green")

    # API Keys
    table.add_row(
        "🔑 Google API Key",
        "✓ Configured" if config.api_keys.google_api_key else "✗ Not set",
    )
    table.add_row(
        "🔑 OpenAI API Key",
        "✓ Configured" if config.api_keys.openai_api_key else "✗ Not set",
    )

    # Docs
    table.add_row("📚 Docs Path", config.docs.local_path)
    table.add_row("📚 Auto Refresh", "Yes" if config.docs.auto_refresh else "No")

    # RAG
    table.add_row("🧠 Confidence Threshold", str(config.rag.confidence_threshold))
    table.add_row("🧠 Max Chunks", str(config.rag.max_chunks))
    table.add_row("🧠 Max Chars/Chunk", str(config.rag.max_chars_per_chunk))

    # Crawl
    table.add_row("🕷️ Allowed Hosts", f"{len(config.crawl.allow_hosts)} hosts")
    table.add_row("🕷️ Trusted Domains", f"{len(config.crawl.trusted_domains)} domains")
    table.add_row("🕷️ Default Seeds", f"{len(config.crawl.default_seeds)} URLs")
    table.add_row("🕷️ Crawl Depth", str(config.crawl.depth))
    table.add_row("🕷️ Max Pages", str(config.crawl.max_pages))
    table.add_row("🕷️ Same Domain Only", "Yes" if config.crawl.same_domain else "No")

    console.print(table)
