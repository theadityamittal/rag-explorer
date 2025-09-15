"""New unified CLI using shared engine services with modular command structure."""

import click

from .commands import (
    index,
    search,
    ask,
    crawl,
    status,
    metrics,
    ping,
    config,
    configure
)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
@click.pass_context
def cli(ctx, verbose, quiet):
    """Support Deflect Bot - Unified CLI with shared engine services."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


# Register all commands from the modular command structure
cli.add_command(index)
cli.add_command(search)
cli.add_command(ask)
cli.add_command(crawl)
cli.add_command(status)
cli.add_command(metrics)
cli.add_command(ping)
cli.add_command(config)
cli.add_command(configure)


if __name__ == "__main__":
    cli()
