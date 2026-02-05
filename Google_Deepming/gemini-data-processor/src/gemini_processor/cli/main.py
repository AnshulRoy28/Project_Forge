"""Main CLI entry point for Gemini Data Processor."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .. import __version__
from ..config import ConfigurationLoader, KeyringManager
from ..models.config import GeminiConfig
from ..models.enums import FileFormat

console = Console()


def get_file_format(file_path: Path) -> FileFormat:
    """Detect the file format based on extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return FileFormat.CSV
    elif suffix == ".json":
        return FileFormat.JSON
    elif suffix in (".txt", ".text"):
        return FileFormat.TEXT
    return FileFormat.UNKNOWN


def validate_file(file_path: str) -> tuple[bool, str, Optional[FileFormat]]:
    """
    Validate input file exists and has supported format.
    
    Returns:
        (is_valid, error_message, file_format)
    """
    path = Path(file_path)
    
    if not path.exists():
        return False, f"File not found: {file_path}", None
    
    if not path.is_file():
        return False, f"Not a file: {file_path}", None
    
    file_format = get_file_format(path)
    if file_format == FileFormat.UNKNOWN:
        return False, (
            f"Unsupported file format: {path.suffix}\n"
            "Supported formats: CSV, JSON, TXT"
        ), None
    
    return True, "", file_format


@click.group()
@click.version_option(version=__version__, prog_name="Gemini Data Processor")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """
    Gemini Data Processor - AI-powered data analysis and processing.
    
    Use 'gdp init' to set up your API key, then 'gdp process <file>' to analyze data.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


@cli.command()
@click.option(
    "--api-key", 
    prompt=False,
    help="Gemini API key (will prompt if not provided)"
)
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Overwrite existing configuration"
)
@click.pass_context
def init(ctx: click.Context, api_key: Optional[str], force: bool) -> None:
    """Initialize Gemini Data Processor with your API key."""
    quiet = ctx.obj.get("quiet", False)
    
    loader = ConfigurationLoader()
    
    # Check if already initialized
    if loader.is_initialized() and not force:
        if not quiet:
            console.print(
                "[yellow]Already initialized.[/yellow] "
                "Use --force to reconfigure."
            )
        return
    
    if not quiet:
        console.print(Panel.fit(
            "[bold blue]Gemini Data Processor Setup[/bold blue]\n\n"
            "This tool uses Google's Gemini AI to analyze and process your data.\n"
            "All processing happens in isolated Docker containers.",
            title="Welcome",
            border_style="blue"
        ))
        console.print()
    
    # Get API key
    if not api_key:
        console.print("[bold]Step 1:[/bold] Configure Gemini API Key\n")
        console.print(
            "Get your API key from: [link=https://aistudio.google.com/apikey]"
            "https://aistudio.google.com/apikey[/link]\n"
        )
        
        api_key = Prompt.ask(
            "[bold cyan]Enter your Gemini API key[/bold cyan]",
            password=True
        )
    
    # Validate API key format
    if not KeyringManager.validate_api_key_format(api_key):
        console.print("[red]Invalid API key format.[/red] Please check and try again.")
        sys.exit(1)
    
    # Store API key securely
    use_keyring = KeyringManager.is_keyring_available()
    if use_keyring:
        if not quiet:
            console.print("\n[dim]Storing API key in system keyring...[/dim]")
    else:
        if not quiet:
            console.print(
                "\n[yellow]System keyring not available.[/yellow] "
                "API key will be stored in environment variable.\n"
                "Consider setting GEMINI_API_KEY in your shell profile."
            )
    
    KeyringManager.set_api_key(api_key, use_keyring=use_keyring)
    
    # Create configuration directory and save default config
    loader.ensure_config_dir()
    
    config = loader.load()
    config.gemini = GeminiConfig(api_key=api_key)
    loader.save(config)
    
    if not quiet:
        console.print("\n[green]✓ Configuration saved successfully![/green]\n")
        
        # Show configuration summary
        table = Table(title="Configuration Summary", show_header=False)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("API Key", "****" + api_key[-4:])
        table.add_row("Model", config.gemini.model_name)
        table.add_row("Config Directory", str(loader.config_dir))
        table.add_row("Max Container Memory", f"{config.resources.max_container_memory_gb} GB")
        table.add_row("Max Container CPU", f"{config.resources.max_container_cpu_cores} cores")
        
        console.print(table)
        console.print("\n[bold]Next steps:[/bold]")
        console.print("  • Run [cyan]gdp process <file>[/cyan] to analyze a data file")
        console.print("  • Run [cyan]gdp status[/cyan] to check Docker and configuration")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Check the status of Gemini Data Processor configuration."""
    loader = ConfigurationLoader()
    
    console.print(Panel.fit(
        "[bold]System Status[/bold]",
        border_style="blue"
    ))
    console.print()
    
    table = Table(show_header=False)
    table.add_column("Check", style="cyan", width=30)
    table.add_column("Status", width=50)
    
    # Check configuration
    if loader.is_initialized():
        table.add_row("Configuration", "[green]✓ Initialized[/green]")
    else:
        table.add_row("Configuration", "[red]✗ Not initialized[/red] - Run 'gdp init'")
    
    # Check API key
    api_key = KeyringManager.get_api_key()
    if api_key:
        table.add_row("API Key", f"[green]✓ Configured[/green] (****{api_key[-4:]})")
    else:
        table.add_row("API Key", "[red]✗ Not configured[/red]")
    
    # Check Docker
    try:
        import docker
        client = docker.from_env()
        version = client.version()
        docker_version = version.get("Version", "unknown")
        table.add_row("Docker", f"[green]✓ Available[/green] (v{docker_version})")
    except ImportError:
        table.add_row("Docker", "[red]✗ Docker SDK not installed[/red]")
    except Exception as e:
        table.add_row("Docker", f"[red]✗ Not available[/red] - {str(e)[:30]}")
    
    # Check keyring
    if KeyringManager.is_keyring_available():
        table.add_row("Keyring", "[green]✓ Available[/green]")
    else:
        table.add_row("Keyring", "[yellow]~ Not available[/yellow] (using env vars)")
    
    console.print(table)


@cli.command()
@click.argument("file", type=click.STRING)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output directory for processed files"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show analysis without executing scripts"
)
@click.pass_context
def process(
    ctx: click.Context, 
    file: str, 
    output: Optional[str],
    dry_run: bool
) -> None:
    """Process a data file using Gemini AI.
    
    FILE: Path to the data file (use quotes for paths with spaces)
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)
    
    # Check initialization
    loader = ConfigurationLoader()
    if not loader.is_initialized():
        console.print(
            "[red]Not initialized.[/red] "
            "Run [cyan]gdp init[/cyan] first to configure your API key."
        )
        sys.exit(1)
    
    # Validate file
    is_valid, error_msg, file_format = validate_file(file)
    if not is_valid:
        console.print(f"[red]Error:[/red] {error_msg}")
        sys.exit(1)
    
    file_path = Path(file)
    
    if not quiet:
        console.print(Panel.fit(
            f"[bold]Processing:[/bold] {file_path.name}\n"
            f"[bold]Format:[/bold] {file_format.value.upper()}\n"
            f"[bold]Size:[/bold] {file_path.stat().st_size / 1024:.1f} KB",
            title="Data Processing",
            border_style="green"
        ))
    
    # Load configuration
    config = loader.load()
    
    # Import and run the processing engine
    from ..core import ProcessingEngine
    
    engine = ProcessingEngine(config)
    
    try:
        result = engine.process_file(
            input_file=str(file_path),
            output_dir=output,
            dry_run=dry_run,
            verbose=verbose
        )
        
        if result.success:
            if not quiet:
                console.print("\n[green]✓ Processing complete![/green]")
                if result.output_files:
                    console.print("\n[bold]Output files:[/bold]")
                    for f in result.output_files:
                        console.print(f"  • {f}")
        else:
            console.print(f"\n[red]✗ Processing failed:[/red] {result.error_message}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option("--all", "-a", "clear_all", is_flag=True, help="Clear all sessions and cache")
@click.pass_context
def clean(ctx: click.Context, clear_all: bool) -> None:
    """Clean up temporary files and old sessions."""
    loader = ConfigurationLoader()
    
    if not loader.config_dir.exists():
        console.print("[yellow]No configuration directory found.[/yellow]")
        return
    
    if clear_all:
        if Confirm.ask("This will delete all sessions and cache. Continue?"):
            import shutil
            sessions_dir = loader.config_dir / "sessions"
            if sessions_dir.exists():
                shutil.rmtree(sessions_dir)
            console.print("[green]✓ Cleaned all sessions.[/green]")
    else:
        console.print("Use --all to clear all sessions and cache.")


def main() -> None:
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
