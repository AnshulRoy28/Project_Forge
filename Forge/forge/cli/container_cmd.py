"""
forge container - Manage persistent Docker containers for model caching.
"""

import typer
from pathlib import Path

from forge.ui.console import console, print_success, print_error, print_info, print_warning
from forge.ui.panels import Panel
from forge.core.config_v2 import ForgeConfig


app = typer.Typer(
    name="container",
    help="🐳 Manage persistent containers for model caching",
    no_args_is_help=True,
)


@app.command()
def list():
    """List all Forge containers."""
    console.print("\n[bold]🐳 All Forge Containers[/]\n")
    
    try:
        import subprocess
        
        # Find all containers with forge- prefix
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=forge-", "--format", "table {{.ID}}\\t{{.Names}}\\t{{.Status}}\\t{{.Image}}\\t{{.CreatedAt}}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                lines = output.split('\n')
                if len(lines) > 1:  # Has header + data
                    console.print(output)
                else:
                    console.print("  [dim]No Forge containers found[/]")
            else:
                console.print("  [dim]No Forge containers found[/]")
        else:
            print_error("Failed to list containers")
        
        console.print()
        console.print("[dim]Use 'forge container status' to see details for the current project[/]")
        console.print("[dim]Use 'forge container cleanup' to remove the current project's container[/]")
        
    except Exception as e:
        print_error(f"Error listing containers: {e}")
        raise typer.Exit(1)


@app.command()
def status():
    """Show status of persistent containers."""
    console.print("\n[bold]🐳 Container Status[/]\n")
    
    # Load config
    config_file = Path("forge.yaml")
    if not config_file.exists():
        print_error("No forge.yaml found. Run 'forge plan' first.")
        raise typer.Exit(1)
    
    try:
        config = ForgeConfig.load(config_file)
        
        from forge.core.container_manager import ContainerManager
        container_mgr = ContainerManager(config)
        
        info = container_mgr.get_container_info()
        
        if info["status"] == "none":
            console.print("  [dim]No persistent container configured[/]")
            console.print("  [dim]Run 'forge train' to create one automatically[/]")
        else:
            console.print(f"  Container ID: [cyan]{info.get('id', 'unknown')}[/]")
            console.print(f"  Name: [cyan]{info.get('name', 'unknown')}[/]")
            console.print(f"  Status: [{'green' if info['status'] == 'running' else 'yellow'}]{info['status']}[/]")
            console.print(f"  Image: [cyan]{info.get('image', 'unknown')}[/]")
            console.print(f"  Model Cached: [{'green' if info.get('model_cached') else 'yellow'}]{info.get('model_cached', False)}[/]")
            
            # Show which model is cached
            if info.get('model_cached') and config.container.cached_model_name:
                console.print(f"  Cached Model: [cyan]{config.container.cached_model_name}[/]")
            
            if info.get('created'):
                console.print(f"  Created: [dim]{info['created'][:19]}[/]")
            if info.get('started'):
                console.print(f"  Started: [dim]{info['started'][:19]}[/]")
        
        console.print()
        
    except Exception as e:
        print_error(f"Error checking container status: {e}")
        raise typer.Exit(1)


@app.command()
def cleanup(
    force: bool = typer.Option(False, "--force", "-f", help="Force cleanup without confirmation")
):
    """Clean up persistent containers."""
    console.print("\n[bold]🧹 Container Cleanup[/]\n")
    
    # Load config
    config_file = Path("forge.yaml")
    if not config_file.exists():
        print_error("No forge.yaml found.")
        raise typer.Exit(1)
    
    try:
        config = ForgeConfig.load(config_file)
        
        if not config.container.container_id:
            print_info("No persistent container to clean up.")
            return
        
        from forge.core.container_manager import ContainerManager
        container_mgr = ContainerManager(config)
        
        info = container_mgr.get_container_info()
        
        console.print(f"  Container: [cyan]{info.get('id', 'unknown')}[/]")
        console.print(f"  Status: [cyan]{info.get('status', 'unknown')}[/]")
        console.print(f"  Model Cached: [cyan]{info.get('model_cached', False)}[/]")
        
        # Show which model will be lost
        if info.get('model_cached') and config.container.cached_model_name:
            console.print(f"  Cached Model: [yellow]{config.container.cached_model_name}[/] (will be lost)")
        
        console.print()
        
        if not force:
            from rich.prompt import Confirm
            if not Confirm.ask("[yellow]Remove this container?[/]", default=False):
                print_info("Cleanup cancelled.")
                return
        
        container_mgr.cleanup_container()
        
        # Save updated config
        config.save(config_file)
        
        print_success("Container cleaned up successfully!")
        console.print("[dim]Next training run will create a fresh container.[/]")
        
    except Exception as e:
        print_error(f"Error during cleanup: {e}")
        raise typer.Exit(1)


@app.command()
def cache(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-caching even if already cached")
):
    """Pre-cache model in persistent container."""
    console.print("\n[bold]📦 Model Caching[/]\n")
    
    # Load config
    config_file = Path("forge.yaml")
    if not config_file.exists():
        print_error("No forge.yaml found. Run 'forge plan' first.")
        raise typer.Exit(1)
    
    try:
        config = ForgeConfig.load(config_file)
        
        from forge.core.container_manager import ContainerManager
        container_mgr = ContainerManager(config)
        
        # Start container if needed
        container_id = container_mgr.start_persistent_container()
        
        if not container_id:
            print_error("Failed to start container")
            raise typer.Exit(1)
        
        # Check if already cached
        if config.container.model_cached and not force:
            print_info("Model already cached in container.")
            console.print("[dim]Use --force to re-cache.[/]")
            return
        
        # Cache the model
        console.print(f"[bold]Caching model: [cyan]{config.model.name}[/]")
        console.print("[dim]This may take 2-5 minutes but will speed up future training...[/]")
        console.print()
        
        if container_mgr.cache_model_in_container():
            # Save updated config
            config.save(config_file)
            print_success("Model cached successfully!")
            console.print("[dim]Future training runs will start much faster.[/]")
        else:
            print_error("Failed to cache model")
            raise typer.Exit(1)
        
    except Exception as e:
        print_error(f"Error caching model: {e}")
        raise typer.Exit(1)


@app.command()
def logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show")
):
    """Show container logs."""
    console.print("\n[bold]📋 Container Logs[/]\n")
    
    # Load config
    config_file = Path("forge.yaml")
    if not config_file.exists():
        print_error("No forge.yaml found.")
        raise typer.Exit(1)
    
    try:
        config = ForgeConfig.load(config_file)
        
        if not config.container.container_id:
            print_info("No persistent container configured.")
            return
        
        import subprocess
        
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), config.container.container_id],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            if result.stdout:
                console.print(result.stdout)
            if result.stderr:
                console.print(f"[red]{result.stderr}[/]")
        else:
            print_error("Failed to get container logs")
        
    except Exception as e:
        print_error(f"Error getting logs: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()