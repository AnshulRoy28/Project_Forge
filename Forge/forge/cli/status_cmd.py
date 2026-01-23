"""
forge status - Display project state, hardware, and configuration health.
"""

from pathlib import Path

import typer
from rich.table import Table
from rich.panel import Panel

from forge.ui.console import console, print_success, print_error, print_info, print_warning
from forge.core.config import load_config, get_config_path


def status_command():
    """
    Display current project state, detected hardware, and configuration health.
    """
    console.print("\n[bold]📊 Forge Status[/]\n")
    
    # Project Config
    _show_config_status()
    
    # Hardware Detection
    _show_hardware_status()
    
    # Docker Status
    _show_docker_status()
    
    # Credentials
    _show_credentials_status()
    
    console.print()


def _show_config_status():
    """Show configuration file status."""
    console.print("[bold]Configuration[/]")
    
    config_path = get_config_path()
    
    if config_path.exists():
        try:
            config = load_config(config_path)
            console.print(f"  forge.yaml: [green]✓ Found[/]")
            console.print(f"  Project: [cyan]{config.name}[/]")
            console.print(f"  Model: [cyan]{config.training.base_model}[/]")
            
            if config.data and config.data.path:
                data_path = Path(config.data.path)
                if data_path.exists() or config.data.path.startswith("/"):
                    console.print(f"  Data: [cyan]{config.data.path}[/]")
                else:
                    console.print(f"  Data: [yellow]⚠ Not found: {config.data.path}[/]")
        except Exception as e:
            console.print(f"  forge.yaml: [red]✗ Invalid - {e}[/]")
    else:
        console.print(f"  forge.yaml: [yellow]Not found[/]")
        console.print(f"  [dim]Run 'forge plan' to generate[/]")
    
    console.print()


def _show_hardware_status():
    """Show detected hardware."""
    console.print("[bold]Hardware[/]")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            console.print(f"  GPU: [green]✓ {gpu_name}[/]")
            console.print(f"  VRAM: [cyan]{vram:.1f} GB[/]")
            console.print(f"  CUDA: [cyan]{torch.version.cuda}[/]")
        else:
            console.print(f"  GPU: [yellow]Not detected[/]")
            console.print(f"  [dim]Training will use Docker[/]")
    except ImportError:
        console.print(f"  PyTorch: [dim]Not installed locally[/]")
        console.print(f"  [dim]Hardware will be detected in Docker[/]")
    
    console.print()


def _show_docker_status():
    """Show Docker container status."""
    import subprocess
    
    console.print("[bold]Docker[/]")
    
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            console.print(f"  Docker: [green]✓ Running[/]")
            
            # Check for Forge images
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "forge"],
                capture_output=True,
                text=True,
            )
            images = [img for img in result.stdout.strip().split("\n") if img.startswith("forge:")]
            
            if images:
                console.print(f"  Images: [cyan]{', '.join(images)}[/]")
            else:
                console.print(f"  Images: [yellow]None built[/]")
                console.print(f"  [dim]Run 'forge docker build' first[/]")
        else:
            console.print(f"  Docker: [red]✗ Not running[/]")
    except FileNotFoundError:
        console.print(f"  Docker: [red]✗ Not installed[/]")
    except subprocess.TimeoutExpired:
        console.print(f"  Docker: [yellow]⚠ Not responding[/]")
    
    console.print()


def _show_credentials_status():
    """Show credentials status."""
    import os
    from forge.core.config import ensure_forge_dir
    
    console.print("[bold]Credentials[/]")
    
    forge_dir = ensure_forge_dir()
    creds_file = forge_dir / "credentials"
    
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    hf_token = os.getenv("HF_TOKEN", "")
    
    # Check credentials file
    if creds_file.exists():
        with open(creds_file, 'r') as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY=") and not gemini_key:
                    gemini_key = line.strip().split("=", 1)[1]
                elif line.startswith("HF_TOKEN=") and not hf_token:
                    hf_token = line.strip().split("=", 1)[1]
    
    if gemini_key:
        masked = gemini_key[:8] + "..." + gemini_key[-4:]
        console.print(f"  Gemini: [green]✓ Configured[/] ({masked})")
    else:
        console.print(f"  Gemini: [yellow]Not configured[/]")
        console.print(f"  [dim]Run 'forge login' to add[/]")
    
    if hf_token:
        console.print(f"  HuggingFace: [green]✓ Configured[/]")
    else:
        console.print(f"  HuggingFace: [dim]Not configured (optional)[/]")
