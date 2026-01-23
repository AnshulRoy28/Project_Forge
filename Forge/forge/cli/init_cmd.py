"""
forge init - Initialize Forge environment and configure session credentials.

Session-based security - API keys are stored only in memory for the current session.
"""

import time
from pathlib import Path

import typer
from rich.prompt import Prompt, Confirm

from forge.ui.console import console, print_banner, print_success, print_error, print_step, print_warning
from forge.ui.panels import create_hardware_panel, display_panel
from forge.ui.progress import spinner
from forge.core.config import ensure_forge_dir
from forge.core.security import (
    prompt_for_credentials, clear_all_credentials, has_credentials
)


def init_command(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-initialization"),
):
    """
    Initialize Forge - lightweight setup for Docker-based training.
    
    This command:
    - Scans your hardware (GPU, VRAM, RAM) 
    - Prompts for session-based API credentials (Gemini + HuggingFace)
    - Creates Forge config directory
    
    Security: API keys are stored only in memory for the current session.
    They are automatically cleared when the session ends.
    
    Note: All ML dependencies (PyTorch, CUDA) are handled by Docker containers.
    """
    print_banner()
    console.print()
    console.print("[bold green]🐳 Docker-based setup - no heavy dependencies needed![/]")
    console.print("[bold yellow]🔒 Session-based security - credentials cleared automatically![/]")
    console.print("[dim]All ML libraries (PyTorch, CUDA) run inside optimized containers.[/]")
    console.print("[dim]API keys stored only in memory for current session.[/]")
    console.print()
    
    # Clear any existing credentials if force
    if force:
        clear_all_credentials()
        console.print("[yellow]Cleared existing session credentials.[/]")
    
    # Run lightweight initialization
    _run_init(force)


def _run_init(force: bool):
    """Run the lightweight initialization (hardware scan, API key, config setup)."""
    from forge.core.hardware import detect_hardware
    
    start_time = time.time()
    
    # Step 1: Hardware Detection
    print_step(1, 3, "Scanning hardware...")
    
    with spinner("Detecting GPU and system specs..."):
        hardware = detect_hardware()
    
    # Display hardware panel
    if hardware.gpu:
        display_panel(create_hardware_panel(
            gpu_name=hardware.gpu.name,
            vram_total=hardware.gpu.vram_total_gb,
            vram_free=hardware.gpu.vram_free_gb,
            cuda_version=hardware.gpu.cuda_version,
            driver_version=hardware.gpu.driver_version,
            ram_total=hardware.system.ram_total_gb,
            ram_free=hardware.system.ram_free_gb,
            cpu_name=hardware.system.cpu_name,
        ))
        
        # Show which Docker container will be used
        arch = _get_gpu_architecture(hardware.gpu.name)
        console.print(f"[bold green]✓[/] Docker container: [cyan]forge:{arch}[/]")
    else:
        console.print("[yellow]⚠️  No CUDA GPU detected.[/]")
        console.print(f"  CPU: {hardware.system.cpu_name}")
        console.print(f"  RAM: {hardware.system.ram_total_gb:.1f} GB")
        console.print()
        console.print("[dim]Training will be very slow without a GPU.[/]")
        console.print("[dim]Docker container: forge:base (CPU-only)[/]")
    
    console.print()
    
    # Step 2: Session Credentials
    print_step(2, 3, "Configuring session credentials...")
    
    # Check if credentials already exist in session
    has_gemini, has_hf = has_credentials()
    
    if has_gemini and has_hf and not force:
        console.print("[green]✓ Session credentials already configured.[/]")
    else:
        # Prompt for credentials
        got_gemini, got_hf = prompt_for_credentials(
            console, 
            force_gemini=True,  # Always require Gemini
            force_hf=False      # HuggingFace is optional
        )
        
        if not got_gemini:
            print_error("Gemini API key is required for Forge to function.")
            raise typer.Exit(1)
    
    console.print()
    
    # Step 3: Create .forge directory
    print_step(3, 3, "Finalizing setup...")
    
    forge_dir = ensure_forge_dir()
    (forge_dir / "cache").mkdir(exist_ok=True)
    print_success(f"Config directory: {forge_dir}")
    
    # Summary
    elapsed = time.time() - start_time
    console.print()
    console.print(f"[bold green]✨ Forge initialized in {elapsed:.1f}s[/]")
    console.print()
    
    # Security reminder
    console.print("[bold yellow]🔒 Security Notice:[/]")
    console.print("  • API keys stored only in memory for this session")
    console.print("  • Credentials automatically cleared when session ends")
    console.print("  • No persistent storage of sensitive data")
    console.print()
    
    # Show recommendations
    if hardware.gpu:
        config = hardware.recommend_config()
        console.print("[bold]Recommended setup for your hardware:[/]")
        console.print(f"  • Model: [cyan]{config.model_recommendation}[/]")
        console.print(f"  • Quantization: [cyan]{config.quantization or 'Full precision'}[/]")
        console.print(f"  • Batch Size: [cyan]{config.batch_size}[/]")
    
    console.print()
    console.print("[dim]Next steps:[/]")
    console.print("  1. Build Docker container: [cyan]forge docker build[/]")
    console.print("  2. Generate training plan: [cyan]forge plan \"your goal\"[/]")
    console.print("  3. Prepare your dataset: [cyan]forge prepare ./data/your_data.csv[/]")
    console.print("  4. Start training: [cyan]forge train[/] (will use session credentials)")
    console.print()


def _get_gpu_architecture(gpu_name: str) -> str:
    """Determine Docker container architecture from GPU name."""
    gpu_lower = gpu_name.lower()
    
    if "rtx 50" in gpu_lower or "5090" in gpu_lower or "5080" in gpu_lower or "5070" in gpu_lower:
        return "blackwell"
    elif "rtx 40" in gpu_lower or "4090" in gpu_lower or "4080" in gpu_lower or "4070" in gpu_lower:
        return "ada"
    elif "rtx 30" in gpu_lower or "3090" in gpu_lower or "3080" in gpu_lower or "3070" in gpu_lower:
        return "ampere"
    elif "h100" in gpu_lower or "h200" in gpu_lower:
        return "hopper"
    else:
        return "base"
