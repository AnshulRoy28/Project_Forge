"""
forge init - Initialize Forge environment and configure API key.

Lightweight initialization - Docker containers handle all heavy ML dependencies.
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
    store_api_key, get_api_key, validate_api_key_format,
    store_hf_token, get_hf_token, validate_hf_token_format
)


def init_command(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-initialization"),
):
    """
    Initialize Forge - lightweight setup for Docker-based training.
    
    This command:
    - Scans your hardware (GPU, VRAM, RAM) 
    - Configures and securely stores your Gemini API key
    - Configures and securely stores your HuggingFace token (for gated models)
    - Creates Forge config directory
    
    Note: All ML dependencies (PyTorch, CUDA) are handled by Docker containers.
    """
    print_banner()
    console.print()
    console.print("[bold green]🐳 Docker-based setup - no heavy dependencies needed![/]")
    console.print("[dim]All ML libraries (PyTorch, CUDA) run inside optimized containers.[/]")
    console.print("[dim]Securely stores Gemini API key + HuggingFace token for gated models.[/]")
    console.print()
    
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
    
    # Step 2: Check/Configure API Keys
    print_step(2, 4, "Configuring API credentials...")
    
    # Gemini API Key
    console.print("[bold]🧠 Gemini API Key[/]")
    existing_key = get_api_key()
    
    if existing_key and not force:
        print_success("Gemini API key already configured.")
        masked_key = existing_key[:8] + "..." + existing_key[-4:]
        console.print(f"  [dim]Key: {masked_key}[/]")
        
        if not Confirm.ask("\n[yellow]Update the Gemini API key?[/]", default=False):
            api_key = existing_key
        else:
            api_key = _prompt_for_gemini_api_key()
    else:
        api_key = _prompt_for_gemini_api_key()
    
    console.print()
    
    # HuggingFace Token
    console.print("[bold]🤗 HuggingFace Token[/]")
    existing_token = get_hf_token()
    
    if existing_token and not force:
        print_success("HuggingFace token already configured.")
        masked_token = existing_token[:8] + "..." + existing_token[-4:]
        console.print(f"  [dim]Token: {masked_token}[/]")
        
        if not Confirm.ask("\n[yellow]Update the HuggingFace token?[/]", default=False):
            hf_token = existing_token
        else:
            hf_token = _prompt_for_hf_token()
    else:
        hf_token = _prompt_for_hf_token()
    
    console.print()
    
    # Step 3: Create .forge directory and validate
    print_step(3, 4, "Finalizing setup...")
    
    forge_dir = ensure_forge_dir()
    (forge_dir / "cache").mkdir(exist_ok=True)
    print_success(f"Config directory: {forge_dir}")
    
    # Quick API validation
    print_step(4, 4, "Testing connections...")
    
    if api_key:
        try:
            with spinner("Testing Gemini connection..."):
                _validate_gemini_connection(api_key)
            print_success("Gemini API connected!")
        except Exception as e:
            print_warning(f"Gemini API test failed: {e}")
    
    if hf_token:
        try:
            with spinner("Testing HuggingFace connection..."):
                _validate_hf_connection(hf_token)
            print_success("HuggingFace authenticated!")
        except Exception as e:
            print_warning(f"HuggingFace test failed: {e}")
            console.print("[dim]  This is optional - you can still use open models[/]")
    
    # Summary
    elapsed = time.time() - start_time
    console.print()
    console.print(f"[bold green]✨ Forge initialized in {elapsed:.1f}s[/]")
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
    console.print("  4. Start training: [cyan]forge train[/] (authentication handled automatically)")
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


def _prompt_for_gemini_api_key() -> str:
    """Prompt for and store Gemini API key."""
    console.print()
    console.print("[bold]Enter your Gemini API key[/]")
    console.print("[dim]Get one at: https://aistudio.google.com/apikey[/]")
    console.print()
    
    while True:
        api_key = Prompt.ask("Gemini API Key", password=True)
        
        if not api_key:
            print_error("API key cannot be empty.")
            continue
        
        if not validate_api_key_format(api_key):
            print_error("Invalid API key format.")
            continue
        
        if store_api_key(api_key):
            print_success("Gemini API key stored securely.")
            return api_key
        else:
            print_error("Failed to store API key.")
            raise typer.Exit(1)


def _prompt_for_hf_token() -> str:
    """Prompt for and store HuggingFace token."""
    console.print()
    console.print("[bold]Enter your HuggingFace token[/]")
    console.print("[dim]Get one at: https://huggingface.co/settings/tokens[/]")
    console.print("[dim]Required for gated models like google/gemma-7b-it[/]")
    console.print()
    
    # Allow skipping HuggingFace token
    if not Confirm.ask("[yellow]Do you want to configure HuggingFace authentication?[/]", default=True):
        console.print("[dim]Skipping HuggingFace token - you can only use open models[/]")
        return ""
    
    while True:
        hf_token = Prompt.ask("HuggingFace Token", password=True)
        
        if not hf_token:
            if Confirm.ask("[yellow]Skip HuggingFace token?[/]", default=False):
                console.print("[dim]Skipping HuggingFace token - you can only use open models[/]")
                return ""
            continue
        
        if not validate_hf_token_format(hf_token):
            print_error("Invalid HuggingFace token format.")
            console.print("[dim]Tokens should start with 'hf_' and be 37+ characters[/]")
            continue
        
        if store_hf_token(hf_token):
            print_success("HuggingFace token stored securely.")
            return hf_token
        else:
            print_error("Failed to store HuggingFace token.")
            raise typer.Exit(1)


def _validate_gemini_connection(api_key: str) -> None:
    """Test Gemini API connection."""
    from google import genai
    
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Say 'ready' in one word.",
    )
    
    if not response.text:
        raise ValueError("Empty response")


def _validate_hf_connection(token: str) -> None:
    """Test HuggingFace API connection."""
    try:
        from huggingface_hub import HfApi
        
        api = HfApi(token=token)
        # Try to get user info - this will fail if token is invalid
        user_info = api.whoami()
        
        if not user_info:
            raise ValueError("Invalid token")
            
    except ImportError:
        # huggingface_hub not installed, skip validation
        console.print("[dim]  huggingface_hub not installed - skipping validation[/]")
        return
    except Exception as e:
        raise ValueError(f"Authentication failed: {e}")
