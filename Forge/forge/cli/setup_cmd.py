"""
forge setup - Create virtual environment and install dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

import typer

from forge.ui.console import console, print_success, print_error, print_info, print_step, print_warning
from forge.ui.progress import spinner


# Default venv directory name
VENV_DIR = ".forge-venv"


def setup_command(
    venv_path: Path = typer.Option(None, "--venv", "-v", help="Custom venv path"),
    with_unsloth: bool = typer.Option(True, "--unsloth/--no-unsloth", help="Install Unsloth for faster training"),
    force: bool = typer.Option(False, "--force", "-f", help="Recreate venv if exists"),
    skip_torch: bool = typer.Option(False, "--skip-torch", help="Skip PyTorch (if already installed)"),
):
    """
    Create a virtual environment and install all Forge dependencies.
    
    This ensures a clean, isolated environment for training.
    """
    console.print("\n[bold]🔧 Forge Environment Setup[/]\n")
    
    # Determine venv path
    if venv_path is None:
        venv_path = Path.cwd() / VENV_DIR
    
    venv_path = venv_path.resolve()
    
    # Check if venv exists
    if venv_path.exists():
        if force:
            print_warning(f"Removing existing venv: {venv_path}")
            import shutil
            shutil.rmtree(venv_path)
        else:
            print_info(f"Virtual environment already exists: {venv_path}")
            if _is_venv_valid(venv_path):
                console.print("[green]✓ Environment is valid[/]")
                console.print(f"\n[dim]To activate:[/]")
                _print_activation_command(venv_path)
                return
            else:
                print_warning("Environment appears corrupted. Use --force to recreate.")
                raise typer.Exit(1)
    
    # Step 1: Create virtual environment
    print_step(1, 4, "Creating virtual environment...")
    
    try:
        with spinner(f"Creating venv at {venv_path}..."):
            _create_venv(venv_path)
        print_success(f"Created: {venv_path}")
    except Exception as e:
        print_error(f"Failed to create venv: {e}")
        raise typer.Exit(1)
    
    # Get pip path in venv
    pip_path = _get_pip_path(venv_path)
    python_path = _get_python_path(venv_path)
    
    # Step 2: Upgrade pip
    print_step(2, 4, "Upgrading pip...")
    
    try:
        with spinner("Upgrading pip..."):
            _run_pip(pip_path, ["install", "--upgrade", "pip"])
        print_success("pip upgraded")
    except Exception as e:
        print_warning(f"pip upgrade failed (continuing): {e}")
    
    # Step 3: Install PyTorch with CUDA
    if not skip_torch:
        print_step(3, 4, "Installing PyTorch with CUDA support...")
        
        try:
            with spinner("Installing PyTorch (this may take a few minutes)..."):
                # Install PyTorch with CUDA 12.1 support
                _run_pip(pip_path, [
                    "install", 
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu121"
                ])
            print_success("PyTorch installed with CUDA support")
        except Exception as e:
            print_warning(f"PyTorch CUDA install failed, trying CPU version: {e}")
            try:
                _run_pip(pip_path, ["install", "torch", "torchvision", "torchaudio"])
                print_success("PyTorch installed (CPU only)")
            except Exception as e2:
                print_error(f"PyTorch installation failed: {e2}")
                raise typer.Exit(1)
    else:
        print_step(3, 4, "Skipping PyTorch installation...")
        print_info("Skipped (--skip-torch flag)")
    
    # Step 4: Install Forge and dependencies
    print_step(4, 4, "Installing Forge and dependencies...")
    
    try:
        # Get the forge package directory
        forge_dir = Path(__file__).parent.parent.parent.resolve()
        
        with spinner("Installing Forge package..."):
            if with_unsloth:
                _run_pip(pip_path, ["install", "-e", f"{forge_dir}[unsloth]"])
            else:
                _run_pip(pip_path, ["install", "-e", str(forge_dir)])
        
        print_success("Forge installed successfully!")
        
    except Exception as e:
        print_error(f"Forge installation failed: {e}")
        raise typer.Exit(1)
    
    # Success summary
    console.print()
    console.print("[bold green]✨ Setup complete![/]")
    console.print()
    console.print("[dim]To activate the environment:[/]")
    _print_activation_command(venv_path)
    console.print()
    console.print("[dim]Then run:[/]")
    console.print(f"  [cyan]forge init[/]")
    console.print()


def _create_venv(path: Path) -> None:
    """Create a virtual environment at the specified path."""
    import venv
    
    # Create venv with pip
    venv.create(path, with_pip=True, clear=True)


def _is_venv_valid(path: Path) -> bool:
    """Check if a venv is valid and has pip."""
    pip_path = _get_pip_path(path)
    return pip_path.exists()


def _get_pip_path(venv_path: Path) -> Path:
    """Get the pip executable path for a venv."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    return venv_path / "bin" / "pip"


def _get_python_path(venv_path: Path) -> Path:
    """Get the Python executable path for a venv."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _run_pip(pip_path: Path, args: list[str]) -> None:
    """Run pip with the given arguments."""
    cmd = [str(pip_path)] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
    )
    
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


def _print_activation_command(venv_path: Path) -> None:
    """Print the activation command for the current OS."""
    if platform.system() == "Windows":
        activate_script = venv_path / "Scripts" / "activate.bat"
        console.print(f"  [cyan]{activate_script}[/]")
        console.print("  [dim]or in PowerShell:[/]")
        ps_script = venv_path / "Scripts" / "Activate.ps1"
        console.print(f"  [cyan].\\{ps_script.relative_to(Path.cwd())}[/]")
    else:
        activate_script = venv_path / "bin" / "activate"
        console.print(f"  [cyan]source {activate_script}[/]")


def check_environment() -> dict:
    """
    Check if we're running in a proper Forge environment.
    
    Returns dict with environment status.
    """
    info = {
        "in_venv": sys.prefix != sys.base_prefix,
        "venv_path": sys.prefix if sys.prefix != sys.base_prefix else None,
        "python_version": sys.version,
        "torch_available": False,
        "cuda_available": False,
        "unsloth_available": False,
    }
    
    try:
        import torch
        info["torch_available"] = True
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_version"] = torch.__version__
    except ImportError:
        pass
    
    try:
        import unsloth
        info["unsloth_available"] = True
    except ImportError:
        pass
    
    return info
