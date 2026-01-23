#!/usr/bin/env python3
"""
Forge Bootstrap Script

Run this script to set up Forge with a virtual environment and all dependencies.
No prior installation required - just Python 3.10+.

Usage:
    python bootstrap.py
    python bootstrap.py --no-unsloth     # Skip Unsloth installation
    python bootstrap.py --venv myenv     # Custom venv name
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path


# Minimum Python version
MIN_PYTHON = (3, 10)
VENV_DIR = ".forge-venv"


def print_banner():
    """Print the Forge banner."""
    banner = """
    ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
    ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
    █████╗  ██║   ██║██████╔╝██║  ███╗█████╗  
    ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  
    ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
    ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
    
    Local-First SLM Fine-Tuning with Gemini 3
    """
    print(banner)


def check_python_version():
    """Check if Python version is sufficient."""
    if sys.version_info < MIN_PYTHON:
        print(f"❌ Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, found {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def create_venv(venv_path: Path) -> None:
    """Create virtual environment."""
    import venv
    
    print(f"\n[1/4] Creating virtual environment at {venv_path}...")
    venv.create(venv_path, with_pip=True, clear=True)
    print("✓ Virtual environment created")


def get_pip_path(venv_path: Path) -> Path:
    """Get pip executable path."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    return venv_path / "bin" / "pip"


def get_python_path(venv_path: Path) -> Path:
    """Get Python executable path."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def run_pip(pip_path: Path, args: list, desc: str = ""):
    """Run pip with arguments."""
    cmd = [str(pip_path)] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"⚠ {desc or 'pip'} warning: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"⚠ {desc or 'pip'} timed out")
        return False


def install_pytorch(pip_path: Path) -> bool:
    """Install PyTorch with CUDA support."""
    print("\n[2/4] Installing PyTorch with CUDA support...")
    print("    (This may take several minutes)")
    
    # Try CUDA 12.1 first
    success = run_pip(pip_path, [
        "install", 
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ], "PyTorch CUDA")
    
    if success:
        print("✓ PyTorch installed with CUDA 12.1 support")
        return True
    
    # Fallback to default (may be CPU or older CUDA)
    print("  Trying default PyTorch...")
    success = run_pip(pip_path, [
        "install", "torch", "torchvision", "torchaudio"
    ], "PyTorch")
    
    if success:
        print("✓ PyTorch installed (default version)")
        return True
    
    print("❌ PyTorch installation failed")
    return False


def install_forge(pip_path: Path, with_unsloth: bool) -> bool:
    """Install Forge package."""
    print("\n[3/4] Installing Forge and dependencies...")
    
    forge_dir = Path(__file__).parent.resolve()
    
    if with_unsloth:
        extras = "[unsloth]"
    else:
        extras = ""
    
    success = run_pip(pip_path, [
        "install", "-e", f"{forge_dir}{extras}"
    ], "Forge")
    
    if success:
        print("✓ Forge installed successfully")
        return True
    
    # Try without unsloth if it failed
    if with_unsloth:
        print("  Retrying without Unsloth...")
        success = run_pip(pip_path, [
            "install", "-e", str(forge_dir)
        ], "Forge (no Unsloth)")
        
        if success:
            print("✓ Forge installed (without Unsloth)")
            return True
    
    print("❌ Forge installation failed")
    return False


def print_activation_instructions(venv_path: Path):
    """Print activation instructions."""
    print("\n[4/4] Setup complete!")
    print("\n" + "=" * 50)
    print("✨ Forge is ready!")
    print("=" * 50)
    
    print("\nTo activate the environment:")
    
    if platform.system() == "Windows":
        ps_script = venv_path / "Scripts" / "Activate.ps1"
        cmd_script = venv_path / "Scripts" / "activate.bat"
        print(f"\n  PowerShell:  .\\{ps_script.relative_to(Path.cwd())}")
        print(f"  CMD:         {cmd_script.relative_to(Path.cwd())}")
    else:
        activate = venv_path / "bin" / "activate"
        print(f"\n  source {activate}")
    
    print("\nThen run:")
    print("  forge init      # Configure your Gemini API key")
    print("  forge --help    # See all commands")
    print()


def main():
    parser = argparse.ArgumentParser(description="Bootstrap Forge environment")
    parser.add_argument("--venv", type=str, default=VENV_DIR, help="Venv directory name")
    parser.add_argument("--no-unsloth", action="store_true", help="Skip Unsloth installation")
    args = parser.parse_args()
    
    print_banner()
    
    # Check Python version
    check_python_version()
    
    # Determine venv path
    venv_path = Path.cwd() / args.venv
    
    if venv_path.exists():
        print(f"\n⚠ Venv already exists: {venv_path}")
        response = input("  Recreate it? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(0)
        import shutil
        shutil.rmtree(venv_path)
    
    # Create venv
    create_venv(venv_path)
    
    # Get pip path
    pip_path = get_pip_path(venv_path)
    
    # Upgrade pip
    run_pip(pip_path, ["install", "--upgrade", "pip"], "pip upgrade")
    
    # Install PyTorch
    if not install_pytorch(pip_path):
        print("\n⚠ Continuing without verified PyTorch...")
    
    # Install Forge
    if not install_forge(pip_path, with_unsloth=not args.no_unsloth):
        print("\n❌ Setup failed. Check errors above.")
        sys.exit(1)
    
    # Print instructions
    print_activation_instructions(venv_path)


if __name__ == "__main__":
    main()
