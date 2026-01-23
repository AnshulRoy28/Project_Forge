"""
forge docker - Docker container management for GPU-optimized training.

Commands for building and running architecture-specific containers.
"""

import subprocess
import platform
from pathlib import Path
from typing import Optional

import typer

from forge.ui.console import console, print_success, print_error, print_info, print_warning
from forge.ui.progress import spinner


docker_app = typer.Typer(
    name="docker",
    help="🐳 Docker container management for GPU training",
    no_args_is_help=True,
)


# GPU architecture mapping
GPU_PROFILES = {
    "blackwell": {"name": "Blackwell", "gpus": "RTX 5090/5080/5070", "dockerfile": "Dockerfile.blackwell"},
    "ada": {"name": "Ada Lovelace", "gpus": "RTX 4090/4080/4070", "dockerfile": "Dockerfile.ada"},
    "ampere": {"name": "Ampere", "gpus": "RTX 3090/3080/3070", "dockerfile": "Dockerfile.ampere"},
    "hopper": {"name": "Hopper", "gpus": "H100/H200", "dockerfile": "Dockerfile.hopper"},
}


def _check_docker_running() -> bool:
    """Check if Docker is running and accessible."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _print_docker_help():
    """Print help for Docker setup issues."""
    console.print("\n[bold yellow]🐳 Docker Setup Required[/]\n")
    console.print("Forge needs Docker Desktop to build and run containers.")
    console.print()
    console.print("[bold]Windows Setup:[/]")
    console.print("  1. Install Docker Desktop from: [cyan]https://docker.com/products/docker-desktop[/]")
    console.print("  2. Start Docker Desktop")
    console.print("  3. Wait for Docker to fully start (check system tray)")
    console.print("  4. Run [cyan]forge docker build[/] again")
    console.print()
    console.print("[bold]Troubleshooting:[/]")
    console.print("  • Make sure Docker Desktop is running (not just installed)")
    console.print("  • Check Docker Desktop system tray icon shows 'running'")
    console.print("  • Try: [cyan]docker info[/] to test Docker connection")
    console.print("  • Restart Docker Desktop if needed")
    console.print()


def _get_docker_dir() -> Path:
    """Get the docker directory path."""
    # Check relative to package or current directory
    for base in [Path(__file__).parent.parent.parent, Path.cwd()]:
        docker_dir = base / "docker"
        if docker_dir.exists():
            return docker_dir
    raise FileNotFoundError("Docker directory not found")
    """Get the docker directory path."""
    # Check relative to package or current directory
    for base in [Path(__file__).parent.parent.parent, Path.cwd()]:
        docker_dir = base / "docker"
        if docker_dir.exists():
            return docker_dir
    raise FileNotFoundError("Docker directory not found")


def _detect_gpu_architecture() -> str:
    """Detect GPU architecture from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            compute_cap = result.stdout.strip().split("\n")[0]
            major = int(compute_cap.split(".")[0])
            minor = int(compute_cap.split(".")[1]) if "." in compute_cap else 0
            
            if major >= 12:
                return "blackwell"
            elif major == 9:
                return "hopper"
            elif major == 8 and minor >= 9:
                return "ada"
            elif major >= 8:
                return "ampere"
    except Exception:
        pass
    
    return "ada"  # Default fallback


def _get_gpu_name() -> str:
    """Get GPU name from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "Unknown GPU"


@docker_app.command("status")
def status_command():
    """
    Check Docker status and connection.
    """
    console.print("\n[bold]🐳 Docker Status Check[/]\n")
    
    # Check if docker command exists
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            console.print(f"  [green]✓[/] Docker installed: [dim]{result.stdout.strip()}[/]")
        else:
            console.print("  [red]✗[/] Docker command failed")
            raise typer.Exit(1)
    except FileNotFoundError:
        console.print("  [red]✗[/] Docker not installed")
        _print_docker_help()
        raise typer.Exit(1)
    except subprocess.TimeoutExpired:
        console.print("  [red]✗[/] Docker command timeout")
        raise typer.Exit(1)
    
    # Check if Docker daemon is running
    if _check_docker_running():
        console.print("  [green]✓[/] Docker daemon running")
        
        # Get Docker info
        try:
            result = subprocess.run(["docker", "system", "info", "--format", "{{.ServerVersion}}"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                console.print(f"  [green]✓[/] Docker version: [dim]{result.stdout.strip()}[/]")
        except:
            pass
            
        # Check GPU support
        try:
            result = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.1-base-ubuntu22.04", "nvidia-smi", "-L"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                gpu_count = len([line for line in result.stdout.split('\n') if 'GPU' in line])
                console.print(f"  [green]✓[/] GPU support: [dim]{gpu_count} GPU(s) detected[/]")
            else:
                console.print("  [yellow]⚠[/] GPU support: [dim]Not available or not configured[/]")
        except:
            console.print("  [yellow]⚠[/] GPU support: [dim]Could not test[/]")
    else:
        console.print("  [red]✗[/] Docker daemon not running")
        _print_docker_help()
        raise typer.Exit(1)
    
    console.print()
    console.print("[green]✨ Docker is ready for Forge![/]")
    console.print()


@docker_app.command("detect")
def detect_command():
    """
    Detect your GPU and show the recommended container.
    """
    console.print("\n[bold]🔍 Detecting GPU...[/]\n")
    
    gpu_name = _get_gpu_name()
    arch = _detect_gpu_architecture()
    profile = GPU_PROFILES.get(arch, GPU_PROFILES["ada"])
    
    console.print(f"  GPU: [cyan]{gpu_name}[/]")
    console.print(f"  Architecture: [cyan]{profile['name']}[/]")
    console.print(f"  Recommended Container: [green]forge:{arch}[/]")
    console.print()
    console.print(f"[dim]Build: forge docker build {arch}[/]")
    console.print(f"[dim]Run:   forge docker run {arch} train[/]")
    console.print()


@docker_app.command("build")
def build_command(
    architecture: str = typer.Argument(
        None, 
        help="GPU architecture: blackwell, ada, ampere, hopper (auto-detects if omitted)"
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Build without cache"),
):
    """
    Build the Docker container for your GPU architecture.
    """
    # Check if Docker is running first
    if not _check_docker_running():
        print_error("Docker is not running or not accessible")
        _print_docker_help()
        raise typer.Exit(1)
    
    # Auto-detect if not specified
    if not architecture:
        architecture = _detect_gpu_architecture()
        print_info(f"Auto-detected: {architecture}")
    
    if architecture not in GPU_PROFILES:
        print_error(f"Unknown architecture: {architecture}")
        console.print(f"[dim]Available: {', '.join(GPU_PROFILES.keys())}[/]")
        raise typer.Exit(1)
    
    profile = GPU_PROFILES[architecture]
    
    console.print(f"\n[bold]🐳 Building Forge container for {profile['name']}[/]")
    console.print(f"  Supports: [cyan]{profile['gpus']}[/]\n")
    
    try:
        docker_dir = _get_docker_dir()
        project_dir = docker_dir.parent
        dockerfile = docker_dir / profile["dockerfile"]
        
        if not dockerfile.exists():
            print_error(f"Dockerfile not found: {dockerfile}")
            raise typer.Exit(1)
        
        # Build command
        cmd = [
            "docker", "build",
            "-t", f"forge:{architecture}",
            "-f", str(dockerfile),
        ]
        
        if no_cache:
            cmd.append("--no-cache")
        
        cmd.append(str(project_dir))
        
        console.print(f"[dim]$ {' '.join(cmd)}[/]\n")
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print_success(f"Built forge:{architecture}")
            console.print()
            console.print(f"[dim]Run with: forge docker run {architecture} train[/]")
        else:
            print_error("Build failed")
            console.print("\n[dim]Common issues:[/]")
            console.print("  • Make sure Docker Desktop is running")
            console.print("  • Check internet connection for base image download")
            console.print("  • Try: [cyan]forge docker build --no-cache[/]")
            raise typer.Exit(1)
            
    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)


@docker_app.command("run")
def run_command(
    architecture: str = typer.Argument(
        None,
        help="GPU architecture: blackwell, ada, ampere, hopper (auto-detects if omitted)"
    ),
    command: list[str] = typer.Argument(
        None,
        help="Forge command to run (e.g., 'train', 'plan \"goal\"')"
    ),
    data_dir: Path = typer.Option(
        Path("./data"), "--data", "-d", 
        help="Data directory to mount"
    ),
    output_dir: Path = typer.Option(
        Path("./output"), "--output", "-o",
        help="Output directory to mount"
    ),
    detach: bool = typer.Option(False, "--detach", "-D", help="Run in background"),
):
    """
    Run a Forge command in the Docker container.
    
    Example:
        forge docker run ada train
        forge docker run -- plan "Make a coding assistant"
    """
    # Check if Docker is running first
    if not _check_docker_running():
        print_error("Docker is not running or not accessible")
        _print_docker_help()
        raise typer.Exit(1)
    
    # Auto-detect if not specified
    if not architecture:
        architecture = _detect_gpu_architecture()
        print_info(f"Auto-detected: {architecture}")
    
    if architecture not in GPU_PROFILES:
        print_error(f"Unknown architecture: {architecture}")
        raise typer.Exit(1)
    
    profile = GPU_PROFILES[architecture]
    image = f"forge:{architecture}"
    
    # Check if image exists
    check = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True
    )
    
    if check.returncode != 0:
        print_warning(f"Image {image} not found. Building...")
        build_command(architecture, no_cache=False)
    
    console.print(f"\n[bold]🐳 Running Forge ({profile['name']})[/]\n")
    
    # Build docker run command
    cmd = [
        "docker", "run",
        "--gpus", "all",
        "--rm",
        "-v", f"{data_dir.resolve()}:/data",
        "-v", f"{output_dir.resolve()}:/output",
        "-v", f"{Path('./checkpoints').resolve()}:/checkpoints",
    ]
    
    # Mount forge.yaml if exists
    config_file = Path("./forge.yaml")
    if config_file.exists():
        cmd.extend(["-v", f"{config_file.resolve()}:/app/forge.yaml:ro"])
    
    # Pass GEMINI_API_KEY if set
    import os
    if "GEMINI_API_KEY" in os.environ:
        cmd.extend(["-e", f"GEMINI_API_KEY={os.environ['GEMINI_API_KEY']}"])
    
    # Interactive mode for test command
    if command and "test" in command:
        cmd.extend(["-it"])
    
    if detach:
        cmd.append("-d")
    
    cmd.append(image)
    
    # Add forge command
    if command:
        cmd.extend(command)
    
    console.print(f"[dim]$ {' '.join(cmd)}[/]\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@docker_app.command("list")
def list_command():
    """
    List available GPU architectures and their containers.
    """
    console.print("\n[bold]🐳 Available Forge Containers[/]\n")
    
    # Check which images exist
    for arch, profile in GPU_PROFILES.items():
        image = f"forge:{arch}"
        check = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True
        )
        
        status = "[green]✓ Built[/]" if check.returncode == 0 else "[dim]Not built[/]"
        console.print(f"  [cyan]{image:20}[/] {profile['name']:15} {profile['gpus']:25} {status}")
    
    console.print()
    console.print("[dim]Build: forge docker build <arch>[/]")
    console.print("[dim]Run:   forge docker run <arch> <command>[/]")
    console.print()


@docker_app.command("shell")
def shell_command(
    architecture: str = typer.Argument(None, help="GPU architecture"),
):
    """
    Open an interactive shell in the container.
    """
    if not architecture:
        architecture = _detect_gpu_architecture()
    
    if architecture not in GPU_PROFILES:
        print_error(f"Unknown architecture: {architecture}")
        raise typer.Exit(1)
    
    image = f"forge:{architecture}"
    
    cmd = [
        "docker", "run",
        "--gpus", "all",
        "--rm", "-it",
        "-v", f"{Path('./data').resolve()}:/data",
        "-v", f"{Path('./output').resolve()}:/output",
        "--entrypoint", "/bin/bash",
        image,
    ]
    
    console.print(f"\n[bold]🐳 Opening shell in forge:{architecture}[/]\n")
    subprocess.run(cmd)
