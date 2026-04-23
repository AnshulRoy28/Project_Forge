"""Stage 5: Docker Environment Setup."""

import docker
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from nnb.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


def generate_dockerfile(project: "Project") -> str:  # noqa: F821
    """Generate Dockerfile based on project spec."""
    spec = project._spec
    
    # Base image selection
    if spec.framework == "pytorch":
        base_image = "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
    elif spec.framework == "tensorflow":
        base_image = "tensorflow/tensorflow:2.14.0-gpu"
    elif spec.framework == "jax":
        base_image = "python:3.10-slim"
    else:
        base_image = "python:3.10-slim"
    
    # Build requirements list
    requirements = []
    
    if spec.framework == "pytorch":
        requirements.extend([
            "torch>=2.1.0",
            "torchvision>=0.16.0",
            "torchaudio>=2.1.0",
        ])
    elif spec.framework == "tensorflow":
        requirements.append("tensorflow>=2.14.0")
    elif spec.framework == "jax":
        requirements.extend([
            "jax[cuda11_pip]>=0.4.20",
            "flax>=0.7.5",
        ])
    
    # Common ML libraries
    requirements.extend([
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.14.0",
    ])
    
    # Generate Dockerfile
    dockerfile = f"""# Auto-generated Dockerfile for {project.project_id}
FROM {base_image}

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    curl \\
    vim \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    {' '.join(requirements)}

# Create data directory
RUN mkdir -p /data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/data/.torch

# Default command
CMD ["/bin/bash"]
"""
    
    return dockerfile


def build_environment(project: "Project") -> None:  # noqa: F821
    """Build Docker environment."""
    try:
        # Generate Dockerfile
        console.print("📝 Generating Dockerfile...")
        dockerfile_content = generate_dockerfile(project)
        
        # Save Dockerfile
        dockerfile_path = project.project_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        logger.info(f"Generated Dockerfile at {dockerfile_path}")
        console.print(f"✓ Dockerfile saved to: {dockerfile_path}")
        
        # Build Docker image
        console.print("\n🐳 Building Docker image...")
        console.print("[dim]This may take a few minutes on first build...[/dim]\n")
        
        client = docker.from_env()
        image_tag = f"nnb-{project.project_id}:latest"
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Building image...", total=None)
            
            try:
                # Build image
                image, build_logs = client.images.build(
                    path=str(project.project_dir),
                    tag=image_tag,
                    rm=True,
                    forcerm=True,
                )
                
                progress.update(task, completed=True)
                logger.info(f"Built Docker image: {image_tag}")
                console.print(f"\n✓ Docker image built: [bold]{image_tag}[/bold]")
                
            except docker.errors.BuildError as e:
                progress.stop()
                console.print(f"\n[red]❌ Docker build failed:[/red]")
                for log in e.build_log:
                    if 'stream' in log:
                        console.print(log['stream'].strip())
                raise
            except docker.errors.APIError as e:
                progress.stop()
                console.print(f"\n[red]❌ Docker API error: {e}[/red]")
                raise
        
        console.print("\n[green]✓ Environment ready![/green]")
        
        # Run health check
        console.print("\n🏥 Running container health check...")
        health_passed = _run_health_check(client, image_tag, project)
        
        if not health_passed:
            console.print("\n[red]❌ Health check failed. Fix the issues above and run: nnb env build[/red]")
            raise RuntimeError("Container health check failed")
        
        console.print("\n💡 Next steps:")
        console.print("  • Generate code: [cyan]nnb generate[/cyan]")
        console.print("  • Open shell: [cyan]nnb env shell[/cyan]")
        
    except docker.errors.DockerException as e:
        console.print(f"\n[red]❌ Docker error: {e}[/red]")
        console.print("\n💡 Make sure Docker is installed and running:")
        console.print("  • Windows/Mac: Docker Desktop")
        console.print("  • Linux: Docker Engine")
        raise
    except Exception as e:
        logger.error(f"Environment build failed: {e}", exc_info=True)
        raise


def _run_health_check(
    client: docker.DockerClient,
    image_tag: str,
    project: "Project",  # noqa: F821
) -> bool:
    """Run health check on freshly built container."""
    
    spec = project._spec
    
    # Framework-specific import check (use string concat — no f-strings to avoid quote issues)
    framework_checks = {
        "pytorch": "import torch; import torchvision; print('PyTorch ' + torch.__version__)",
        "tensorflow": "import tensorflow as tf; print('TensorFlow ' + tf.__version__)",
        "jax": "import jax; print('JAX ' + jax.__version__)",
        "scikit-learn": "import sklearn; print('scikit-learn ' + sklearn.__version__)",
    }
    
    framework_cmd = framework_checks.get(spec.framework, "print('Unknown framework')")
    
    # Each check: (name, command_list)
    # Using list form avoids all shell quoting issues
    checks = [
        ("Python version", ["python3", "--version"]),
        ("Framework import", ["python3", "-c", framework_cmd]),
        ("Workspace writable", ["/bin/bash", "-c", "touch /workspace/.health_check && rm /workspace/.health_check"]),
        ("Data dir exists", ["ls", "/data"]),
    ]
    
    # Start a temporary container
    workspace_dir = str(project.project_dir / "workspace")
    data_dir = str(project.project_dir / "data")
    
    (project.project_dir / "data").mkdir(exist_ok=True)
    (project.project_dir / "workspace").mkdir(exist_ok=True)
    
    all_passed = True
    
    try:
        container = client.containers.run(
            image_tag,
            command="sleep 30",
            detach=True,
            remove=True,
            volumes={
                data_dir: {"bind": "/data", "mode": "ro"},
                workspace_dir: {"bind": "/workspace", "mode": "rw"},
            },
        )
        
        for check_name, cmd in checks:
            try:
                result = container.exec_run(cmd)
                if result.exit_code == 0:
                    output = result.output.decode().strip()
                    console.print(f"  [green]✓[/green] {check_name}: {output}")
                else:
                    error = result.output.decode().strip()
                    console.print(f"  [red]✗[/red] {check_name}: {error}")
                    all_passed = False
            except Exception as e:
                console.print(f"  [red]✗[/red] {check_name}: {e}")
                all_passed = False
        
        # Clean up
        try:
            container.stop(timeout=2)
        except Exception:
            pass
        
    except Exception as e:
        logger.error(f"Health check container failed: {e}", exc_info=True)
        console.print(f"  [red]✗[/red] Could not start health check container: {e}")
        all_passed = False
    
    if all_passed:
        console.print("\n[green]✓ All health checks passed[/green]")
    
    return all_passed
