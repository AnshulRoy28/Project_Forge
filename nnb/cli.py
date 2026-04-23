"""Main CLI entry point."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from nnb.orchestrator.project import Project
from nnb.orchestrator.state import State
from nnb.utils.logging import setup_logging
from nnb.utils.api_key_manager import (
    APIKeyManager,
    setup_api_key_interactive,
    delete_api_key_interactive,
    show_api_key_status,
)

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """CLI Neural Network Builder - Build and train neural networks through conversation."""
    setup_logging()


@main.command()
def start() -> None:
    """Start a new neural network project."""
    try:
        # Check for API key first
        if not APIKeyManager.has_api_key():
            console.print("[yellow]⚠️  No Gemini API key configured[/yellow]")
            console.print("\n💡 Let's set that up first...\n")
            
            if not setup_api_key_interactive():
                console.print("\n[red]❌ Cannot proceed without API key[/red]")
                console.print("Run [cyan]nnb config setup[/cyan] to configure it later")
                sys.exit(1)
            
            console.print()  # Blank line
        
        project = Project.create()
        console.print(f"✓ Created project: [bold]{project.project_id}[/bold]")
        console.print(f"📁 Project directory: {project.project_dir}")
        console.print("\n🎯 Starting conversation...")
        
        project.start_conversation()
        
    except KeyboardInterrupt:
        console.print("\n⚠️  Interrupted. Resume with: nnb resume <project-id>")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("project_id")
def resume(project_id: str) -> None:
    """Resume an existing project."""
    try:
        project = Project.load(project_id)
        console.print(f"✓ Resumed project: [bold]{project_id}[/bold]")
        console.print(f"📊 Current state: {project.state.value}")
        
        # Continue from current state
        project.continue_from_state()
        
    except FileNotFoundError:
        console.print(f"[red]❌ Project '{project_id}' not found[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--project-id", help="Project ID (uses current directory if not specified)")
def status(project_id: Optional[str]) -> None:
    """Show project status and next action."""
    try:
        if project_id:
            project = Project.load(project_id)
        else:
            project = Project.load_from_current_dir()
        
        console.print(f"\n📊 Project: [bold]{project.project_id}[/bold]")
        console.print(f"📁 Location: {project.project_dir}")
        console.print(f"🎯 State: [bold]{project.state.value}[/bold]")
        
        # Show next action
        next_action = project.get_next_action()
        console.print(f"\n💡 Next: {next_action}")
        
    except FileNotFoundError:
        console.print("[red]❌ No project found in current directory[/red]")
        console.print("💡 Start a new project with: nnb start")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@main.group()
def data() -> None:
    """Data validation commands."""
    pass


@data.command("validate")
@click.option("--path", type=click.Path(exists=True), help="Path to data directory (not needed for torchvision datasets)")
@click.option("--project-id", help="Project ID (uses current directory if not specified)")
def data_validate(path: str, project_id: Optional[str]) -> None:
    """Validate training data."""
    try:
        if project_id:
            project = Project.load(project_id)
        else:
            project = Project.load_from_current_dir()
        
        # Check if using torchvision dataset
        dataset_source = project._spec.dataset_source
        if dataset_source in ["torchvision", "MNIST", "CIFAR10", "CIFAR100", "FashionMNIST", "ImageNet"]:
            console.print(f"\n[green]✓ Using torchvision dataset: {dataset_source}[/green]")
            console.print("  Dataset will be automatically downloaded during training")
            
            # Call validate_data to transition state
            result = project.validate_data(None)
            
            console.print("  Validation passed!\n")
            console.print("[bold]Next step:[/bold]")
            console.print("  Run: [cyan]nnb env build[/cyan]")
            return
        
        # For custom datasets, require path
        if not path:
            console.print("[red]❌ --path is required for custom datasets[/red]")
            console.print("💡 Example: nnb data validate --path /path/to/your/data")
            sys.exit(1)
        
        console.print(f"🔍 Validating data at: {path}")
        
        result = project.validate_data(Path(path))
        
        if result.status == "pass":
            console.print("[green]✓ Data validation passed[/green]")
        elif result.status == "warn":
            console.print("[yellow]⚠️  Data validation passed with warnings[/yellow]")
        else:
            console.print("[red]❌ Data validation failed[/red]")
        
        # Show issues
        if result.issues:
            console.print("\n📋 Issues found:")
            for issue in result.issues:
                icon = "🔴" if issue.severity == "error" else "🟡"
                console.print(f"  {icon} {issue.message}")
                console.print(f"     Fix: {issue.fix}")
        
        if result.status == "fail":
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@data.command("status")
@click.option("--project-id", help="Project ID (uses current directory if not specified)")
def data_status(project_id: Optional[str]) -> None:
    """Show data requirements status."""
    try:
        if project_id:
            project = Project.load(project_id)
        else:
            project = Project.load_from_current_dir()
        
        requirements = project.get_data_requirements()
        console.print("\n📋 Data Requirements:")
        console.print(requirements)
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@main.group()
def env() -> None:
    """Docker environment commands."""
    pass


@env.command("build")
@click.option("--project-id", help="Project ID (uses current directory if not specified)")
def env_build(project_id: Optional[str]) -> None:
    """Build Docker environment."""
    try:
        if project_id:
            project = Project.load(project_id)
        else:
            project = Project.load_from_current_dir()
        
        console.print("🔧 Building Docker container...")
        project.build_environment()
        console.print("[green]✓ Container built successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@env.command("shell")
@click.option("--project-id", help="Project ID (uses current directory if not specified)")
def env_shell(project_id: Optional[str]) -> None:
    """Open shell in Docker container."""
    try:
        if project_id:
            project = Project.load(project_id)
        else:
            project = Project.load_from_current_dir()
        
        console.print("🐚 Opening container shell...")
        project.open_shell()
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@main.command("generate")
@click.option("--project-id", help="Project ID (uses current directory if not specified)")
@click.option("--from", "from_path", help="Copy code from a previous session (e.g. .nnb/nnb-xxx)")
def generate(project_id: Optional[str], from_path: Optional[str]) -> None:
    """Generate training code using Gemini, or copy from a previous session."""
    try:
        if project_id:
            project = Project.load(project_id)
        else:
            project = Project.load_from_current_dir()
        
        if from_path:
            _copy_from_session(project, from_path)
        else:
            project.generate_code()
        
    except KeyboardInterrupt:
        console.print("\n⚠️  Code generation interrupted")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


def _copy_from_session(project: "Project", from_path: str) -> None:
    """Copy workspace files from a previous session into the current project."""
    import shutil
    from pathlib import Path
    
    source = Path(from_path)
    
    # Handle relative paths 
    if not source.is_absolute():
        source = Path.cwd() / source
    
    # Check if it's a project dir or workspace dir
    source_workspace = source / "workspace" if (source / "workspace").exists() else source
    
    if not source_workspace.exists():
        console.print(f"[red]❌ Source path not found: {source}[/red]")
        console.print("💡 Expected a path like: .nnb/nnb-20260423-091628-af157ba1")
        raise FileNotFoundError(f"Source path not found: {source}")
    
    # Check for expected files
    expected = ["train.py", "model.py", "dataset.py"]
    found = [f for f in expected if (source_workspace / f).exists()]
    
    if not found:
        console.print(f"[red]❌ No training files found in {source_workspace}[/red]")
        raise FileNotFoundError("No training files in source workspace")
    
    # Copy files
    dest_workspace = project.project_dir / "workspace"
    dest_workspace.mkdir(exist_ok=True)
    
    console.print(f"\n📂 Copying from: [cyan]{source_workspace}[/cyan]\n")
    
    copied = 0
    for item in source_workspace.iterdir():
        if item.is_file() and not item.name.startswith('.'):
            dest = dest_workspace / item.name
            shutil.copy2(item, dest)
            console.print(f"  [green]✓[/green] {item.name}")
            copied += 1
    
    console.print(f"\n✅ Copied {copied} files from previous session")
    
    # Also copy imp_points.md if it exists in the source project root
    source_imp = source / "imp_points.md"
    if source_imp.exists() and source != source_workspace:
        dest_imp = project.project_dir / "imp_points.md"
        if not dest_imp.exists():
            shutil.copy2(source_imp, dest_imp)
            console.print("  [green]✓[/green] imp_points.md (key decisions)")
    
    # Transition state
    from nnb.orchestrator.state import State
    project.transition_to(State.CODE_GENERATED)
    console.print("\n💡 Next step: [cyan]nnb mock-run[/cyan]")


@main.command("mock-run")
@click.option("--project-id", help="Project ID (uses current directory if not specified)")
def mock_run(project_id: Optional[str]) -> None:
    """Run mock training pass."""
    try:
        if project_id:
            project = Project.load(project_id)
        else:
            project = Project.load_from_current_dir()
        
        console.print("🧪 Running mock training pass...")
        result = project.run_mock()
        
        if result.succeeded:
            console.print("[green]✓ Mock run passed[/green]")
        else:
            console.print("[red]❌ Mock run failed[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@main.command("train")
@click.option("--project-id", help="Project ID (uses current directory if not specified)")
def train(project_id: Optional[str]) -> None:
    """Start training."""
    try:
        if project_id:
            project = Project.load(project_id)
        else:
            project = Project.load_from_current_dir()
        
        project.start_training()
        
    except KeyboardInterrupt:
        console.print("\n⚠️  Detached from training (it continues in background)")
        console.print("💡 Reattach: [cyan]nnb attach[/cyan]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@main.command("attach")
@click.option("--project-id", help="Project ID (uses current directory if not specified)")
def attach(project_id: Optional[str]) -> None:
    """Attach to running training."""
    try:
        if project_id:
            project = Project.load(project_id)
        else:
            project = Project.load_from_current_dir()
        
        console.print("📊 Attaching to training...")
        project.attach_to_training()
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@main.group()
def config() -> None:
    """Configuration management."""
    pass


@config.command("setup")
def config_setup() -> None:
    """Set up Gemini API key."""
    try:
        setup_api_key_interactive()
    except KeyboardInterrupt:
        console.print("\n⚠️  Setup cancelled")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@config.command("delete-key")
def config_delete_key() -> None:
    """Delete stored Gemini API key."""
    try:
        delete_api_key_interactive()
    except KeyboardInterrupt:
        console.print("\n⚠️  Cancelled")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@config.command("status")
def config_status() -> None:
    """Show API key configuration status."""
    try:
        show_api_key_status()
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
