"""Stage 7: Training — Run training in Docker with live progress."""

import re
import time
import threading
from pathlib import Path

import docker
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from nnb.docker_runtime.container import get_container
from nnb.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


# =============================================================================
# TRAINING STATE — tracks progress parsed from container logs
# =============================================================================

class TrainingProgress:
    """Tracks training progress parsed from container log lines."""

    def __init__(self, total_epochs: int = 0):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.train_loss = 0.0
        self.train_acc = 0.0
        self.val_loss = 0.0
        self.val_acc = 0.0
        self.best_acc = 0.0
        self.status = "Starting..."
        self.finished = False
        self.failed = False
        self.error_message = ""
        self.last_batch_info = ""
        self.start_time = time.time()

    def elapsed(self) -> str:
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            return f"{hrs}h {mins}m {secs}s"
        return f"{mins}m {secs}s"

    def eta(self) -> str:
        if self.current_epoch == 0 or self.total_epochs == 0:
            return "calculating..."
        elapsed = time.time() - self.start_time
        per_epoch = elapsed / self.current_epoch
        remaining = per_epoch * (self.total_epochs - self.current_epoch)
        mins, secs = divmod(int(remaining), 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            return f"~{hrs}h {mins}m"
        return f"~{mins}m {secs}s"

    def progress_bar(self, width: int = 30) -> str:
        if self.total_epochs == 0:
            return "[" + "?" * width + "]"
        filled = int(width * self.current_epoch / self.total_epochs)
        bar = "█" * filled + "░" * (width - filled)
        pct = 100 * self.current_epoch / self.total_epochs
        return f"[{bar}] {pct:.0f}%"


def _parse_log_line(line: str, progress: TrainingProgress) -> None:
    """Parse a container log line and update training progress."""

    # Parse total epochs from config log: "--- Epoch 1/10 ---"
    m = re.search(r'Epoch (\d+)/(\d+)', line)
    if m:
        progress.current_epoch = int(m.group(1)) - 1  # still training this epoch
        progress.total_epochs = int(m.group(2))
        progress.status = f"Training epoch {m.group(1)}/{m.group(2)}"

    # Parse batch progress: "Batch 100/938 - Loss: 0.1234"
    m = re.search(r'Batch (\d+)/(\d+) - Loss: ([\d.]+)', line)
    if m:
        progress.last_batch_info = f"Batch {m.group(1)}/{m.group(2)} | Loss: {m.group(3)}"

    # Parse training epoch results: "Epoch X Training - Loss: X, Accuracy: X"
    m = re.search(r'Epoch (\d+) Training - Loss: ([\d.]+), Accuracy: ([\d.]+)', line)
    if m:
        progress.current_epoch = int(m.group(1))
        progress.train_loss = float(m.group(2))
        progress.train_acc = float(m.group(3))
        progress.status = f"Epoch {m.group(1)} — validating..."

    # Parse validation results: "Epoch X Validation - Loss: X, Accuracy: X"
    m = re.search(r'Epoch (\d+) Validation - Loss: ([\d.]+), Accuracy: ([\d.]+)', line)
    if m:
        progress.val_loss = float(m.group(2))
        progress.val_acc = float(m.group(3))
        if progress.val_acc > progress.best_acc:
            progress.best_acc = progress.val_acc
        progress.status = f"Epoch {m.group(1)} complete"
        progress.last_batch_info = ""

    # Parse completion
    if "Training process finished successfully" in line:
        progress.finished = True
        progress.status = "Training complete!"

    # Parse failure
    if "Training failed" in line or "critical error" in line.lower():
        progress.failed = True
        progress.error_message = line.strip()
        progress.status = "Training failed"


def _build_progress_display(progress: TrainingProgress) -> Panel:
    """Build a rich Panel displaying training progress."""

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="bold cyan", width=14)
    table.add_column("value")

    # Progress bar
    table.add_row("Progress", progress.progress_bar())
    table.add_row("Status", progress.status)
    table.add_row("Elapsed", progress.elapsed())

    if progress.total_epochs > 0 and not progress.finished:
        table.add_row("ETA", progress.eta())

    if progress.current_epoch > 0:
        table.add_row("", "")  # spacer
        table.add_row("Train Loss", f"{progress.train_loss:.4f}")
        table.add_row("Train Acc", f"{progress.train_acc:.2%}")
        table.add_row("Val Loss", f"{progress.val_loss:.4f}")
        table.add_row("Val Acc", f"{progress.val_acc:.2%}")
        table.add_row("Best Acc", f"⭐ {progress.best_acc:.2%}")

    if progress.last_batch_info:
        table.add_row("", "")
        table.add_row("Current", progress.last_batch_info)

    title = "🚀 Training" if not progress.finished else "✅ Training Complete"
    if progress.failed:
        title = "❌ Training Failed"

    return Panel(table, title=title, border_style="green" if progress.finished else "blue")


# =============================================================================
# PUBLIC API
# =============================================================================

def start_training(project: "Project") -> None:  # noqa: F821
    """Start training in Docker container and stream progress."""

    console.print("\n[bold]🚀 Starting training...[/bold]\n")

    workspace = project.project_dir / "workspace"
    data_dir = str(project.project_dir / "data")
    workspace_dir = str(workspace)

    (project.project_dir / "data").mkdir(exist_ok=True)

    # Ensure train.py exists
    train_file = workspace / "train.py"
    if not train_file.exists():
        raise FileNotFoundError(
            "train.py not found in workspace. Run 'nnb generate' first."
        )

    # Get container
    container = get_container(project.project_id)

    # Start container in detached mode
    console.print("🐳 Starting training container...")
    docker_container = container.start(
        command="python3 -u /workspace/train.py",  # -u for unbuffered output
        detach=True,
        workspace_dir=workspace_dir,
        data_dir=data_dir,
        remove=False,
    )

    logger.info(f"Training container started: {docker_container.id[:12]}")
    console.print(f"📦 Container: [dim]{docker_container.id[:12]}[/dim]\n")

    # Parse config for total epochs
    progress = TrainingProgress()
    config_file = workspace / "config.yaml"
    if config_file.exists():
        try:
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
            progress.total_epochs = config.get("training_params", {}).get("epochs", 0)
        except Exception:
            pass

    # Stream logs with live progress display
    interrupted = False
    try:
        _stream_with_progress(docker_container, progress)
    except KeyboardInterrupt:
        interrupted = True
        console.print("\n\n⚠️  [yellow]Training continues in background[/yellow]")
        console.print("💡 Reattach with: [cyan]nnb attach[/cyan]")
        console.print(f"💡 Container ID: [dim]{docker_container.id[:12]}[/dim]")
        return  # Don't transition state — training is still running

    # Check final result
    if progress.finished:
        console.print("\n[bold green]✅ Training complete![/bold green]")
        console.print(f"⭐ Best accuracy: {progress.best_acc:.2%}")
        console.print(f"⏱️  Total time: {progress.elapsed()}")
        console.print(f"\n💡 Model saved to: [cyan]workspace/model.pth[/cyan]")
        console.print(f"💡 Checkpoints in: [cyan]workspace/checkpoints/[/cyan]")
        # Transition happens in orchestrator after this returns
    elif progress.failed:
        console.print(f"\n[red]❌ Training failed: {progress.error_message}[/red]")
        console.print("💡 Check logs: [cyan]nnb env shell[/cyan] → cat /workspace/logs/training.log")
        raise RuntimeError(f"Training failed: {progress.error_message}")
    else:
        # Container exited but we didn't see success/fail marker
        docker_container.reload()
        exit_code = docker_container.attrs.get("State", {}).get("ExitCode", -1)
        if exit_code == 0:
            console.print("\n[bold green]✅ Training complete![/bold green]")
        else:
            logs = docker_container.logs(tail=20).decode("utf-8", errors="replace")
            console.print(f"\n[red]❌ Container exited with code {exit_code}[/red]")
            console.print(f"\n📋 Last logs:\n{logs}")
            raise RuntimeError(f"Training container exited with code {exit_code}")

    # Cleanup container
    try:
        docker_container.remove(force=True)
    except Exception:
        pass


def _stream_with_progress(
    docker_container: "docker.models.containers.Container",
    progress: TrainingProgress,
) -> None:
    """Stream container logs and display live progress."""

    # Use Rich Live display for animated progress
    with Live(_build_progress_display(progress), console=console, refresh_per_second=4) as live:
        try:
            # Stream logs in real-time
            for chunk in docker_container.logs(stream=True, follow=True):
                line = chunk.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                _parse_log_line(line, progress)
                live.update(_build_progress_display(progress))

                if progress.finished or progress.failed:
                    break

        except KeyboardInterrupt:
            raise  # Re-raise so start_training handles it


def attach_to_training(project: "Project") -> None:  # noqa: F821
    """Attach to a running training container and stream progress."""

    console.print("\n[bold]📊 Attaching to training...[/bold]\n")

    try:
        client = docker.from_env()
        container_name = f"nnb-{project.project_id}"
        docker_container = client.containers.get(container_name)

        if docker_container.status != "running":
            console.print(f"[yellow]Container status: {docker_container.status}[/yellow]")
            if docker_container.status == "exited":
                exit_code = docker_container.attrs.get("State", {}).get("ExitCode", -1)
                logs = docker_container.logs(tail=30).decode("utf-8", errors="replace")
                console.print(f"\n📋 Final logs:\n{logs}")
                if exit_code == 0:
                    console.print("\n[green]✅ Training already completed![/green]")
                else:
                    console.print(f"\n[red]❌ Training exited with code {exit_code}[/red]")
                return

        # Parse config for total epochs
        progress = TrainingProgress()
        config_file = project.project_dir / "workspace" / "config.yaml"
        if config_file.exists():
            try:
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                progress.total_epochs = config.get("training_params", {}).get("epochs", 0)
            except Exception:
                pass

        console.print(f"📦 Container: [dim]{docker_container.id[:12]}[/dim]")
        console.print("[dim]Press Ctrl+C to detach (training continues)[/dim]\n")

        # Replay recent logs to catch up on progress
        recent_logs = docker_container.logs(tail=50).decode("utf-8", errors="replace")
        for line in recent_logs.split("\n"):
            if line.strip():
                _parse_log_line(line.strip(), progress)

        # Stream remaining logs
        try:
            _stream_with_progress(docker_container, progress)
        except KeyboardInterrupt:
            console.print("\n\n⚠️  [yellow]Detached — training continues in background[/yellow]")
            console.print("💡 Reattach with: [cyan]nnb attach[/cyan]")

        if progress.finished:
            console.print("\n[bold green]✅ Training complete![/bold green]")
            console.print(f"⭐ Best accuracy: {progress.best_acc:.2%}")

    except docker.errors.NotFound:
        console.print("[red]❌ No training container found[/red]")
        console.print("💡 Start training with: [cyan]nnb train[/cyan]")
    except docker.errors.DockerException as e:
        console.print(f"[red]❌ Docker error: {e}[/red]")
