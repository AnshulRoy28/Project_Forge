"""
Rich progress bars and spinners for Forge operations.
"""

from contextlib import contextmanager
from typing import Generator

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.live import Live
from rich.table import Table

from forge.ui.console import console


def create_training_progress() -> Progress:
    """Create a progress bar for training."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[forge.progress]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )


def create_download_progress() -> Progress:
    """Create a progress bar for downloads."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[forge.info]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )


@contextmanager
def spinner(message: str) -> Generator[None, None, None]:
    """Context manager for a simple spinner."""
    with console.status(f"[forge.progress]{message}[/]", spinner="dots"):
        yield


@contextmanager
def thinking_spinner(message: str = "Gemini is thinking...") -> Generator[None, None, None]:
    """Context manager for Gemini thinking spinner."""
    with console.status(f"[forge.gemini]🧠 {message}[/]", spinner="dots12"):
        yield


class LiveTrainingDisplay:
    """Live updating training display."""
    
    def __init__(self) -> None:
        self.table = Table(show_header=True, header_style="bold cyan")
        self.table.add_column("Epoch", style="dim")
        self.table.add_column("Step", style="dim")
        self.table.add_column("Loss", style="green")
        self.table.add_column("LR", style="yellow")
        self.table.add_column("VRAM", style="magenta")
        self.table.add_column("Time", style="blue")
        
        self._live: Live | None = None
    
    def __enter__(self) -> "LiveTrainingDisplay":
        self._live = Live(self.table, console=console, refresh_per_second=4)
        self._live.__enter__()
        return self
    
    def __exit__(self, *args) -> None:
        if self._live:
            self._live.__exit__(*args)
    
    def update(
        self,
        epoch: int,
        step: int,
        loss: float,
        lr: float,
        vram_gb: float,
        elapsed: str,
    ) -> None:
        """Update the training display with new values."""
        # Clear existing rows and add new one (keeps last N rows)
        self.table = Table(show_header=True, header_style="bold cyan")
        self.table.add_column("Epoch", style="dim")
        self.table.add_column("Step", style="dim")
        self.table.add_column("Loss", style="green")
        self.table.add_column("LR", style="yellow")
        self.table.add_column("VRAM", style="magenta")
        self.table.add_column("Time", style="blue")
        
        loss_style = "green" if loss < 1.0 else "yellow" if loss < 2.0 else "red"
        self.table.add_row(
            str(epoch),
            f"{step:,}",
            f"[{loss_style}]{loss:.4f}[/]",
            f"{lr:.2e}",
            f"{vram_gb:.1f}GB",
            elapsed,
        )
        
        if self._live:
            self._live.update(self.table)
