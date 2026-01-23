"""
Rich panels for status displays in Forge.
"""

from rich.panel import Panel
from rich.table import Table
from rich.box import ROUNDED

from forge.ui.console import console


def create_hardware_panel(
    gpu_name: str,
    vram_total: float,
    vram_free: float,
    cuda_version: str,
    driver_version: str,
    ram_total: float,
    ram_free: float,
    cpu_name: str,
) -> Panel:
    """Create a hardware status panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="forge.muted")
    table.add_column("Value", style="bold")
    
    # GPU info
    table.add_row("🎮 GPU", f"[forge.gpu]{gpu_name}[/]")
    vram_used = vram_total - vram_free
    vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0
    vram_color = "green" if vram_pct < 50 else "yellow" if vram_pct < 80 else "red"
    table.add_row("💾 VRAM", f"[{vram_color}]{vram_used:.1f}GB / {vram_total:.1f}GB ({vram_pct:.0f}%)[/]")
    table.add_row("🔧 CUDA", cuda_version)
    table.add_row("📦 Driver", driver_version)
    
    # System info
    table.add_row("", "")  # Spacer
    ram_used = ram_total - ram_free
    ram_pct = (ram_used / ram_total * 100) if ram_total > 0 else 0
    ram_color = "green" if ram_pct < 50 else "yellow" if ram_pct < 80 else "red"
    table.add_row("🖥️  CPU", cpu_name)
    table.add_row("🧮 RAM", f"[{ram_color}]{ram_used:.1f}GB / {ram_total:.1f}GB ({ram_pct:.0f}%)[/]")
    
    return Panel(
        table,
        title="[forge.title]⚙️  Hardware Status[/]",
        border_style="cyan",
        box=ROUNDED,
    )


def create_training_panel(
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    loss: float,
    learning_rate: float,
    vram_used: float,
    gpu_temp: int,
) -> Panel:
    """Create a training progress panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="forge.muted")
    table.add_column("Value", style="bold")
    
    table.add_row("📊 Epoch", f"{epoch} / {total_epochs}")
    table.add_row("🔄 Step", f"{step:,} / {total_steps:,}")
    
    loss_color = "green" if loss < 1.0 else "yellow" if loss < 2.0 else "red"
    table.add_row("📉 Loss", f"[{loss_color}]{loss:.4f}[/]")
    table.add_row("📈 LR", f"{learning_rate:.2e}")
    
    table.add_row("", "")  # Spacer
    table.add_row("💾 VRAM", f"{vram_used:.1f}GB")
    
    temp_color = "green" if gpu_temp < 70 else "yellow" if gpu_temp < 85 else "red"
    table.add_row("🌡️  Temp", f"[{temp_color}]{gpu_temp}°C[/]")
    
    return Panel(
        table,
        title="[forge.title]🔥 Training Progress[/]",
        border_style="magenta",
        box=ROUNDED,
    )


def create_gemini_panel(message: str, thinking: bool = False) -> Panel:
    """Create a panel for Gemini responses."""
    title = "[forge.gemini]🧠 Gemini is thinking...[/]" if thinking else "[forge.gemini]🧠 Gemini Analysis[/]"
    return Panel(
        message,
        title=title,
        border_style="bright_magenta",
        box=ROUNDED,
    )


def create_dataset_panel(
    file_path: str,
    total_samples: int,
    train_samples: int,
    val_samples: int,
    avg_tokens: float,
    quality_score: float,
) -> Panel:
    """Create a dataset summary panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="forge.muted")
    table.add_column("Value", style="bold")
    
    table.add_row("📁 File", file_path)
    table.add_row("📊 Total Samples", f"{total_samples:,}")
    table.add_row("🎯 Train / Val", f"{train_samples:,} / {val_samples:,}")
    table.add_row("📝 Avg Tokens", f"{avg_tokens:.0f}")
    
    quality_color = "green" if quality_score >= 0.8 else "yellow" if quality_score >= 0.5 else "red"
    table.add_row("⭐ Quality", f"[{quality_color}]{quality_score:.0%}[/]")
    
    return Panel(
        table,
        title="[forge.title]📋 Dataset Summary[/]",
        border_style="green",
        box=ROUNDED,
    )


def display_panel(panel: Panel) -> None:
    """Display a panel to the console."""
    console.print(panel)
