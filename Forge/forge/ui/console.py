"""
Rich console wrapper for consistent terminal UI across Forge.
"""

from rich.console import Console
from rich.theme import Theme

# Custom Forge theme with vibrant colors
FORGE_THEME = Theme({
    "forge.title": "bold bright_cyan",
    "forge.success": "bold green",
    "forge.warning": "bold yellow",
    "forge.error": "bold red",
    "forge.info": "dim cyan",
    "forge.highlight": "bold magenta",
    "forge.muted": "dim white",
    "forge.gpu": "bold bright_green",
    "forge.vram": "bold bright_yellow",
    "forge.progress": "bold blue",
    "forge.gemini": "bold bright_magenta",
})

# Global console instance
console = Console(theme=FORGE_THEME)


def print_banner() -> None:
    """Print the Forge ASCII banner."""
    banner = """
[forge.title]
    ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
    ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
    █████╗  ██║   ██║██████╔╝██║  ███╗█████╗  
    ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  
    ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
    ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
[/]
[forge.muted]    Local-First SLM Fine-Tuning with Gemini 3[/]
"""
    console.print(banner)


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[forge.success]✓[/] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[forge.error]✗[/] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[forge.warning]⚠[/] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[forge.info]ℹ[/] {message}")


def print_step(step: int, total: int, message: str) -> None:
    """Print a numbered step."""
    console.print(f"[forge.progress][{step}/{total}][/] {message}")


def print_gemini(message: str) -> None:
    """Print a message from Gemini."""
    console.print(f"[forge.gemini]🧠 Gemini:[/] {message}")
