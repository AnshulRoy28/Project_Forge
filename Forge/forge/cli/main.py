"""
Forge CLI - Main entry point.

Docker-based CLI orchestration tool for fine-tuning Small Language Models.
"""

import typer
from typing import Optional
from pathlib import Path

from forge.ui.console import console, print_banner

# Create the main Typer app
app = typer.Typer(
    name="forge",
    help="🔥 Forge - Docker-based SLM fine-tuning with Gemini",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
):
    """
    🔥 Forge - Docker-based SLM fine-tuning with Gemini
    
    Train state-of-the-art language models on your GPU using Docker containers.
    """
    if version:
        from forge import __version__
        console.print(f"Forge CLI v{__version__}")
        raise typer.Exit()
    
    if ctx.invoked_subcommand is None:
        print_banner()
        console.print("\nRun [bold cyan]forge --help[/] for available commands.\n")


# Import and register commands
from forge.cli.init_cmd import init_command
from forge.cli.login_cmd import login_command
from forge.cli.status_cmd import status_command
from forge.cli.study_cmd import study_command
from forge.cli.prepare_cmd_v2 import prepare_command
from forge.cli.plan_cmd import plan_command
from forge.cli.train_cmd import train_command, train_internal_command
from forge.cli.inference_cmd import inference_command
from forge.cli.docker_cmd import docker_app
from forge.cli.export_cmd import export_app

# Project Setup
app.command(name="init", help="🔧 Initialize Forge (lightweight Docker-based setup)")(init_command)
app.command(name="login", help="🔐 Update or verify API credentials")(login_command)
app.command(name="status", help="📊 Display project state and health")(status_command)

# Data Engineering
app.command(name="study", help="📊 Analyze a dataset with Gemini")(study_command)
app.command(name="prepare", help="⚡ Model-aware preprocessing with Docker execution")(prepare_command)

# AI Planning
app.command(name="plan", help="📝 Generate training config from natural language")(plan_command)

# Training (Docker-based)
app.command(name="train", help="🔥 Start training via Docker container")(train_command)
app.command(name="train-internal", hidden=True)(train_internal_command)  # Internal: runs inside Docker

# Inference
app.command(name="inference", help="💬 Run inference on trained model")(inference_command)

# Docker & Export
app.add_typer(docker_app, name="docker", help="🐳 Docker container management")
app.add_typer(export_app, name="export", help="📦 Export models to various formats")


if __name__ == "__main__":
    app()

