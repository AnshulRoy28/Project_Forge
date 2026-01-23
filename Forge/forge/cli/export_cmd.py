"""
forge export - Export trained models to various formats.
"""

from pathlib import Path
from typing import Optional

import typer

from forge.ui.console import console, print_success, print_error, print_info, print_warning
from forge.ui.progress import spinner
from forge.core.config import load_config, get_config_path


export_app = typer.Typer(
    name="export",
    help="📦 Export trained models to various formats",
    no_args_is_help=True,
)


@export_app.command("lora")
def export_lora(
    output_dir: Path = typer.Option(Path("./output"), "--output", "-o", help="Output directory"),
):
    """
    Export LoRA adapter weights only.
    
    This is the lightest export option - only the adapter weights are saved.
    """
    console.print("\n[bold]📦 Exporting LoRA adapter[/]\n")
    
    # Check if model exists
    model_path = output_dir
    if not model_path.exists():
        print_error(f"Model not found: {model_path}")
        console.print("[dim]Train a model first with 'forge train'[/]")
        raise typer.Exit(1)
    
    # Check for adapter files
    adapter_files = list(model_path.glob("adapter_*.safetensors")) + list(model_path.glob("adapter_*.bin"))
    
    if adapter_files:
        print_success(f"LoRA adapter found: {model_path}")
        for f in adapter_files:
            console.print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print_warning("No LoRA adapter files found")
        console.print("[dim]Model may have been saved as merged weights[/]")


@export_app.command("merged")
def export_merged(
    output_dir: Path = typer.Option(Path("./output/merged"), "--output", "-o", help="Output directory"),
):
    """
    Export merged model with full weights.
    
    Merges LoRA adapter into base model for standalone deployment.
    """
    console.print("\n[bold]📦 Exporting merged model[/]\n")
    
    if output_dir.exists() and (output_dir / "config.json").exists():
        print_success(f"Merged model found: {output_dir}")
        
        # Show files
        files = list(output_dir.glob("*.safetensors")) + list(output_dir.glob("*.bin"))
        total_size = sum(f.stat().st_size for f in files) / (1024**3)
        console.print(f"  Total size: {total_size:.2f} GB")
    else:
        print_info("Merged model not found. Creating...")
        console.print("[dim]This requires loading the model. Use 'forge docker run <arch> train' to merge.[/]")


@export_app.command("gguf")
def export_gguf(
    output_dir: Path = typer.Option(Path("./output/gguf"), "--output", "-o", help="Output directory"),
    quantization: str = typer.Option("q4_k_m", "--quant", "-q", help="Quantization: q4_k_m, q5_k_m, q8_0, f16"),
):
    """
    Export to GGUF format for llama.cpp/Ollama.
    
    Quantizes the model for efficient local inference.
    """
    console.print(f"\n[bold]📦 Exporting to GGUF ({quantization})[/]\n")
    
    # Check for existing GGUF
    if output_dir.exists():
        gguf_files = list(output_dir.glob("*.gguf"))
        if gguf_files:
            print_success("GGUF files found:")
            for f in gguf_files:
                console.print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024 / 1024:.2f} GB)")
            return
    
    print_info("GGUF not found. Export happens during training.")
    console.print("[dim]Add 'gguf' to export_formats in forge.yaml and re-train[/]")


@export_app.command("ollama")
def export_ollama(
    model_name: Optional[str] = typer.Option(None, "--name", "-n", help="Model name in Ollama"),
    gguf_dir: Path = typer.Option(Path("./output/gguf"), "--gguf", "-g", help="GGUF directory"),
):
    """
    Register model with local Ollama instance.
    
    Creates a Modelfile and runs 'ollama create'.
    """
    import subprocess
    
    console.print("\n[bold]📦 Registering with Ollama[/]\n")
    
    # Check Ollama is running
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            print_error("Ollama is not running. Start with 'ollama serve'")
            raise typer.Exit(1)
    except FileNotFoundError:
        print_error("Ollama not found. Install from https://ollama.ai")
        raise typer.Exit(1)
    
    # Find GGUF file
    gguf_files = list(gguf_dir.glob("*.gguf")) if gguf_dir.exists() else []
    
    if not gguf_files:
        print_error(f"No GGUF files found in {gguf_dir}")
        console.print("[dim]Run 'forge export gguf' first[/]")
        raise typer.Exit(1)
    
    gguf_file = gguf_files[0]
    
    # Determine model name
    if not model_name:
        config_path = get_config_path()
        if config_path.exists():
            config = load_config(config_path)
            model_name = config.name.replace(" ", "-").lower()
        else:
            model_name = "forge-model"
    
    # Create Modelfile
    modelfile_path = gguf_dir / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(f"FROM {gguf_file.absolute()}\n")
        f.write("\nPARAMETER temperature 0.7\n")
        f.write("PARAMETER top_p 0.9\n")
    
    console.print(f"Created Modelfile: {modelfile_path}")
    
    # Register with Ollama
    with spinner(f"Registering as '{model_name}'..."):
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
        )
    
    if result.returncode == 0:
        print_success(f"Model registered: {model_name}")
        console.print()
        console.print(f"[dim]Test with: ollama run {model_name}[/]")
    else:
        print_error(f"Registration failed: {result.stderr}")
        raise typer.Exit(1)
