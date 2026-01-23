"""
Enhanced forge prepare command - Model-aware preprocessing with Docker execution.

This version integrates with the plan configuration to provide model-aware
preprocessing using Docker containers instead of venv sandboxes.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import typer
from rich.prompt import Confirm

from forge.ui.console import console, print_success, print_error, print_info, print_warning
from forge.ui.progress import thinking_spinner, spinner
from forge.core.config_v2 import ForgeConfig
from forge.core.docker_preprocessor import DockerPreprocessor, ModelAwareScriptGenerator
from forge.core.templates import ChatTemplateRegistry, TemplateFormatter
from forge.core.security import analyze_script


def prepare_command_v2(
    path: Path = typer.Argument(..., help="Path to the dataset file"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to forge.yaml config"),
    output_dir: Path = typer.Option(Path("./data"), "--output", "-o", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate script without executing"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompts"),
):
    """
    Model-aware data preprocessing with Docker execution.
    
    This enhanced version:
    - Reads model configuration from forge.yaml (created by 'forge plan')
    - Uses the correct chat template for the target model
    - Executes preprocessing in GPU-optimized Docker containers
    - Applies ML engineering best practices
    
    Example:
        forge prepare ./data/conversations.csv
        forge prepare ./data/code_examples.json --output ./processed
    """
    
    # Check if dataset exists
    if not path.exists():
        print_error(f"Dataset not found: {path}")
        raise typer.Exit(1)
    
    # Load configuration
    config_path = config_file or Path("forge.yaml")
    if not config_path.exists():
        print_error("No configuration found. Run 'forge plan' first to create forge.yaml")
        console.print()
        console.print("[dim]Example: forge plan \"Create a helpful coding assistant\"[/]")
        raise typer.Exit(1)
    
    try:
        config = ForgeConfig.load(config_path)
    except Exception as e:
        print_error(f"Failed to load configuration: {e}")
        raise typer.Exit(1)
    
    console.print(f"\n[bold]🔧 Model-Aware Preprocessing:[/] {path}\n")
    
    # Display configuration info
    _display_config_info(config)
    
    # Analyze dataset
    console.print(f"[bold]📊 Dataset Analysis:[/]")
    metadata, analysis_text = _get_analysis(path)
    
    console.print(f"  Samples: [cyan]{metadata.get('num_samples', 'unknown')}[/]")
    console.print(f"  Format: [cyan]{metadata.get('format', 'unknown')}[/]")
    console.print(f"  Columns: [cyan]{', '.join(metadata.get('columns', []))}[/]")
    console.print()
    
    # Validate template compatibility
    _validate_template_compatibility(config, metadata)
    
    # Generate preprocessing script
    script_generator = ModelAwareScriptGenerator(config)
    
    with thinking_spinner("Generating model-aware preprocessing script..."):
        script_content = script_generator.generate_script(
            metadata, analysis_text, path, output_dir
        )
    
    if not script_content:
        print_error("Failed to generate preprocessing script")
        raise typer.Exit(1)
    
    print_success("Preprocessing script generated!")
    
    # Show script preview
    if not force:
        _display_script_preview(script_content)
        console.print()
    
    # Security analysis
    security_report = analyze_script(script_content)
    console.print(f"Security Analysis: [{'green' if security_report.is_safe else 'red'}]{security_report.risk_level.upper()}[/]")
    
    if not security_report.is_safe and not force:
        console.print("[yellow]Security concerns detected:[/]")
        for concern in security_report.concerns:
            console.print(f"  • {concern}")
        console.print()
        
        if not Confirm.ask("Continue with execution?", default=False):
            print_info("Preprocessing cancelled by user")
            return
    
    if dry_run:
        # Save script for manual inspection
        script_path = Path("./preprocess_generated.py")
        with open(script_path, "w") as f:
            f.write(script_content)
        print_info(f"Script saved to {script_path} (dry run mode)")
        return
    
    # Confirm execution
    if not force:
        console.print(f"[bold]Ready to preprocess data using:[/]")
        console.print(f"  • Model: [cyan]{config.model.name}[/]")
        console.print(f"  • Template: [cyan]{config.model.chat_template.value}[/]")
        console.print(f"  • Container: [cyan]forge:{config.hardware.gpu_arch.value}[/]")
        console.print()
        
        if not Confirm.ask("Execute preprocessing in Docker container?", default=True):
            print_info("Preprocessing cancelled by user")
            return
    
    # Execute preprocessing
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = DockerPreprocessor(config)
    success = preprocessor.execute_preprocessing(script_content, path, output_dir)
    
    if success:
        console.print()
        print_success("✨ Model-aware preprocessing complete!")
        _show_output_stats(output_dir, config)
        
        console.print()
        console.print("[dim]Next steps:[/]")
        console.print("  1. Review the processed data files")
        console.print("  2. Run [cyan]forge train[/] to start training")
        console.print()
    else:
        print_error("Preprocessing failed")
        console.print()
        console.print("[dim]Troubleshooting:[/]")
        console.print("  • Check that Docker Desktop is running")
        console.print(f"  • Ensure container 'forge:{config.hardware.gpu_arch.value}' exists")
        console.print("  • Run [cyan]forge docker build[/] if needed")
        console.print()


def _display_config_info(config: ForgeConfig):
    """Display configuration information."""
    console.print(f"[bold]🤖 Target Model:[/]")
    console.print(f"  Model: [cyan]{config.model.name}[/]")
    console.print(f"  Template: [cyan]{config.model.chat_template.value}[/]")
    console.print(f"  Max Length: [cyan]{config.model.max_length}[/]")
    
    # Show template format
    template_config = ChatTemplateRegistry.get_template(config.model.chat_template)
    console.print(f"  Format: [dim]{template_config['description']}[/]")
    console.print()


def _get_analysis(path: Path) -> Tuple[dict, str]:
    """Get dataset analysis."""
    from forge.cli.study_cmd import _collect_metadata
    
    with spinner("Analyzing dataset structure..."):
        metadata = _collect_metadata(path)
    
    if metadata.get("error"):
        print_error(f"Failed to read dataset: {metadata['error']}")
        raise typer.Exit(1)
    
    # Try to get Gemini analysis
    analysis_text = ""
    try:
        with thinking_spinner("Getting AI analysis..."):
            from forge.brain.client import create_brain
            brain = create_brain()
            response = brain.analyze_dataset(metadata)
            analysis_text = response.text
    except Exception as e:
        print_warning(f"AI analysis unavailable: {e}")
    
    return metadata, analysis_text


def _validate_template_compatibility(config: ForgeConfig, metadata: dict):
    """Validate that the dataset can work with the selected template."""
    
    template_config = ChatTemplateRegistry.get_template(config.model.chat_template)
    required_fields = template_config["required_fields"]
    available_columns = metadata.get("columns", [])
    
    console.print(f"[bold]🔍 Template Validation:[/]")
    console.print(f"  Required Fields: [cyan]{', '.join(required_fields)}[/]")
    console.print(f"  Available Columns: [cyan]{', '.join(available_columns)}[/]")
    
    # Check if we have the required fields or can map them
    missing_fields = []
    for field in required_fields:
        if field not in available_columns:
            # Check for common mappings
            field_mappings = {
                "query": ["question", "input", "instruction", "prompt", "user"],
                "response": ["answer", "output", "completion", "assistant", "reply"],
            }
            
            if field in field_mappings:
                mapped = any(alt in available_columns for alt in field_mappings[field])
                if not mapped:
                    missing_fields.append(field)
            else:
                missing_fields.append(field)
    
    if missing_fields:
        console.print(f"  [yellow]⚠ Missing: {', '.join(missing_fields)}[/]")
        console.print("  [dim]The preprocessing script will attempt to map or create these fields[/]")
    else:
        console.print("  [green]✓ All required fields available or mappable[/]")
    
    console.print()


def _display_script_preview(script_content: str):
    """Display a preview of the generated script."""
    from rich.syntax import Syntax
    from rich.panel import Panel
    
    lines = script_content.split('\n')
    preview_lines = lines[:30]  # Show first 30 lines
    
    if len(lines) > 30:
        preview_lines.append(f"... ({len(lines) - 30} more lines)")
    
    preview = '\n'.join(preview_lines)
    
    syntax = Syntax(preview, "python", theme="monokai", line_numbers=True)
    console.print(Panel(
        syntax, 
        title="Generated Preprocessing Script (Preview)",
        border_style="cyan"
    ))


def _show_output_stats(output_dir: Path, config: ForgeConfig):
    """Show statistics about the processed output files."""
    
    train_file = output_dir / "processed_train.jsonl"
    val_file = output_dir / "processed_val.jsonl"
    
    console.print(f"[bold]📈 Output Statistics:[/]")
    
    if train_file.exists():
        train_count = sum(1 for _ in open(train_file, encoding="utf-8"))
        console.print(f"  Training samples: [cyan]{train_count:,}[/]")
        
        # Show sample
        with open(train_file, 'r', encoding="utf-8") as f:
            sample = json.loads(f.readline())
            sample_text = sample.get("text", "")[:100] + "..." if len(sample.get("text", "")) > 100 else sample.get("text", "")
            console.print(f"  Sample format: [dim]{sample_text}[/]")
    
    if val_file.exists():
        val_count = sum(1 for _ in open(val_file, encoding="utf-8"))
        console.print(f"  Validation samples: [cyan]{val_count:,}[/]")
    
    # Show split ratios
    if train_file.exists() and val_file.exists():
        total = train_count + val_count
        train_ratio = train_count / total if total > 0 else 0
        val_ratio = val_count / total if total > 0 else 0
        console.print(f"  Actual split: [cyan]{train_ratio:.1%} / {val_ratio:.1%}[/]")
        console.print(f"  Target split: [dim]{config.preprocessing.train_split:.1%} / {config.preprocessing.validation_split:.1%}[/]")
    
    console.print(f"  Output directory: [cyan]{output_dir.resolve()}[/]")


# For backward compatibility, we can alias the new command
prepare_command = prepare_command_v2