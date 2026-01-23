"""
forge prepare - Auto-generate and run preprocessing scripts via Gemini.

Features:
- Isolated sandbox venv for script execution
- Auto-installs any dependencies Gemini needs
- Self-healing on errors
- Cleanup after completion
"""

import json
import subprocess
import sys
import re
import shutil
import tempfile
import venv
import platform
from pathlib import Path
from typing import Optional, Tuple, List

import typer
from rich.syntax import Syntax
from rich.panel import Panel
from rich.prompt import Confirm

from forge.ui.console import console, print_success, print_error, print_info, print_warning, print_gemini
from forge.ui.panels import create_gemini_panel, display_panel
from forge.ui.progress import thinking_spinner, spinner
from forge.core.security import analyze_script


MAX_HEAL_ATTEMPTS = 3
SANDBOX_DIR = ".forge-sandbox"


PREPROCESSING_PROMPT = '''You are an expert ML data engineer. Generate a Python preprocessing script.

## Dataset Analysis
{analysis}

## Dataset Info
- Path: {dataset_path}
- Format: {format}
- Columns: {columns}
- Num Samples: {num_samples}

## Requirements
Generate a COMPLETE Python script that:
1. Loads the dataset from: {dataset_path}
2. Handles missing values and removes duplicates
3. Cleans and normalizes text fields
4. Creates a 'text' column formatted for LLM chat fine-tuning
5. Splits into train/validation (90/10)
6. Saves to {output_dir}/processed_train.jsonl and {output_dir}/processed_val.jsonl
7. Prints progress messages

## Output JSONL format (each line):
{{"text": "<|im_start|>user\\n{{query}}<|im_end|>\\n<|im_start|>assistant\\n{{response}}<|im_end|>"}}

## Libraries Available
You can use ANY Python libraries. Common ones:
- pandas, numpy, sklearn, json, os, random
Start with required imports.

Output ONLY the Python script in ```python``` blocks.
'''


HEAL_PROMPT = '''The preprocessing script failed:

```
{error}
```

## Script:
```python
{script}
```

## Fix the error. Common issues:
- ModuleNotFoundError: Add the import, we'll install it
- FileNotFoundError: Check the path
- KeyError: Check column names

Output the FIXED script in ```python``` blocks.
'''


def prepare_command(
    path: Path = typer.Argument(..., help="Path to the dataset file"),
    analysis_file: Optional[Path] = typer.Option(None, "--analysis", "-a", help="Previous analysis JSON"),
    output_dir: Path = typer.Option(Path("./data"), "--output", "-o", help="Output directory"),
    auto_heal: bool = typer.Option(True, "--heal/--no-heal", help="Auto-fix errors"),
    keep_sandbox: bool = typer.Option(False, "--keep-sandbox", help="Don't delete sandbox after"),
):
    """
    Auto-generate and run preprocessing scripts using Gemini.
    
    Creates an isolated sandbox environment for script execution.
    """
    if not path.exists():
        print_error(f"Dataset not found: {path}")
        raise typer.Exit(1)
    
    console.print(f"\n[bold]🔧 Preparing dataset:[/] {path}\n")
    
    # Get analysis
    metadata, analysis_text = _get_analysis(path, analysis_file)
    
    console.print(f"  Samples: [cyan]{metadata.get('num_samples', 'unknown')}[/]")
    console.print(f"  Columns: [cyan]{', '.join(metadata.get('columns', []))}[/]\n")
    
    # Generate script
    script_content = _generate_script(metadata, analysis_text, path, output_dir)
    
    if not script_content:
        raise typer.Exit(1)
    
    # Show script preview
    console.print("[bold]Generated Script:[/]\n")
    _display_script(script_content)
    
    # Security check
    security_report = analyze_script(script_content)
    console.print(f"\nRisk Level: [{'green' if security_report.is_safe else 'red'}]{security_report.risk_level.upper()}[/]\n")
    
    if not security_report.is_safe:
        if not Confirm.ask("[yellow]Script has security concerns. Continue?[/]", default=False):
            return
    
    if not Confirm.ask("[bold]Execute in sandbox?[/]", default=True):
        # Save script for manual execution
        script_path = Path("./preprocess_data.py")
        with open(script_path, "w") as f:
            f.write(script_content)
        print_info(f"Script saved to {script_path}")
        return
    
    # Create sandbox and execute
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = _run_in_sandbox(
        script_content, 
        output_dir,
        auto_heal,
        keep_sandbox
    )
    
    if success:
        console.print()
        print_success("✨ Preprocessing complete!")
        _show_output_stats(output_dir)
        console.print("\n[dim]Next: forge plan \"your goal\" --data ./data/processed_train.jsonl[/]")
    else:
        print_error("Preprocessing failed")
    
    console.print()


def _get_analysis(path: Path, analysis_file: Optional[Path]) -> Tuple[dict, str]:
    """Get dataset analysis."""
    if analysis_file and analysis_file.exists():
        with open(analysis_file) as f:
            data = json.load(f)
        return data.get("metadata", {}), data.get("analysis", "")
    
    from forge.cli.study_cmd import _collect_metadata
    
    with spinner("Scanning dataset..."):
        metadata = _collect_metadata(path)
    
    if metadata.get("error"):
        print_error(f"Failed to read: {metadata['error']}")
        raise typer.Exit(1)
    
    try:
        with thinking_spinner("Gemini analyzing..."):
            from forge.brain.client import create_brain
            brain = create_brain()
            response = brain.analyze_dataset(metadata)
        return metadata, response.text
    except Exception as e:
        print_warning(f"Analysis failed: {e}")
        return metadata, ""


def _generate_script(metadata: dict, analysis: str, path: Path, output_dir: Path) -> Optional[str]:
    """Generate preprocessing script via Gemini."""
    print_info("Generating preprocessing script...")
    
    try:
        with thinking_spinner("Gemini writing script..."):
            from forge.brain.client import create_brain
            brain = create_brain()
            
            prompt = PREPROCESSING_PROMPT.format(
                analysis=analysis[:2000],
                dataset_path=str(path.resolve()),
                format=metadata.get("format", "csv"),
                columns=metadata.get("columns", []),
                num_samples=metadata.get("num_samples", 0),
                output_dir=str(output_dir.resolve()),
            )
            
            response = brain.reason_sync(prompt)
        
        script = _extract_python_code(response.text)
        if script:
            print_success("Script generated!")
        return script
        
    except Exception as e:
        print_error(f"Generation failed: {e}")
        return None


def _run_in_sandbox(
    script_content: str,
    output_dir: Path,
    auto_heal: bool,
    keep_sandbox: bool,
) -> bool:
    """Run script in isolated sandbox venv."""
    
    sandbox_path = Path(SANDBOX_DIR).resolve()
    
    console.print()
    print_info(f"Creating sandbox environment: {sandbox_path}")
    
    try:
        # Create sandbox venv
        with spinner("Creating isolated venv..."):
            if sandbox_path.exists():
                shutil.rmtree(sandbox_path)
            venv.create(sandbox_path, with_pip=True)
        
        # Get paths
        if platform.system() == "Windows":
            pip_path = sandbox_path / "Scripts" / "pip.exe"
            python_path = sandbox_path / "Scripts" / "python.exe"
        else:
            pip_path = sandbox_path / "bin" / "pip"
            python_path = sandbox_path / "bin" / "python"
        
        # Install base dependencies
        with spinner("Installing pandas..."):
            subprocess.run(
                [str(pip_path), "install", "pandas", "-q"],
                capture_output=True, timeout=120
            )
        
        print_success("Sandbox ready!")
        
        # Execute with healing loop
        current_script = script_content
        script_path = sandbox_path / "preprocess.py"
        
        for attempt in range(MAX_HEAL_ATTEMPTS):
            # Save current script
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(current_script)
            
            # Detect and install dependencies
            deps = _extract_imports(current_script)
            if deps:
                with spinner(f"Installing dependencies: {', '.join(deps)}..."):
                    for dep in deps:
                        subprocess.run(
                            [str(pip_path), "install", dep, "-q"],
                            capture_output=True, timeout=60
                        )
            
            # Run script
            console.print()
            if attempt > 0:
                print_info(f"Retry attempt {attempt + 1}/{MAX_HEAL_ATTEMPTS}")
            else:
                print_info("Executing in sandbox...")
            console.print()
            
            result = subprocess.run(
                [str(python_path), str(script_path)],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Path.cwd()),
            )
            
            if result.returncode == 0:
                if result.stdout:
                    console.print(result.stdout)
                return True
            
            # Failed
            error = result.stderr or result.stdout
            console.print(f"[red]{error[-1500:]}[/]")
            
            if not auto_heal or attempt >= MAX_HEAL_ATTEMPTS - 1:
                break
            
            # Self-heal
            console.print()
            print_warning("Script failed! Self-healing...")
            
            fixed = _heal_script(current_script, error)
            if not fixed:
                break
            
            current_script = fixed
            console.print("[bold]Fixed script:[/]\n")
            _display_script(fixed)
        
        return False
        
    finally:
        # Cleanup sandbox
        if not keep_sandbox and sandbox_path.exists():
            console.print()
            with spinner("Cleaning up sandbox..."):
                shutil.rmtree(sandbox_path, ignore_errors=True)
            print_info("Sandbox removed")


def _extract_imports(script: str) -> List[str]:
    """Extract pip package names from import statements."""
    deps = set()
    
    # Common import -> pip package mappings
    pip_names = {
        "sklearn": "scikit-learn",
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "yaml": "pyyaml",
    }
    
    # Find imports
    import_pattern = r"^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    
    for line in script.split("\n"):
        match = re.match(import_pattern, line.strip())
        if match:
            module = match.group(1)
            # Skip standard library
            if module in ["os", "sys", "json", "re", "random", "math", "datetime", 
                         "collections", "itertools", "functools", "pathlib", "typing"]:
                continue
            # Skip already installed
            if module in ["pandas", "numpy"]:
                continue
            
            pkg = pip_names.get(module, module)
            deps.add(pkg)
    
    return list(deps)


def _heal_script(script: str, error: str) -> Optional[str]:
    """Use Gemini to fix script errors."""
    try:
        with thinking_spinner("Gemini fixing..."):
            from forge.brain.client import create_brain
            brain = create_brain()
            
            prompt = HEAL_PROMPT.format(
                error=error[-1500:],
                script=script,
            )
            
            response = brain.reason_sync(prompt)
        
        fixed = _extract_python_code(response.text)
        if fixed:
            print_success("Fix generated!")
        return fixed
        
    except Exception as e:
        print_error(f"Healing failed: {e}")
        return None


def _display_script(script: str) -> None:
    """Display script preview."""
    lines = script.split('\n')
    preview = '\n'.join(lines[:40])
    if len(lines) > 40:
        preview += f'\n... ({len(lines) - 40} more lines)'
    
    syntax = Syntax(preview, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, border_style="cyan"))


def _show_output_stats(output_dir: Path) -> None:
    """Show output file stats."""
    for name in ["processed_train.jsonl", "processed_val.jsonl"]:
        f = output_dir / name
        if f.exists():
            count = sum(1 for _ in open(f, encoding="utf-8"))
            console.print(f"  {name}: [cyan]{count} samples[/]")


def _extract_python_code(text: str) -> Optional[str]:
    """Extract Python code from response."""
    patterns = [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    if "import " in text and ("pd." in text or "pandas" in text):
        return "\n".join(l for l in text.split("\n") if not l.startswith("```")).strip()
    
    return None
