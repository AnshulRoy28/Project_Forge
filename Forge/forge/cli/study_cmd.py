"""
forge study - Analyze a dataset with Gemini.
"""

import os
import json
from pathlib import Path
from typing import Optional

import typer

from forge.ui.console import console, print_success, print_error, print_info, print_warning
from forge.ui.panels import create_dataset_panel, create_gemini_panel, display_panel
from forge.ui.progress import spinner, thinking_spinner


def study_command(
    path: Path = typer.Argument(..., help="Path to the dataset file or directory"),
    deep_audit: bool = typer.Option(False, "--deep", "-d", help="Enable deep data audit (sends samples to Gemini)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save analysis to file"),
):
    """
    Analyze a dataset for quality, bias, and formatting issues.
    
    By default, only metadata is sent to Gemini (privacy-safe).
    Use --deep to enable full content analysis.
    """
    # Validate path
    if not path.exists():
        print_error(f"Path not found: {path}")
        raise typer.Exit(1)
    
    console.print(f"\n[bold]📊 Analyzing dataset:[/] {path}\n")
    
    # Collect metadata
    with spinner("Scanning dataset..."):
        metadata = _collect_metadata(path)
    
    if metadata["error"]:
        print_error(f"Failed to read dataset: {metadata['error']}")
        raise typer.Exit(1)
    
    # Display basic stats
    display_panel(create_dataset_panel(
        file_path=str(path.name),
        total_samples=metadata["num_samples"],
        train_samples=int(metadata["num_samples"] * 0.9),
        val_samples=int(metadata["num_samples"] * 0.1),
        avg_tokens=metadata.get("avg_length", 0) * 0.75,  # Rough token estimate
        quality_score=0.0,  # Will be set by Gemini
    ))
    
    console.print()
    
    if deep_audit:
        print_warning("Deep audit enabled - sample data will be sent to Gemini.")
        console.print()
    
    # Get Gemini analysis
    try:
        with thinking_spinner("Gemini is analyzing your dataset..."):
            from forge.brain.client import create_brain
            
            brain = create_brain()
            
            if deep_audit:
                # Include sample data for deep analysis
                metadata["samples"] = _get_sample_data(path, n=5)
            
            response = brain.analyze_dataset(metadata)
        
        # Display Gemini's analysis
        display_panel(create_gemini_panel(response.text))
        
        # Save if requested
        if output:
            _save_analysis(output, metadata, response.text)
            print_success(f"Analysis saved to {output}")
        
    except Exception as e:
        print_error(f"Gemini analysis failed: {e}")
        console.print("[dim]Basic metadata analysis completed. Gemini features unavailable.[/]")
    
    console.print()


def _collect_metadata(path: Path) -> dict:
    """Collect metadata from the dataset without loading full content."""
    metadata = {
        "filename": path.name,
        "file_size": 0,
        "format": "unknown",
        "num_samples": 0,
        "columns": [],
        "sample_lengths": {},
        "error": None,
    }
    
    try:
        if path.is_file():
            metadata["file_size"] = path.stat().st_size
            suffix = path.suffix.lower()
            
            if suffix == ".csv":
                metadata.update(_analyze_csv(path))
            elif suffix == ".json":
                metadata.update(_analyze_json(path))
            elif suffix == ".jsonl":
                metadata.update(_analyze_jsonl(path))
            elif suffix == ".txt":
                metadata.update(_analyze_txt(path))
            else:
                metadata["error"] = f"Unsupported format: {suffix}"
        
        elif path.is_dir():
            metadata.update(_analyze_directory(path))
        
    except Exception as e:
        metadata["error"] = str(e)
    
    return metadata


def _analyze_csv(path: Path) -> dict:
    """Analyze a CSV file."""
    import csv
    
    result = {"format": "csv", "columns": [], "num_samples": 0, "sample_lengths": {}}
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        
        # Get header
        try:
            header = next(reader)
            result["columns"] = header
        except StopIteration:
            return result
        
        # Count rows and sample lengths
        lengths = []
        for i, row in enumerate(reader):
            if row:
                text_len = sum(len(cell) for cell in row)
                lengths.append(text_len)
            
            if i >= 10000:  # Limit for large files
                break
        
        result["num_samples"] = len(lengths)
        
        if lengths:
            result["sample_lengths"] = {
                "min": min(lengths),
                "max": max(lengths),
                "avg": sum(lengths) / len(lengths),
            }
    
    return result


def _analyze_json(path: Path) -> dict:
    """Analyze a JSON file."""
    result = {"format": "json", "columns": [], "num_samples": 0, "sample_lengths": {}}
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        result["num_samples"] = len(data)
        
        if data and isinstance(data[0], dict):
            result["columns"] = list(data[0].keys())
            
            lengths = [len(json.dumps(item)) for item in data[:1000]]
            if lengths:
                result["sample_lengths"] = {
                    "min": min(lengths),
                    "max": max(lengths),
                    "avg": sum(lengths) / len(lengths),
                }
    
    elif isinstance(data, dict):
        result["columns"] = list(data.keys())
        result["num_samples"] = 1
    
    return result


def _analyze_jsonl(path: Path) -> dict:
    """Analyze a JSONL file."""
    result = {"format": "jsonl", "columns": [], "num_samples": 0, "sample_lengths": {}}
    
    lengths = []
    columns_found = set()
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    columns_found.update(obj.keys())
                lengths.append(len(line))
            except json.JSONDecodeError:
                continue
            
            if i >= 10000:
                break
    
    result["num_samples"] = len(lengths)
    result["columns"] = list(columns_found)
    
    if lengths:
        result["sample_lengths"] = {
            "min": min(lengths),
            "max": max(lengths),
            "avg": sum(lengths) / len(lengths),
        }
    
    return result


def _analyze_txt(path: Path) -> dict:
    """Analyze a text file."""
    result = {"format": "txt", "columns": ["text"], "num_samples": 0, "sample_lengths": {}}
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Split by double newlines (paragraphs) or treat as single sample
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    
    if not paragraphs:
        paragraphs = [content.strip()] if content.strip() else []
    
    result["num_samples"] = len(paragraphs)
    
    lengths = [len(p) for p in paragraphs]
    if lengths:
        result["sample_lengths"] = {
            "min": min(lengths),
            "max": max(lengths),
            "avg": sum(lengths) / len(lengths),
        }
    
    return result


def _analyze_directory(path: Path) -> dict:
    """Analyze a directory of files."""
    result = {"format": "directory", "columns": [], "num_samples": 0}
    
    supported = {".csv", ".json", ".jsonl", ".txt"}
    files = [f for f in path.iterdir() if f.suffix.lower() in supported]
    
    result["num_samples"] = len(files)
    result["columns"] = [f.name for f in files[:10]]  # First 10 filenames
    
    return result


def _get_sample_data(path: Path, n: int = 5) -> list[str]:
    """Get sample data for deep audit (limited)."""
    samples = []
    
    if path.suffix.lower() == ".csv":
        import csv
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= n:
                    break
                samples.append(json.dumps(row))
    
    elif path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                samples.append(line.strip())
    
    return samples


def _save_analysis(output: Path, metadata: dict, analysis: str) -> None:
    """Save analysis results to a file."""
    result = {
        "metadata": metadata,
        "analysis": analysis,
    }
    
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
