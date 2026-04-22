"""Stage 4: Data Validation (Format & Structure Only)."""

import json
from pathlib import Path
from typing import Dict, List
import random

from rich.console import Console

from nnb.gemini_brain.client import GeminiClient
from nnb.models.project_spec import ValidationResult, ValidationIssue
from nnb.utils.logging import get_logger
from nnb.utils.steering_loader import SteeringLoader

console = Console()
logger = get_logger(__name__)


DATA_VALIDATION_PROMPT = """
{steering_context}

================================================================================
YOUR TASK: DATA VALIDATION (FORMAT & STRUCTURE ONLY)
================================================================================

You are a data validation expert.

Project type: {project_type}
Expected format: {expected_format}

Data requirements:
{data_requirements}

Data sample information:
{data_snapshot}

Since the data is already CLEANED, validate ONLY:
1. File format matches requirements (extensions, structure)
2. Folder structure matches expected layout
3. File naming conventions are consistent
4. For images: check if files can be opened and have reasonable dimensions
5. For CSV/tabular: check if required columns exist
6. Label/class distribution (just report, don't flag as error)

DO NOT check for:
- Data quality issues (already cleaned)
- Corrupt files (already cleaned)
- Missing values (already cleaned)

REMEMBER:
- Data will be mounted READ-ONLY at /data in container
- Focus on structure and format, not quality
- Provide actionable fixes for any issues

Output format (JSON):
{{
  "status": "pass|warn|fail",
  "issues": [
    {{"severity": "error|warning", "message": "...", "fix": "..."}}
  ],
  "class_distribution": {{}},
  "total_samples": 0,
  "estimated_training_time": "..."
}}
"""


def sample_data(data_path: Path, sample_size: int = 50) -> Dict:
    """Sample data for validation without loading everything."""
    
    all_files = list(data_path.rglob("*"))
    all_files = [f for f in all_files if f.is_file()]
    
    # Get file extensions
    extensions = {}
    for f in all_files:
        ext = f.suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1
    
    # Get directory structure
    directories = {}
    for f in all_files:
        rel_path = f.relative_to(data_path)
        dir_name = str(rel_path.parent)
        directories[dir_name] = directories.get(dir_name, 0) + 1
    
    # Sample files
    sample_files = random.sample(all_files, min(sample_size, len(all_files)))
    
    return {
        "total_files": len(all_files),
        "extensions": extensions,
        "directories": directories,
        "sample_files": [str(f.relative_to(data_path)) for f in sample_files[:10]],
        "directory_structure": str(data_path),
    }


def validate_data(project: "Project", data_path: Path) -> ValidationResult:  # noqa: F821
    """Validate training data format and structure."""
    
    try:
        console.print("🔍 Sampling data...")
        
        # Sample data
        snapshot = sample_data(data_path)
        
        console.print(f"  Found {snapshot['total_files']} files")
        console.print(f"  Extensions: {snapshot['extensions']}")
        console.print(f"  Directories: {len(snapshot['directories'])}")
        
        console.print("\n🤖 Validating with Gemini...")
        
        # Get validation from Gemini
        gemini = GeminiClient()
        
        # Include steering context
        steering_context = SteeringLoader.format_for_prompt("data_validation")
        
        requirements = project.data_requirements_file.read_text() if project.data_requirements_file.exists() else "See project spec"
        
        prompt = DATA_VALIDATION_PROMPT.format(
            steering_context=steering_context,
            project_type=project._spec.project_type,
            expected_format=snapshot['extensions'],
            data_requirements=requirements,
            data_snapshot=json.dumps(snapshot, indent=2)
        )
        
        response = gemini.generate_json(prompt, temperature=0.2)
        
        # Parse response
        issues = [ValidationIssue(**issue) for issue in response.get("issues", [])]
        
        result = ValidationResult(
            status=response["status"],
            issues=issues,
            class_distribution=response.get("class_distribution"),
            estimated_training_time=response.get("estimated_training_time"),
            total_samples=snapshot["total_files"]
        )
        
        # Save data manifest
        manifest = {
            "data_path": str(data_path),
            "total_samples": snapshot["total_files"],
            "extensions": snapshot["extensions"],
            "directories": snapshot["directories"],
            "validation_result": result.dict(),
        }
        
        with open(project.data_manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Data validation: {result.status}, {result.total_samples} samples")
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating data: {e}", exc_info=True)
        console.print(f"[red]❌ Error: {e}[/red]")
        raise
