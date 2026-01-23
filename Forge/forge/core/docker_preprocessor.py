"""
Docker-based preprocessing engine for model-aware data preparation.

This module provides Docker container execution for preprocessing scripts,
replacing the venv-based sandbox with GPU-optimized containers.
"""

import subprocess
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

from .config_v2 import ForgeConfig
from ..ui.console import console, print_success, print_error, print_info, print_warning
from ..ui.progress import spinner


class DockerPreprocessor:
    """Docker-based preprocessing engine."""
    
    def __init__(self, config: ForgeConfig):
        self.config = config
        self.container_name = f"forge:{config.hardware.gpu_arch.value}"
        self.temp_dir = None
    
    def execute_preprocessing(self, script_content: str, data_path: Path, output_dir: Path) -> bool:
        """Execute preprocessing script in Docker container."""
        
        # Check if Docker is running
        if not self._check_docker_running():
            print_error("Docker is not running. Please start Docker Desktop.")
            return False
        
        # Check if container exists
        if not self._check_container_exists():
            print_error(f"Container {self.container_name} not found. Run 'forge docker build' first.")
            return False
        
        # Create temporary directory for script and execution
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = Path(temp_dir)
            
            try:
                # Prepare script and environment
                script_path = self._prepare_script(script_content)
                
                # Execute in container
                return self._run_in_container(script_path, data_path, output_dir)
                
            except Exception as e:
                print_error(f"Preprocessing execution failed: {e}")
                return False
            finally:
                self.temp_dir = None
    
    def _check_docker_running(self) -> bool:
        """Check if Docker daemon is running."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _check_container_exists(self) -> bool:
        """Check if the required container image exists."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self.container_name],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _prepare_script(self, script_content: str) -> Path:
        """Prepare preprocessing script for container execution."""
        
        # Create script file
        script_path = self.temp_dir / "preprocess.py"
        
        # Add container-specific imports and setup
        enhanced_script = self._enhance_script_for_container(script_content)
        
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(enhanced_script)
        
        return script_path
    
    def _enhance_script_for_container(self, script_content: str) -> str:
        """Enhance script with container-specific setup and error handling."""
        
        enhanced_script = f'''#!/usr/bin/env python3
"""
Auto-generated preprocessing script for Docker execution.
Model: {self.config.model.name}
Template: {self.config.model.chat_template.value}
"""

import sys
import os
import json
import traceback
from pathlib import Path

# Container paths
DATA_DIR = Path("/data")
OUTPUT_DIR = Path("/output")
SCRIPT_DIR = Path("/tmp/forge_script")

# Configuration from forge.yaml
CONFIG = {{
    "model": {{
        "name": "{self.config.model.name}",
        "chat_template": "{self.config.model.chat_template.value}",
        "max_length": {self.config.model.max_length},
        "architecture": "{self.config.model.architecture}",
    }},
    "preprocessing": {{
        "train_split": {self.config.preprocessing.train_split},
        "validation_split": {self.config.preprocessing.validation_split},
        "chunk_size": {self.config.preprocessing.chunk_size},
        "min_text_length": {self.config.preprocessing.min_text_length},
        "max_text_length": {self.config.preprocessing.max_text_length},
        "remove_duplicates": {str(self.config.preprocessing.remove_duplicates)},
        "random_seed": {self.config.preprocessing.random_seed},
    }}
}}

def log_progress(message: str):
    """Log progress message."""
    print(f"[FORGE] {{message}}", flush=True)

def log_error(message: str):
    """Log error message."""
    print(f"[FORGE ERROR] {{message}}", file=sys.stderr, flush=True)

def main():
    """Main preprocessing function."""
    try:
        log_progress("Starting model-aware preprocessing...")
        log_progress(f"Model: {{CONFIG['model']['name']}}")
        log_progress(f"Template: {{CONFIG['model']['chat_template']}}")
        
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Execute preprocessing logic
        execute_preprocessing()
        
        log_progress("Preprocessing completed successfully!")
        return 0
        
    except Exception as e:
        log_error(f"Preprocessing failed: {{e}}")
        log_error(f"Traceback: {{traceback.format_exc()}}")
        return 1

def execute_preprocessing():
    """Execute the main preprocessing logic."""
    import pandas as pd
    import json
    import random
    from pathlib import Path
    
{script_content}

if __name__ == "__main__":
    sys.exit(main())
'''
        
        return enhanced_script
    
    def _run_in_container(self, script_path: Path, data_path: Path, output_dir: Path) -> bool:
        """Run preprocessing script in Docker container."""
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build Docker command
        cmd = [
            "docker", "run",
            "--rm",
            "--gpus", "all" if self.config.hardware.use_gpu_preprocessing else "0",
            "-v", f"{data_path.parent.resolve()}:/data",
            "-v", f"{output_dir.resolve()}:/output", 
            "-v", f"{script_path.parent.resolve()}:/tmp/forge_script",
            "-w", "/tmp/forge_script",
            "--entrypoint", "python3",  # Override entrypoint to use python3 directly
            self.container_name,
            "preprocess.py"
        ]
        
        console.print()
        print_info(f"Executing preprocessing in {self.container_name} container...")
        console.print(f"[dim]$ {' '.join(cmd[:-1])} preprocess.py[/]")
        console.print()
        
        # Execute with real-time output
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Parse and display Forge log messages
                    if output.startswith('[FORGE]'):
                        message = output[7:].strip()
                        console.print(f"  {message}")
                    elif output.startswith('[FORGE ERROR]'):
                        message = output[13:].strip()
                        console.print(f"  [red]Error: {message}[/]")
                    else:
                        # Regular output
                        console.print(f"  [dim]{output.strip()}[/]")
            
            # Wait for completion
            return_code = process.poll()
            
            if return_code == 0:
                print_success("Preprocessing completed successfully!")
                return True
            else:
                print_error(f"Preprocessing failed with exit code {return_code}")
                return False
                
        except subprocess.TimeoutExpired:
            print_error("Preprocessing timed out")
            return False
        except Exception as e:
            print_error(f"Container execution failed: {e}")
            return False
    
    def get_container_info(self) -> Dict[str, Any]:
        """Get information about the container."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self.container_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                info = json.loads(result.stdout)[0]
                return {
                    "name": self.container_name,
                    "created": info.get("Created", "Unknown"),
                    "size": info.get("Size", 0),
                    "architecture": info.get("Architecture", "Unknown"),
                    "exists": True
                }
            else:
                return {"name": self.container_name, "exists": False}
                
        except Exception:
            return {"name": self.container_name, "exists": False}


class ModelAwareScriptGenerator:
    """Enhanced script generator with model-aware templates and optimizations."""
    
    def __init__(self, config: ForgeConfig):
        self.config = config
    
    def generate_script(self, metadata: Dict, analysis: str, data_path: Path, output_dir: Path) -> str:
        """Generate model-aware preprocessing script."""
        
        # Get template information
        from .templates import ChatTemplateRegistry
        template_config = ChatTemplateRegistry.get_template(self.config.model.chat_template)
        template_string = template_config["template"]
        required_fields = template_config["required_fields"]
        
        # Create enhanced prompt with model context
        prompt = self._create_model_aware_prompt(
            metadata, analysis, data_path, output_dir, 
            template_string, required_fields
        )
        
        # Generate script using template (Gemini disabled for now)
        return self._generate_template_script(metadata, data_path, output_dir)
        
        # Generate script using Gemini
        try:
            from ..brain.client import create_brain
            brain = create_brain()
            response = brain.reason_sync(prompt)
            
            # Extract Python code
            script = self._extract_python_code(response.text)
            if script and len(script.strip()) > 100:  # Only use if substantial
                return script
            else:
                # Fallback to template-based generation
                print_warning("Gemini script too short, using template fallback")
                return self._generate_template_script(metadata, data_path, output_dir)
                
        except Exception as e:
            print_warning(f"Gemini script generation failed: {e}")
            return self._generate_template_script(metadata, data_path, output_dir)
    
    def _create_model_aware_prompt(
        self, metadata: Dict, analysis: str, data_path: Path, output_dir: Path,
        template_string: str, required_fields: List[str]
    ) -> str:
        """Create model-aware preprocessing prompt."""
        
        prompt = f'''You are an expert ML data engineer. Generate a Python preprocessing script for model-aware fine-tuning.

## Target Model Configuration
- Model: {self.config.model.name}
- Architecture: {self.config.model.architecture}
- Chat Template: {self.config.model.chat_template.value}
- Max Length: {self.config.model.max_length}

## Chat Template Format
Template: {template_string}
Required Fields: {required_fields}

## Dataset Information
- Path: {data_path}
- Format: {metadata.get("format", "csv")}
- Columns: {metadata.get("columns", [])}
- Samples: {metadata.get("num_samples", 0)}

## Analysis
{analysis[:2000] if analysis else "No analysis provided"}

## Preprocessing Configuration
- Train Split: {self.config.preprocessing.train_split}
- Validation Split: {self.config.preprocessing.validation_split}
- Chunk Size: {self.config.preprocessing.chunk_size}
- Min Text Length: {self.config.preprocessing.min_text_length}
- Max Text Length: {self.config.preprocessing.max_text_length or "None"}
- Remove Duplicates: {self.config.preprocessing.remove_duplicates}
- Random Seed: {self.config.preprocessing.random_seed}

## Requirements
Generate a COMPLETE Python script that:

1. **Loads the dataset** from: {data_path}
2. **Handles data quality** (missing values, duplicates, encoding issues)
3. **Applies the correct chat template** for {self.config.model.chat_template.value}
4. **Formats data** using the template: {template_string}
5. **Splits data** according to the configuration ratios
6. **Saves output** to /output/processed_train.jsonl and /output/processed_val.jsonl
7. **Provides progress updates** using log_progress() function
8. **Handles errors gracefully** with try/catch blocks

## Template Application Example
For input data with fields {required_fields}, apply the template like this:
```python
formatted_text = "{template_string}".format(
    {", ".join(f'{field}=row["{field}"]' for field in required_fields)}
)
```

## Output Format
Each line in the JSONL files should be:
{{"text": "formatted_template_result"}}

## Available Functions
- log_progress(message): Log progress updates
- log_error(message): Log error messages  
- CONFIG: Dictionary with model and preprocessing configuration

## Libraries Available
Use standard libraries: pandas, numpy, json, os, random, pathlib
The script runs in a Docker container with these pre-installed.

Output ONLY the Python script in ```python``` blocks.
'''
        
        return prompt
    
    def _generate_template_script(self, metadata: Dict, data_path: Path, output_dir: Path) -> str:
        """Generate a basic template-based script as fallback."""
        
        from .templates import ChatTemplateRegistry
        template_config = ChatTemplateRegistry.get_template(self.config.model.chat_template)
        template_string = template_config["template"]
        
        script = f'''    # Load dataset
    log_progress("Loading dataset...")
    data_path = Path("/data") / "{data_path.name}"

    if data_path.suffix.lower() == '.csv':
        df = pd.read_csv(data_path)
    elif data_path.suffix.lower() in ['.json', '.jsonl']:
        df = pd.read_json(data_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {{data_path.suffix}}")

    log_progress(f"Loaded {{len(df)}} samples")

    # Basic data cleaning
    if CONFIG["preprocessing"]["remove_duplicates"]:
        df = df.drop_duplicates()
        log_progress(f"After deduplication: {{len(df)}} samples")

    # Apply chat template
    log_progress("Applying chat template...")
    formatted_data = []

    for _, row in df.iterrows():
        try:
            # Map columns to required fields
            if 'user_query_en' in df.columns and 'bot_response_en' in df.columns:
                query = str(row['user_query_en'])
                response = str(row['bot_response_en'])
            elif 'query' in df.columns and 'response' in df.columns:
                query = str(row['query'])
                response = str(row['response'])
            elif 'text' in df.columns:
                # Split text into query/response or use as-is
                text = str(row['text'])
                query = text
                response = "I understand."
            else:
                # Use first two text columns
                cols = [col for col in df.columns if df[col].dtype == 'object']
                if len(cols) >= 2:
                    query = str(row[cols[0]])
                    response = str(row[cols[1]])
                else:
                    continue
            
            # Apply template - using simple format for safety
            if "{self.config.model.chat_template.value}" == "gemma":
                formatted_text = f"<start_of_turn>user\\n{{query}}<end_of_turn>\\n<start_of_turn>model\\n{{response}}<end_of_turn>"
            elif "{self.config.model.chat_template.value}" == "llama":
                formatted_text = f"<s>[INST] {{query}} [/INST] {{response}} </s>"
            elif "{self.config.model.chat_template.value}" == "chatml":
                formatted_text = f"<|im_start|>user\\n{{query}}<|im_end|>\\n<|im_start|>assistant\\n{{response}}<|im_end|>"
            else:
                # Default format
                formatted_text = f"User: {{query}}\\nAssistant: {{response}}"
            
            formatted_data.append({{"text": formatted_text}})
            
        except Exception as e:
            log_error(f"Failed to format row: {{e}}")
            continue

    log_progress(f"Formatted {{len(formatted_data)}} samples")

    # Split data
    random.seed(CONFIG["preprocessing"]["random_seed"])
    random.shuffle(formatted_data)

    train_size = int(len(formatted_data) * CONFIG["preprocessing"]["train_split"])
    train_data = formatted_data[:train_size]
    val_data = formatted_data[train_size:]

    # Save outputs
    output_dir = Path("/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "processed_train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\\n")

    with open(output_dir / "processed_val.jsonl", "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\\n")

    log_progress(f"Saved {{len(train_data)}} training samples")
    log_progress(f"Saved {{len(val_data)}} validation samples")'''
        
        return script
    
    def _extract_python_code(self, text: str) -> Optional[str]:
        """Extract Python code from Gemini response."""
        import re
        
        patterns = [
            r"```python\s*\n(.*?)```",
            r"```\s*\n(.*?)```"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        # If no code blocks found, try to extract Python-like content
        lines = text.split('\n')
        python_lines = []
        in_code = False
        
        for line in lines:
            if any(line.strip().startswith(kw) for kw in ['import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ']):
                in_code = True
            
            if in_code:
                python_lines.append(line)
        
        if python_lines:
            return '\n'.join(python_lines).strip()
        
        return None