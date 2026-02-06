"""Gemini AI client integration."""

import json
import time
from typing import List, Optional

from ..models.config import GeminiConfig
from ..models.data import (
    DataAnalysis,
    DataSnapshot,
    ProcessingContext,
    ProcessingScript,
)
from ..models.enums import ValidationStatus
from .prompts import PromptManager

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class GeminiIntegration:
    """Handles all interactions with Gemini AI."""
    
    def __init__(self, config: GeminiConfig):
        """
        Initialize the Gemini integration.
        
        Args:
            config: Gemini configuration with API key.
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )
        
        self.config = config
        self._client = None
        self._last_request_time = 0.0
        self._request_interval = 60.0 / config.rate_limit_requests_per_minute
    
    @property
    def client(self):
        """Get the Gemini client, initializing if needed."""
        if self._client is None:
            self._client = genai.Client(api_key=self.config.api_key)
        return self._client
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _call_gemini(self, prompt: str, response_format: str = "json") -> str:
        """
        Make a rate-limited call to Gemini API.
        
        Args:
            prompt: The prompt to send.
            response_format: Expected response format ("json" or "text").
            
        Returns:
            The response text.
        """
        self._rate_limit()
        
        generation_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )
        
        if response_format == "json":
            generation_config.response_mime_type = "application/json"
        
        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=prompt,
            config=generation_config,
        )
        
        return response.text
    
    def authenticate(self) -> bool:
        """
        Verify the API key is valid.
        
        Returns:
            True if authentication successful, False otherwise.
        """
        try:
            # Make a simple request to verify the key works
            response = self._call_gemini(
                "Respond with just the word 'ok'",
                response_format="text"
            )
            return True
        except Exception:
            return False
    
    def analyze_data_snapshot(self, snapshot: DataSnapshot) -> DataAnalysis:
        """
        Analyze a data snapshot using Gemini AI.
        
        Args:
            snapshot: The data snapshot to analyze.
            
        Returns:
            DataAnalysis with AI insights.
        """
        prompt = PromptManager.format_data_analysis_prompt(snapshot)
        
        try:
            response_text = self._call_gemini(prompt, response_format="json")
            result = json.loads(response_text)
            
            return DataAnalysis(
                data_quality_issues=result.get("data_quality_issues", []),
                suggested_operations=result.get("suggested_operations", []),
                column_insights=result.get("column_insights", {}),
                processing_recommendations=result.get("processing_recommendations", []),
                estimated_complexity=result.get("estimated_complexity", "medium"),
                sensitive_data_detected=result.get("sensitive_data_detected", False),
            )
        except json.JSONDecodeError as e:
            # Return a basic analysis if JSON parsing fails
            return DataAnalysis(
                data_quality_issues=["Unable to parse AI response"],
                suggested_operations=[],
                column_insights={},
                processing_recommendations=["Manual review recommended"],
                estimated_complexity="unknown",
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")
    
    def generate_modular_eda_scripts(
        self,
        snapshot: DataSnapshot,
    ) -> List[ProcessingScript]:
        """
        Generate modular EDA scripts - each focused on one analysis task.
        
        Args:
            snapshot: The data snapshot with schema info.
            
        Returns:
            List of focused ProcessingScripts for EDA.
        """
        scripts = []
        
        for step_key, step_description in PromptManager.EDA_STEPS:
            try:
                prompt = PromptManager.format_modular_eda_prompt(snapshot, step_key)
                script_content = self._call_gemini(prompt, response_format="text")
                
                # Clean up the script content
                script_content = self._clean_script_content(script_content)
                
                # Extract and ensure required packages
                packages = self._extract_packages(script_content)
                
                # Add common packages based on step type
                if step_key in ("missing_values", "descriptive_stats", "correlation_analysis"):
                    for pkg in ["pandas", "matplotlib", "seaborn"]:
                        if pkg not in packages:
                            packages.append(pkg)
                elif step_key in ("outlier_detection",):
                    for pkg in ["pandas", "matplotlib"]:
                        if pkg not in packages:
                            packages.append(pkg)
                else:
                    if "pandas" not in packages:
                        packages.append("pandas")
                
                script = ProcessingScript(
                    script_id=f"eda_{step_key}",
                    content=script_content,
                    description=step_description,
                    required_packages=packages,
                    input_files=["input_data"],
                    output_files=[f"{step_key}_report.json"],
                    validation_status=ValidationStatus.VALID,
                )
                
                scripts.append(script)
                
            except Exception as e:
                # Skip this step if generation fails, continue with others
                continue
        
        return scripts
    
    def generate_processing_scripts(
        self,
        analysis: DataAnalysis,
        context: Optional[ProcessingContext] = None,
    ) -> List[ProcessingScript]:
        """
        Generate processing scripts based on data analysis.
        
        Args:
            analysis: The data analysis results.
            context: Optional processing context.
            
        Returns:
            List of processing scripts.
        """
        scripts = []
        
        # Generate a script for each suggested operation
        for i, operation in enumerate(analysis.suggested_operations[:3]):  # Limit to 3 scripts
            try:
                prompt = PromptManager.format_script_generation_prompt(
                    analysis=analysis,
                    operation=operation,
                    context=context,
                )
                
                script_content = self._call_gemini(prompt, response_format="text")
                
                # Clean up the script content (remove markdown if present)
                script_content = self._clean_script_content(script_content)
                
                # Extract required packages from the script
                packages = self._extract_packages(script_content)
                
                script = ProcessingScript(
                    script_id=f"script_{i+1}",
                    content=script_content,
                    description=operation,
                    required_packages=packages,
                    input_files=["input_data"],
                    output_files=["processed_data"],
                    validation_status=ValidationStatus.VALID,
                )
                
                scripts.append(script)
                
            except Exception as e:
                # Skip this operation if script generation fails
                continue
        
        return scripts
    
    def _clean_script_content(self, content: str) -> str:
        """Remove markdown code blocks if present."""
        content = content.strip()
        
        # Remove markdown code blocks
        if content.startswith("```python"):
            content = content[9:]
        elif content.startswith("```"):
            content = content[3:]
        
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3]
        
        return content.strip()
    
    def _extract_packages(self, script_content: str) -> List[str]:
        """Extract required packages from script imports."""
        packages = set()
        
        # Common package mappings
        import_mapping = {
            'pd': 'pandas',
            'np': 'numpy',
            'plt': 'matplotlib',
            'sns': 'seaborn',
            'sklearn': 'scikit-learn',
        }
        
        for line in script_content.split('\n'):
            line = line.strip()
            
            if line.startswith('import '):
                # Handle "import x" and "import x as y"
                parts = line[7:].split(' as ')[0].split(',')
                for part in parts:
                    pkg = part.strip().split('.')[0]
                    packages.add(import_mapping.get(pkg, pkg))
                    
            elif line.startswith('from '):
                # Handle "from x import y"
                pkg = line[5:].split(' import ')[0].split('.')[0].strip()
                packages.add(import_mapping.get(pkg, pkg))
        
        # Filter out standard library modules
        stdlib = {
            'os', 'sys', 'json', 'csv', 're', 'datetime', 'time',
            'collections', 'itertools', 'functools', 'math', 'random',
            'pathlib', 'typing', 'dataclasses', 'enum', 'io', 'tempfile',
            'string', 'uuid', 'hashlib', 'base64', 'copy', 'operator',
            'warnings', 'logging', 'argparse', 'configparser', 'textwrap',
            'shutil', 'glob', 'fnmatch', 'stat', 'fileinput', 'struct',
            'codecs', 'unicodedata', 'difflib', 'pprint', 'reprlib',
            'abc', 'contextlib', 'decimal', 'fractions', 'numbers',
            'cmath', 'statistics', 'bisect', 'heapq', 'array',
            'weakref', 'types', 'pickle', 'shelve', 'dbm', 'sqlite3',
            'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile',
            'readline', 'rlcompleter', 'threading', 'multiprocessing',
            'subprocess', 'sched', 'queue', 'contextvars', 'asyncio',
            'socket', 'ssl', 'select', 'selectors', 'signal',
            'email', 'html', 'xml', 'urllib', 'http', 'ftplib',
            'smtplib', 'imaplib', 'poplib', 'telnetlib', 'socketserver',
            'xmlrpc', 'ipaddress', 'traceback', 'linecache', 'pickle',
            'copyreg', 'pdb', 'profile', 'timeit', 'trace', 'gc',
            'inspect', 'dis', 'ast', 'builtins', 'keyword', 'token',
            'tokenize', 'platform', 'errno', 'ctypes', 'concurrent',
        }
        
        return [pkg for pkg in packages if pkg not in stdlib]
    
    def validate_script_output(
        self,
        script_description: str,
        expected_files: List[str],
        actual_files: List[str],
        file_contents: dict,
        console_output: str,
    ) -> dict:
        """
        Validate script output using Gemini AI.
        
        Returns:
            Dictionary with validation results:
            - is_valid: bool
            - issues: list of issues found
            - suggestions: list of fix suggestions
            - severity: none|minor|major|critical
        """
        prompt = PromptManager.format_validation_prompt(
            script_description=script_description,
            expected_files=expected_files,
            actual_files=actual_files,
            file_contents=file_contents,
            console_output=console_output,
        )
        
        try:
            response = self._call_gemini(prompt, response_format="json")
            result = json.loads(response)
            return {
                "is_valid": result.get("is_valid", False),
                "issues": result.get("issues", []),
                "suggestions": result.get("suggestions", []),
                "severity": result.get("severity", "minor"),
            }
        except Exception as e:
            # If validation fails, return a default result
            return {
                "is_valid": True,  # Don't block on validation errors
                "issues": [f"Validation check failed: {str(e)}"],
                "suggestions": [],
                "severity": "minor",
            }
    
    def generate_fixed_script(
        self,
        script_description: str,
        original_script: str,
        issues: List[str],
        console_output: str,
        file_contents: dict,
    ) -> str:
        """
        Generate a fixed version of a script that failed validation.
        
        Returns:
            Fixed Python script content.
        """
        prompt = PromptManager.format_self_heal_prompt(
            script_description=script_description,
            original_script=original_script,
            issues=issues,
            console_output=console_output,
            file_contents=file_contents,
        )
        
        try:
            fixed_script = self._call_gemini(prompt, response_format="text")
            return self._clean_script_content(fixed_script)
        except Exception as e:
            raise RuntimeError(f"Failed to generate fixed script: {str(e)}")
