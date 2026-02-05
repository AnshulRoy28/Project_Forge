"""Script generation, validation, and coordination."""

import ast
import re
import uuid
from typing import List, Optional, Tuple

from ..models.data import (
    DataAnalysis,
    ProcessingContext,
    ProcessingScript,
    ValidationResult,
)
from ..models.enums import ValidationStatus


class ScriptCoordinator:
    """Manages script generation, validation, and execution sequencing."""
    
    # Patterns that indicate potentially dangerous operations
    DANGEROUS_PATTERNS = [
        r'\bos\.system\b',
        r'\bsubprocess\.',
        r'\beval\b\(',
        r'\bexec\b\(',
        r'\b__import__\b',
        r'\bopen\s*\([^)]*["\']\/(?!data|output)',  # Opening files outside data/output
        r'\brequests\.',  # Network requests
        r'\burllib\.',
        r'\bsocket\.',
        r'\bshutil\.rmtree\b',
        r'\bos\.remove\b',
        r'\bos\.unlink\b',
    ]
    
    # Safe packages that are commonly used for data processing
    SAFE_PACKAGES = {
        'pandas', 'numpy', 'scipy', 'sklearn', 'scikit-learn',
        'matplotlib', 'seaborn', 'plotly',
        'json', 'csv', 're', 'datetime', 'collections',
        'itertools', 'functools', 'math', 'statistics',
        'openpyxl', 'xlrd', 'xlwt',
    }
    
    def __init__(self):
        """Initialize the script coordinator."""
        self._executed_scripts: List[ProcessingScript] = []
    
    def validate_script_syntax(self, script: ProcessingScript) -> ValidationResult:
        """
        Validate the syntax and security of a processing script.
        
        Args:
            script: The script to validate.
            
        Returns:
            ValidationResult with syntax errors and security warnings.
        """
        syntax_errors: List[str] = []
        security_warnings: List[str] = []
        suggested_fixes: List[str] = []
        
        # Check Python syntax
        try:
            ast.parse(script.content)
        except SyntaxError as e:
            syntax_errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            suggested_fixes.append(f"Fix the syntax error near line {e.lineno}")
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            matches = re.findall(pattern, script.content)
            if matches:
                security_warnings.append(
                    f"Potentially dangerous pattern detected: {matches[0]}"
                )
        
        # Check for unknown imports
        try:
            tree = ast.parse(script.content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        pkg = alias.name.split('.')[0]
                        if pkg not in self.SAFE_PACKAGES:
                            security_warnings.append(
                                f"Unknown package import: {pkg}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        pkg = node.module.split('.')[0]
                        if pkg not in self.SAFE_PACKAGES:
                            security_warnings.append(
                                f"Unknown package import: {pkg}"
                            )
        except SyntaxError:
            pass  # Already caught above
        
        is_valid = len(syntax_errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            syntax_errors=syntax_errors,
            security_warnings=security_warnings,
            suggested_fixes=suggested_fixes,
        )
    
    def create_script(
        self,
        content: str,
        description: str,
        required_packages: Optional[List[str]] = None,
        input_files: Optional[List[str]] = None,
        output_files: Optional[List[str]] = None,
    ) -> ProcessingScript:
        """
        Create a new processing script.
        
        Args:
            content: The Python script content.
            description: Human-readable description of what the script does.
            required_packages: List of required Python packages.
            input_files: List of input file paths.
            output_files: List of output file paths.
            
        Returns:
            A new ProcessingScript instance.
        """
        script_id = str(uuid.uuid4())[:8]
        
        script = ProcessingScript(
            script_id=script_id,
            content=content,
            description=description,
            required_packages=required_packages or [],
            input_files=input_files or [],
            output_files=output_files or [],
        )
        
        # Validate the script
        validation = self.validate_script_syntax(script)
        if validation.is_valid:
            if validation.security_warnings:
                script.validation_status = ValidationStatus.WARNING
                script.security_level = "caution"
            else:
                script.validation_status = ValidationStatus.VALID
                script.security_level = "safe"
        else:
            script.validation_status = ValidationStatus.INVALID
            script.security_level = "restricted"
        
        return script
    
    def generate_script_sequence(
        self,
        analysis: DataAnalysis,
        context: Optional[ProcessingContext] = None
    ) -> List[ProcessingScript]:
        """
        Generate a sequence of scripts based on data analysis.
        
        This is a placeholder that will be replaced with Gemini AI generation.
        
        Args:
            analysis: The data analysis results.
            context: Optional processing context from previous operations.
            
        Returns:
            List of processing scripts to execute.
        """
        scripts: List[ProcessingScript] = []
        
        # Generate scripts based on suggested operations
        for i, operation in enumerate(analysis.suggested_operations):
            # This is a placeholder - actual implementation will use Gemini
            script = self.create_script(
                content=f"# {operation}\n# Script will be generated by Gemini AI",
                description=operation,
                required_packages=["pandas"],
            )
            scripts.append(script)
        
        return scripts
    
    def record_execution(self, script: ProcessingScript) -> None:
        """Record a script as executed."""
        self._executed_scripts.append(script)
    
    def get_executed_scripts(self) -> List[ProcessingScript]:
        """Get list of all executed scripts."""
        return self._executed_scripts.copy()
