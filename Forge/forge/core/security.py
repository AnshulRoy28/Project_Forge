"""
Security Sentinel for Forge.

Reviews Gemini-generated scripts and manages secure credential storage.
"""

import re
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import keyring

# Keyring service name
KEYRING_SERVICE = "forge-cli"
KEYRING_API_KEY = "gemini_api_key"
KEYRING_HF_TOKEN = "huggingface_token"


@dataclass
class SecurityReport:
    """Report from security analysis."""
    
    is_safe: bool
    risk_level: str  # "low", "medium", "high", "critical"
    concerns: list[str]  # Changed from 'issues' to 'concerns'
    script_hash: str


# Dangerous patterns to detect in generated code
DANGEROUS_PATTERNS = [
    # File system operations
    (r"shutil\.rmtree\s*\(", "high", "Recursive directory deletion detected"),
    (r"os\.remove\s*\(", "medium", "File deletion detected"),
    (r"os\.unlink\s*\(", "medium", "File deletion detected"),
    (r"pathlib\.Path.*\.unlink\s*\(", "medium", "File deletion detected"),
    
    # Network operations
    (r"requests\.(get|post|put|delete)\s*\(", "medium", "HTTP request detected"),
    (r"urllib\.request\.urlopen\s*\(", "medium", "URL fetch detected"),
    (r"socket\.", "high", "Raw socket operations detected"),
    
    # Code execution
    (r"exec\s*\(", "critical", "Dynamic code execution (exec) detected"),
    (r"eval\s*\(", "critical", "Dynamic code evaluation (eval) detected"),
    (r"subprocess\.(run|call|Popen)\s*\(", "high", "Subprocess execution detected"),
    (r"os\.system\s*\(", "high", "System command execution detected"),
    (r"os\.popen\s*\(", "high", "System command execution detected"),
    
    # Dangerous imports
    (r"import\s+pickle", "medium", "Pickle import detected (potential code execution)"),
    (r"from\s+pickle\s+import", "medium", "Pickle import detected (potential code execution)"),
    
    # Environment manipulation
    (r"os\.environ\[.*\]\s*=", "low", "Environment variable modification detected"),
    
    # Credential access
    (r"keyring\.(get_password|set_password|delete_password)", "medium", "Keyring access detected"),
]


def analyze_script(script_content: str) -> SecurityReport:
    """
    Analyze a script for security issues.
    
    Returns a SecurityReport with findings.
    """
    concerns: list[str] = []  # Changed from 'issues' to 'concerns'
    max_risk = "low"
    risk_priority = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    
    for pattern, risk, message in DANGEROUS_PATTERNS:
        if re.search(pattern, script_content, re.IGNORECASE):
            concerns.append(f"[{risk.upper()}] {message}")
            if risk_priority.get(risk, 0) > risk_priority.get(max_risk, 0):
                max_risk = risk
    
    # Calculate hash for tracking
    script_hash = hashlib.sha256(script_content.encode()).hexdigest()[:16]
    
    # Determine if safe (allow low and medium by default)
    is_safe = max_risk in ("low", "medium")
    
    return SecurityReport(
        is_safe=is_safe,
        risk_level=max_risk,
        concerns=concerns,  # Changed from 'issues' to 'concerns'
        script_hash=script_hash,
    )


def review_script_interactive(script_content: str, console) -> bool:
    """
    Interactively review a script with the user.
    
    Returns True if user approves execution.
    """
    from rich.syntax import Syntax
    from rich.panel import Panel
    
    report = analyze_script(script_content)
    
    # Show the script
    syntax = Syntax(script_content, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="[bold]Generated Script[/]", border_style="cyan"))
    
    # Show security report
    if report.concerns:  # Changed from 'issues' to 'concerns'
        console.print("\n[bold yellow]⚠️  Security Analysis:[/]")
        for concern in report.concerns:  # Changed from 'issues' to 'concerns'
            console.print(f"  • {concern}")
        console.print()
    
    risk_colors = {
        "low": "green",
        "medium": "yellow",
        "high": "red",
        "critical": "bold red",
    }
    risk_color = risk_colors.get(report.risk_level, "white")
    console.print(f"Risk Level: [{risk_color}]{report.risk_level.upper()}[/]")
    console.print(f"Script Hash: [dim]{report.script_hash}[/]\n")
    
    if report.risk_level == "critical":
        console.print("[bold red]⛔ CRITICAL RISK: This script contains potentially dangerous operations.[/]")
        console.print("[bold red]Automatic execution is blocked. Manual review required.[/]\n")
        return False
    
    if report.is_safe:
        console.print("[green]✓ Script passed security checks.[/]")
        return True
    
    # Ask for confirmation on high-risk scripts
    from rich.prompt import Confirm
    return Confirm.ask("[yellow]Do you want to execute this script?[/]", default=False)


# === Credential Management ===

def store_api_key(api_key: str) -> bool:
    """
    Store the Gemini API key securely in the system keyring.
    
    Returns True on success.
    """
    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_API_KEY, api_key)
        return True
    except Exception:
        return False


def get_api_key() -> Optional[str]:
    """
    Retrieve the Gemini API key from the system keyring.
    
    Returns None if not found.
    """
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_API_KEY)
    except Exception:
        return None


def delete_api_key() -> bool:
    """
    Delete the stored API key.
    
    Returns True on success.
    """
    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_API_KEY)
        return True
    except Exception:
        return False


def store_hf_token(token: str) -> bool:
    """
    Store the HuggingFace token securely in the system keyring.
    
    Returns True on success.
    """
    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_HF_TOKEN, token)
        return True
    except Exception:
        return False


def get_hf_token() -> Optional[str]:
    """
    Retrieve the HuggingFace token from the system keyring.
    
    Returns None if not found.
    """
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_HF_TOKEN)
    except Exception:
        return None


def delete_hf_token() -> bool:
    """
    Delete the stored HuggingFace token.
    
    Returns True on success.
    """
    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_HF_TOKEN)
        return True
    except Exception:
        return False


def validate_api_key_format(api_key: str) -> bool:
    """
    Validate that the API key has the expected format.
    
    Note: This doesn't verify the key is valid with the API.
    """
    if not api_key:
        return False
    
    # Gemini API keys are typically 39 characters, alphanumeric with underscores
    if len(api_key) < 20:
        return False
    
    # Should contain at least some alphanumeric characters
    if not re.search(r"[a-zA-Z0-9]", api_key):
        return False
    
    return True


def validate_hf_token_format(token: str) -> bool:
    """
    Validate that the HuggingFace token has the expected format.
    
    HuggingFace tokens typically start with 'hf_' and are 37+ characters long.
    """
    if not token:
        return False
    
    # HuggingFace tokens typically start with 'hf_' and are 37+ characters
    if token.startswith('hf_') and len(token) >= 37:
        return True
    
    # Also accept older format tokens (without hf_ prefix)
    if len(token) >= 20 and re.match(r'^[a-zA-Z0-9_-]+$', token):
        return True
    
    return False
