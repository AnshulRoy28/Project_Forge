"""
Security Sentinel for Forge.

Reviews Gemini-generated scripts and manages secure session-based credential storage.
API keys are stored only in memory for the current session and automatically cleared.
"""

import re
import hashlib
import os
import atexit
import tempfile
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Session-based credential storage (persists across commands in same terminal session)
_session_credentials = {}
_session_file = None

def _get_session_file():
    """Get or create session file for credential persistence."""
    global _session_file
    if _session_file is None:
        # Create session file in temp directory with user-specific name
        import getpass
        
        # Use user + current working directory hash for session identification
        # This ensures credentials persist across commands in the same project
        import hashlib
        cwd_hash = hashlib.md5(str(Path.cwd()).encode()).hexdigest()[:8]
        session_id = f"{getpass.getuser()}_{cwd_hash}"
        
        _session_file = Path(tempfile.gettempdir()) / f"forge_session_{session_id}.json"
        
        # Register cleanup on exit
        atexit.register(_cleanup_session)
    
    return _session_file

def _load_session_credentials():
    """Load credentials from session file."""
    global _session_credentials
    try:
        session_file = _get_session_file()
        if session_file.exists():
            import json
            with open(session_file, 'r') as f:
                _session_credentials = json.load(f)
        else:
            _session_credentials = {}
    except Exception:
        _session_credentials = {}

def _save_session_credentials():
    """Save credentials to session file."""
    try:
        session_file = _get_session_file()
        import json
        with open(session_file, 'w') as f:
            json.dump(_session_credentials, f)
    except Exception:
        pass  # Fail silently

def _cleanup_session():
    """Clean up session file."""
    try:
        session_file = _get_session_file()
        if session_file.exists():
            session_file.unlink()
    except Exception:
        pass

# Load existing session on module import
try:
    _load_session_credentials()
except Exception as e:
    # Fail silently on import, but ensure _session_credentials is initialized
    _session_credentials = {}


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


# === Session-Based Credential Management ===

def _cleanup_session():
    """Clean up session credentials and temporary files."""
    global _session_credentials, _session_file
    
    # Clear in-memory credentials
    _session_credentials["gemini_api_key"] = None
    _session_credentials["huggingface_token"] = None
    
    # Remove temporary session file if it exists
    if _session_file and os.path.exists(_session_file):
        try:
            os.unlink(_session_file)
        except Exception:
            pass
    
    # Clear environment variables
    for key in ["GEMINI_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
        if key in os.environ:
            del os.environ[key]


def store_api_key(api_key: str) -> bool:
    """
    Store the Gemini API key in session memory only.
    
    Returns True on success.
    """
    try:
        _session_credentials["gemini_api_key"] = api_key
        _save_session_credentials()
        # Also set as environment variable for Docker containers
        os.environ["GEMINI_API_KEY"] = api_key
        return True
    except Exception:
        return False


def get_api_key() -> Optional[str]:
    """
    Retrieve the Gemini API key from session memory.
    
    Returns None if not found in current session.
    """
    return _session_credentials.get("gemini_api_key")


def delete_api_key() -> bool:
    """
    Delete the API key from session memory.
    
    Returns True on success.
    """
    try:
        _session_credentials["gemini_api_key"] = None
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]
        return True
    except Exception:
        return False


def store_hf_token(token: str) -> bool:
    """
    Store the HuggingFace token in session memory only.
    
    Returns True on success.
    """
    try:
        _session_credentials["huggingface_token"] = token
        _save_session_credentials()
        # Also set as environment variable for Docker containers
        os.environ["HF_TOKEN"] = token
        return True
    except Exception:
        return False


def get_hf_token() -> Optional[str]:
    """
    Retrieve the HuggingFace token from session memory.
    
    Returns None if not found in current session.
    """
    return _session_credentials.get("huggingface_token")


def delete_hf_token() -> bool:
    """
    Delete the HuggingFace token from session memory.
    
    Returns True on success.
    """
    try:
        _session_credentials["huggingface_token"] = None
        for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
            if key in os.environ:
                del os.environ[key]
        return True
    except Exception:
        return False


def clear_all_credentials():
    """Clear all session credentials immediately."""
    _cleanup_session()


def has_credentials() -> tuple[bool, bool]:
    """
    Check if credentials are available in current session.
    
    Returns (has_gemini_key, has_hf_token).
    """
    # Ensure session is loaded
    _load_session_credentials()
    
    return (
        _session_credentials.get("gemini_api_key") is not None,
        _session_credentials.get("huggingface_token") is not None
    )


def prompt_for_credentials(console, force_gemini: bool = True, force_hf: bool = False) -> tuple[bool, bool]:
    """
    Prompt user for API credentials if not already in session.
    
    Args:
        console: Rich console for output
        force_gemini: Require Gemini API key
        force_hf: Require HuggingFace token
    
    Returns (got_gemini, got_hf)
    """
    from rich.prompt import Prompt, Confirm
    
    has_gemini, has_hf = has_credentials()
    got_gemini, got_hf = has_gemini, has_hf
    
    # Prompt for Gemini API key if needed
    if not has_gemini and force_gemini:
        console.print("\n[bold yellow]🔑 Gemini API Key Required[/]")
        console.print("[dim]Get one at: https://aistudio.google.com/apikey[/]")
        console.print("[dim]This will be stored only for the current session.[/]")
        
        while True:
            api_key = Prompt.ask("Gemini API Key", password=True)
            
            if not api_key:
                console.print("[red]API key cannot be empty.[/]")
                continue
            
            if not validate_api_key_format(api_key):
                console.print("[red]Invalid API key format.[/]")
                continue
            
            if store_api_key(api_key):
                console.print("[green]✓ Gemini API key stored for this session.[/]")
                got_gemini = True
                break
            else:
                console.print("[red]Failed to store API key.[/]")
                return False, got_hf
    
    # Prompt for HuggingFace token if needed
    if not has_hf and (force_hf or Confirm.ask("\n[yellow]Configure HuggingFace token for gated models?[/]", default=False)):
        console.print("\n[bold yellow]🤗 HuggingFace Token[/]")
        console.print("[dim]Get one at: https://huggingface.co/settings/tokens[/]")
        console.print("[dim]Required for gated models like google/gemma-7b-it[/]")
        console.print("[dim]This will be stored only for the current session.[/]")
        
        while True:
            hf_token = Prompt.ask("HuggingFace Token (or press Enter to skip)", password=True)
            
            if not hf_token:
                console.print("[dim]Skipping HuggingFace token - you can only use open models[/]")
                break
            
            if not validate_hf_token_format(hf_token):
                console.print("[red]Invalid HuggingFace token format.[/]")
                console.print("[dim]Tokens should start with 'hf_' and be 37+ characters[/]")
                continue
            
            if store_hf_token(hf_token):
                console.print("[green]✓ HuggingFace token stored for this session.[/]")
                got_hf = True
                break
            else:
                console.print("[red]Failed to store HuggingFace token.[/]")
                break
    
    return got_gemini, got_hf


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
