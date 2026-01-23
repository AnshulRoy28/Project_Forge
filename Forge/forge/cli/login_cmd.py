"""
forge login - Update or verify API credentials.
"""

import os
from pathlib import Path

import typer
from rich.prompt import Prompt, Confirm

from forge.ui.console import console, print_success, print_error, print_info, print_warning
from forge.core.config import ensure_forge_dir


def login_command(
    verify: bool = typer.Option(False, "--verify", "-v", help="Only verify current credentials"),
):
    """
    Update or verify your Gemini/HuggingFace API credentials.
    
    Credentials are stored in environment variables and .forge/credentials.
    """
    console.print("\n[bold]🔐 Forge Credentials[/]\n")
    
    forge_dir = ensure_forge_dir()
    creds_file = forge_dir / "credentials"
    
    # Load existing credentials
    existing_gemini = os.getenv("GEMINI_API_KEY", "")
    existing_hf = os.getenv("HF_TOKEN", "")
    
    if creds_file.exists():
        with open(creds_file, 'r') as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    existing_gemini = existing_gemini or line.strip().split("=", 1)[1]
                elif line.startswith("HF_TOKEN="):
                    existing_hf = existing_hf or line.strip().split("=", 1)[1]
    
    if verify:
        # Just verify
        _verify_credentials(existing_gemini, existing_hf)
        return
    
    # Update credentials
    console.print("[dim]Press Enter to keep existing value[/]\n")
    
    # Gemini API Key
    if existing_gemini:
        masked = existing_gemini[:8] + "..." + existing_gemini[-4:]
        console.print(f"Current Gemini API Key: [cyan]{masked}[/]")
    
    new_gemini = Prompt.ask(
        "Gemini API Key",
        default="",
        password=True,
    )
    gemini_key = new_gemini if new_gemini else existing_gemini
    
    # HuggingFace Token (optional)
    if existing_hf:
        masked = existing_hf[:8] + "..." + existing_hf[-4:]
        console.print(f"Current HuggingFace Token: [cyan]{masked}[/]")
    
    new_hf = Prompt.ask(
        "HuggingFace Token (for gated models)",
        default="",
        password=True,
    )
    hf_token = new_hf if new_hf else existing_hf
    
    # Save credentials
    with open(creds_file, 'w') as f:
        if gemini_key:
            f.write(f"GEMINI_API_KEY={gemini_key}\n")
        if hf_token:
            f.write(f"HF_TOKEN={hf_token}\n")
    
    console.print()
    print_success(f"Credentials saved to {creds_file}")
    
    # Verify
    _verify_credentials(gemini_key, hf_token)


def _verify_credentials(gemini_key: str, hf_token: str):
    """Verify that credentials work."""
    console.print("\n[bold]Verifying credentials...[/]\n")
    
    # Verify Gemini
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content("Say OK")
            print_success("Gemini API: Connected ✓")
        except Exception as e:
            print_error(f"Gemini API: Failed - {e}")
    else:
        print_warning("Gemini API: Not configured")
    
    # Verify HuggingFace
    if hf_token:
        try:
            import requests
            headers = {"Authorization": f"Bearer {hf_token}"}
            response = requests.get(
                "https://huggingface.co/api/whoami",
                headers=headers,
                timeout=10,
            )
            if response.ok:
                username = response.json().get("name", "Unknown")
                print_success(f"HuggingFace: Logged in as {username} ✓")
            else:
                print_error("HuggingFace: Invalid token")
        except Exception as e:
            print_warning(f"HuggingFace: Could not verify - {e}")
    else:
        print_info("HuggingFace: Not configured (optional)")
    
    console.print()
