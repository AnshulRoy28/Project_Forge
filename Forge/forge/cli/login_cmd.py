"""
forge login - Update or verify session API credentials.
"""

import typer

from forge.ui.console import console, print_success, print_info
from forge.core.security import prompt_for_credentials, has_credentials, clear_all_credentials


def login_command(
    verify: bool = typer.Option(False, "--verify", "-v", help="Only verify current session credentials"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-entry of credentials"),
):
    """
    Update or verify your session API credentials.
    
    Credentials are stored only in memory for the current session.
    They are automatically cleared when the session ends.
    """
    console.print("\n[bold]🔐 Session Credentials[/]\n")
    
    if verify:
        # Just check what's in session
        has_gemini, has_hf = has_credentials()
        
        if has_gemini:
            print_success("Gemini API key: Available in session ✓")
        else:
            print_info("Gemini API key: Not configured for this session")
        
        if has_hf:
            print_success("HuggingFace token: Available in session ✓")
        else:
            print_info("HuggingFace token: Not configured for this session")
        
        console.print()
        console.print("[dim]Note: Session credentials are cleared when the terminal closes.[/]")
        return
    
    # Clear existing if force
    if force:
        clear_all_credentials()
        console.print("[yellow]Cleared existing session credentials.[/]")
        console.print()
    
    # Prompt for new credentials
    console.print("[bold yellow]🔑 Configure Session Credentials[/]")
    console.print("[dim]These will be stored only in memory for the current session.[/]")
    console.print()
    
    got_gemini, got_hf = prompt_for_credentials(
        console, 
        force_gemini=True,  # Always require Gemini
        force_hf=False      # HuggingFace is optional
    )
    
    console.print()
    
    if got_gemini:
        print_success("Session credentials configured!")
    else:
        print_info("No credentials configured.")
    
    console.print("[dim]Credentials will be automatically cleared when this session ends.[/]")
