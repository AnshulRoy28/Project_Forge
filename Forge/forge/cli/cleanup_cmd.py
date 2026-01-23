"""
forge cleanup - Clear session credentials and temporary files.

Provides secure cleanup of session-based credentials.
"""

import typer
from rich.prompt import Confirm

from forge.ui.console import console, print_success, print_info
from forge.core.security import clear_all_credentials, has_credentials


def cleanup_command(
    force: bool = typer.Option(False, "--force", "-f", help="Force cleanup without confirmation"),
):
    """
    Clear session credentials and temporary files.
    
    This command:
    - Clears Gemini API key from session memory
    - Clears HuggingFace token from session memory  
    - Removes temporary session files
    - Clears environment variables
    
    Security: This ensures no credentials remain in memory after use.
    """
    has_gemini, has_hf = has_credentials()
    
    if not has_gemini and not has_hf:
        print_info("No session credentials found to clean up.")
        return
    
    console.print("[bold yellow]🧹 Session Cleanup[/]")
    console.print()
    
    # Show what will be cleared
    if has_gemini:
        console.print("  • Gemini API key (session memory)")
    if has_hf:
        console.print("  • HuggingFace token (session memory)")
    
    console.print("  • Temporary session files")
    console.print("  • Environment variables")
    console.print()
    
    # Confirm unless forced
    if not force:
        if not Confirm.ask("[yellow]Clear all session credentials?[/]", default=True):
            print_info("Cleanup cancelled.")
            return
    
    # Perform cleanup
    clear_all_credentials()
    
    print_success("Session credentials cleared!")
    console.print("[dim]All sensitive data removed from memory.[/]")
    console.print()
    console.print("[dim]Next time you run forge commands, you'll be prompted for credentials again.[/]")