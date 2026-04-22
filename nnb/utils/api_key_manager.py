"""Secure API key management using system keyring."""

import os
from typing import Optional
import keyring
from keyring.errors import KeyringError

from rich.console import Console
from rich.prompt import Prompt, Confirm

console = Console()

SERVICE_NAME = "nnb-cli"
KEY_NAME = "gemini_api_key"


class APIKeyManager:
    """Manages Gemini API key storage in system keyring."""
    
    @staticmethod
    def get_api_key() -> Optional[str]:
        """Get API key from keyring or environment variable."""
        # First check environment variable (for CI/CD, Docker, etc.)
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            return env_key
        
        # Then check system keyring
        try:
            key = keyring.get_password(SERVICE_NAME, KEY_NAME)
            return key
        except KeyringError as e:
            console.print(f"[yellow]⚠️  Warning: Could not access keyring: {e}[/yellow]")
            return None
    
    @staticmethod
    def set_api_key(api_key: str) -> bool:
        """Store API key in system keyring."""
        try:
            keyring.set_password(SERVICE_NAME, KEY_NAME, api_key)
            return True
        except KeyringError as e:
            console.print(f"[red]❌ Failed to store API key: {e}[/red]")
            return False
    
    @staticmethod
    def delete_api_key() -> bool:
        """Delete API key from system keyring."""
        try:
            keyring.delete_password(SERVICE_NAME, KEY_NAME)
            return True
        except keyring.errors.PasswordDeleteError:
            # Key doesn't exist, that's fine
            return True
        except KeyringError as e:
            console.print(f"[red]❌ Failed to delete API key: {e}[/red]")
            return False
    
    @staticmethod
    def has_api_key() -> bool:
        """Check if API key is configured."""
        return APIKeyManager.get_api_key() is not None
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Basic validation of API key format."""
        # Gemini API keys typically start with "AIza" and are 39 characters
        if not api_key:
            return False
        
        api_key = api_key.strip()
        
        if len(api_key) < 20:
            console.print("[yellow]⚠️  API key seems too short[/yellow]")
            return False
        
        if not api_key.startswith("AIza"):
            console.print("[yellow]⚠️  Gemini API keys typically start with 'AIza'[/yellow]")
            # Don't fail, just warn
        
        return True


def setup_api_key_interactive() -> bool:
    """Interactive setup for API key."""
    console.print("\n[bold]🔑 Gemini API Key Setup[/bold]\n")
    
    # Check if key already exists
    if APIKeyManager.has_api_key():
        console.print("[green]✓ API key is already configured[/green]")
        
        if Confirm.ask("Do you want to update it?", default=False):
            pass  # Continue to update
        else:
            return True
    
    # Instructions
    console.print("To use this tool, you need a Gemini API key.")
    console.print("Get one for free at: [cyan]https://makersuite.google.com/app/apikey[/cyan]\n")
    
    console.print("[dim]Your API key will be stored securely in your system keyring\n(like passwords in your browser or password manager).[/dim]\n")
    
    # Get API key
    api_key = Prompt.ask(
        "Enter your Gemini API key",
        password=True  # Hide input
    )
    
    # Validate
    if not APIKeyManager.validate_api_key(api_key):
        console.print("[red]❌ Invalid API key format[/red]")
        return False
    
    # Test the key
    console.print("\n🔍 Testing API key...")
    
    try:
        from nnb.gemini_brain.client import GeminiClient
        
        client = GeminiClient(api_key=api_key)
        # Simple test
        response = client.generate("Say 'OK' if you can read this.", temperature=0.1)
        
        if response:
            console.print("[green]✓ API key is valid![/green]")
        else:
            console.print("[red]❌ API key test failed[/red]")
            return False
            
    except Exception as e:
        console.print(f"[red]❌ API key test failed: {e}[/red]")
        console.print("\n[yellow]The key might be invalid or there's a network issue.[/yellow]")
        
        if not Confirm.ask("Store it anyway?", default=False):
            return False
    
    # Store the key
    console.print("\n💾 Storing API key securely...")
    
    if APIKeyManager.set_api_key(api_key):
        console.print("[green]✓ API key stored successfully![/green]")
        console.print("\n[dim]Your key is now stored in your system keyring.")
        console.print("You can delete it anytime with: nnb config delete-key[/dim]\n")
        return True
    else:
        console.print("[red]❌ Failed to store API key[/red]")
        return False


def delete_api_key_interactive() -> bool:
    """Interactive deletion of API key."""
    console.print("\n[bold]🗑️  Delete Gemini API Key[/bold]\n")
    
    if not APIKeyManager.has_api_key():
        console.print("[yellow]⚠️  No API key is currently stored[/yellow]")
        return True
    
    console.print("This will delete your Gemini API key from the system keyring.")
    console.print("[dim]You can always add it back later with: nnb config setup[/dim]\n")
    
    if Confirm.ask("Are you sure you want to delete your API key?", default=False):
        if APIKeyManager.delete_api_key():
            console.print("[green]✓ API key deleted successfully[/green]")
            return True
        else:
            console.print("[red]❌ Failed to delete API key[/red]")
            return False
    else:
        console.print("Cancelled")
        return False


def show_api_key_status() -> None:
    """Show API key configuration status."""
    console.print("\n[bold]🔑 API Key Status[/bold]\n")
    
    # Check environment variable
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        console.print("[green]✓ API key found in environment variable[/green]")
        console.print(f"  Key: {env_key[:10]}...{env_key[-4:]}")
        console.print("  [dim](Environment variables take precedence over keyring)[/dim]")
        return
    
    # Check keyring
    if APIKeyManager.has_api_key():
        key = APIKeyManager.get_api_key()
        console.print("[green]✓ API key found in system keyring[/green]")
        if key:
            console.print(f"  Key: {key[:10]}...{key[-4:]}")
    else:
        console.print("[yellow]⚠️  No API key configured[/yellow]")
        console.print("\n💡 Set up your API key with: [cyan]nnb config setup[/cyan]")
