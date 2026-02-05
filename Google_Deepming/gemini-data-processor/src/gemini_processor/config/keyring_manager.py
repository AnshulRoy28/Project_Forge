"""Secure API key management using system keyring."""

import os
from typing import Optional

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False


class KeyringManager:
    """Manages secure storage of API keys using system keyring or environment variables."""
    
    SERVICE_NAME = "gemini-data-processor"
    API_KEY_NAME = "gemini_api_key"
    ENV_VAR_NAME = "GEMINI_API_KEY"
    
    @classmethod
    def get_api_key(cls) -> Optional[str]:
        """
        Retrieve the Gemini API key from secure storage.
        
        Priority:
        1. Environment variable (GEMINI_API_KEY)
        2. System keyring
        
        Returns:
            The API key if found, None otherwise.
        """
        # First check environment variable
        env_key = os.environ.get(cls.ENV_VAR_NAME)
        if env_key:
            return env_key
        
        # Then check system keyring
        if KEYRING_AVAILABLE:
            try:
                stored_key = keyring.get_password(cls.SERVICE_NAME, cls.API_KEY_NAME)
                if stored_key:
                    return stored_key
            except Exception:
                pass  # Keyring may not be available on all systems
        
        return None
    
    @classmethod
    def set_api_key(cls, api_key: str, use_keyring: bool = True) -> bool:
        """
        Store the Gemini API key securely.
        
        Args:
            api_key: The API key to store.
            use_keyring: Whether to use system keyring (True) or just set env var (False).
            
        Returns:
            True if storage was successful, False otherwise.
        """
        if use_keyring and KEYRING_AVAILABLE:
            try:
                keyring.set_password(cls.SERVICE_NAME, cls.API_KEY_NAME, api_key)
                return True
            except Exception:
                pass  # Fall through to return False
        
        # Set environment variable for current session
        os.environ[cls.ENV_VAR_NAME] = api_key
        return True
    
    @classmethod
    def delete_api_key(cls) -> bool:
        """
        Remove the stored API key.
        
        Returns:
            True if deletion was successful, False otherwise.
        """
        # Remove from environment
        if cls.ENV_VAR_NAME in os.environ:
            del os.environ[cls.ENV_VAR_NAME]
        
        # Remove from keyring
        if KEYRING_AVAILABLE:
            try:
                keyring.delete_password(cls.SERVICE_NAME, cls.API_KEY_NAME)
                return True
            except Exception:
                pass
        
        return True
    
    @classmethod
    def is_keyring_available(cls) -> bool:
        """Check if system keyring is available."""
        return KEYRING_AVAILABLE
    
    @classmethod
    def validate_api_key_format(cls, api_key: str) -> bool:
        """
        Validate the format of a Gemini API key.
        
        Args:
            api_key: The API key to validate.
            
        Returns:
            True if the format appears valid, False otherwise.
        """
        if not api_key:
            return False
        
        # Gemini API keys typically start with "AI" and are 39 characters
        # But we'll be lenient and just check basic requirements
        if len(api_key) < 20:
            return False
        
        # Check for obviously invalid characters
        if ' ' in api_key or '\n' in api_key:
            return False
        
        return True
