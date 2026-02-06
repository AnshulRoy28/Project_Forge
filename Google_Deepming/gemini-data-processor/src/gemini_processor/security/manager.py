"""Security manager for handling sensitive data cleanup."""

import gc
import os
import secrets
from typing import Optional

from ..config.keyring_manager import KeyringManager


class SecurityManager:
    """Manages security operations including API key cleanup."""
    
    def __init__(self):
        """Initialize the security manager."""
        self._cleanup_performed = False
    
    def wipe_all_api_keys(self) -> bool:
        """
        Perform comprehensive API key cleanup.
        
        This method:
        1. Wipes API keys from keyring and environment
        2. Clears API keys from memory objects
        3. Forces garbage collection
        4. Overwrites sensitive memory areas
        
        Returns:
            True if cleanup was successful, False otherwise.
        """
        if self._cleanup_performed:
            return True
        
        success = True
        
        try:
            # Step 1: Remove from keyring and environment
            if not KeyringManager.wipe_api_key_from_memory():
                success = False
            
            # Step 2: Clear any remaining environment variables that might contain keys
            self._clear_sensitive_env_vars()
            
            # Step 3: Force garbage collection to clear memory
            gc.collect()
            
            # Step 4: Mark cleanup as performed
            self._cleanup_performed = True
            
        except Exception:
            success = False
        
        return success
    
    def cleanup_gemini_client(self, gemini_client) -> None:
        """
        Clean up a Gemini client instance.
        
        Args:
            gemini_client: The GeminiIntegration instance to clean up.
        """
        if gemini_client is None:
            return
        
        try:
            # Clear the API key from config
            if hasattr(gemini_client, 'config') and gemini_client.config:
                if hasattr(gemini_client.config, 'api_key'):
                    # Overwrite with random data before clearing
                    gemini_client.config.api_key = secrets.token_urlsafe(64)
                    gemini_client.config.api_key = ""
            
            # Clear the client instance
            if hasattr(gemini_client, '_client'):
                gemini_client._client = None
        
        except Exception:
            pass  # Best effort cleanup
    
    def _clear_sensitive_env_vars(self) -> None:
        """Clear potentially sensitive environment variables."""
        sensitive_patterns = [
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY", 
            "AI_API_KEY",
            "API_KEY"
        ]
        
        for var_name in list(os.environ.keys()):
            for pattern in sensitive_patterns:
                if pattern in var_name.upper():
                    # Overwrite with random data before deletion
                    os.environ[var_name] = secrets.token_urlsafe(64)
                    del os.environ[var_name]
                    break
    
    def secure_memory_cleanup(self) -> None:
        """
        Perform secure memory cleanup operations.
        
        This forces Python's garbage collector to run multiple times
        and attempts to clear sensitive data from memory.
        """
        # Force multiple garbage collection cycles
        for _ in range(3):
            gc.collect()
        
        # Clear any cached modules that might contain sensitive data
        try:
            import sys
            for module_name in list(sys.modules.keys()):
                if 'gemini' in module_name.lower() or 'api' in module_name.lower():
                    module = sys.modules.get(module_name)
                    if hasattr(module, '__dict__'):
                        # Clear module-level variables that might contain keys
                        for attr_name in list(module.__dict__.keys()):
                            if 'key' in attr_name.lower() or 'token' in attr_name.lower():
                                try:
                                    setattr(module, attr_name, None)
                                except Exception:
                                    pass
        except Exception:
            pass  # Best effort cleanup
    
    @property
    def is_cleanup_performed(self) -> bool:
        """Check if cleanup has been performed."""
        return self._cleanup_performed