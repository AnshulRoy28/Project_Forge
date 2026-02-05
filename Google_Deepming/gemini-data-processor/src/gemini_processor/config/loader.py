"""Configuration loading and validation."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from ..models.config import (
    AppConfig,
    CLIConfig,
    GeminiConfig,
    ResourceConfig,
    SecurityConfig,
)
from .keyring_manager import KeyringManager


class ConfigurationLoader:
    """Loads and validates configuration from multiple sources."""
    
    CONFIG_FILE_NAME = ".gemini-processor.json"
    CONFIG_DIR_NAME = ".gemini-processor"
    
    def __init__(self, project_dir: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            project_dir: The project directory to use. Defaults to current directory.
        """
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self._config: Optional[AppConfig] = None
        
        # Load .env file if present
        dotenv_path = self.project_dir / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
    
    @property
    def config_dir(self) -> Path:
        """Get the configuration directory path."""
        return self.project_dir / self.CONFIG_DIR_NAME
    
    @property
    def config_file(self) -> Path:
        """Get the configuration file path."""
        return self.config_dir / self.CONFIG_FILE_NAME
    
    def ensure_config_dir(self) -> Path:
        """Ensure the configuration directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        return self.config_dir
    
    def load(self) -> AppConfig:
        """
        Load configuration from all sources.
        
        Priority (highest to lowest):
        1. Environment variables
        2. Config file in project directory
        3. Default values
        
        Returns:
            The loaded and validated application configuration.
        """
        # Start with defaults
        config = AppConfig(project_dir=str(self.project_dir))
        
        # Load from config file if exists
        if self.config_file.exists():
            file_config = self._load_from_file()
            config = self._merge_config(config, file_config)
        
        # Load Gemini API key
        api_key = KeyringManager.get_api_key()
        if api_key:
            config.gemini = GeminiConfig(api_key=api_key)
        
        # Override from environment variables
        config = self._load_from_env(config)
        
        self._config = config
        return config
    
    def save(self, config: AppConfig) -> None:
        """
        Save configuration to the config file.
        
        Note: API keys are NOT saved to file - they use keyring/env vars.
        
        Args:
            config: The configuration to save.
        """
        self.ensure_config_dir()
        
        # Create config dict without sensitive data
        config_dict = {
            "security": {
                "enable_container_isolation": config.security.enable_container_isolation,
                "run_containers_as_non_root": config.security.run_containers_as_non_root,
                "enable_network_isolation": config.security.enable_network_isolation,
                "sanitize_sensitive_data": config.security.sanitize_sensitive_data,
            },
            "resources": {
                "max_container_memory_gb": config.resources.max_container_memory_gb,
                "max_container_cpu_cores": config.resources.max_container_cpu_cores,
                "max_container_disk_gb": config.resources.max_container_disk_gb,
                "max_execution_time_minutes": config.resources.max_execution_time_minutes,
            },
            "gemini": {
                "model_name": config.gemini.model_name if config.gemini else "gemini-2.0-flash",
                "max_tokens": config.gemini.max_tokens if config.gemini else 8192,
                "temperature": config.gemini.temperature if config.gemini else 0.7,
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def is_initialized(self) -> bool:
        """Check if the project has been initialized with a valid configuration."""
        return self.config_dir.exists() and KeyringManager.get_api_key() is not None
    
    def _load_from_file(self) -> Dict[str, Any]:
        """Load configuration from the config file."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _load_from_env(self, config: AppConfig) -> AppConfig:
        """Override configuration from environment variables."""
        # Gemini settings
        if model := os.environ.get("GEMINI_MODEL"):
            if config.gemini:
                config.gemini.model_name = model
        
        if max_tokens := os.environ.get("GEMINI_MAX_TOKENS"):
            if config.gemini:
                try:
                    config.gemini.max_tokens = int(max_tokens)
                except ValueError:
                    pass
        
        # Resource settings
        if memory := os.environ.get("GDP_MAX_MEMORY_GB"):
            try:
                config.resources.max_container_memory_gb = int(memory)
            except ValueError:
                pass
        
        if cpu := os.environ.get("GDP_MAX_CPU_CORES"):
            try:
                config.resources.max_container_cpu_cores = int(cpu)
            except ValueError:
                pass
        
        return config
    
    def _merge_config(self, base: AppConfig, override: Dict[str, Any]) -> AppConfig:
        """Merge file configuration into base configuration."""
        if "security" in override:
            sec = override["security"]
            base.security = SecurityConfig(
                enable_container_isolation=sec.get(
                    "enable_container_isolation", 
                    base.security.enable_container_isolation
                ),
                run_containers_as_non_root=sec.get(
                    "run_containers_as_non_root",
                    base.security.run_containers_as_non_root
                ),
                enable_network_isolation=sec.get(
                    "enable_network_isolation",
                    base.security.enable_network_isolation
                ),
                sanitize_sensitive_data=sec.get(
                    "sanitize_sensitive_data",
                    base.security.sanitize_sensitive_data
                ),
            )
        
        if "resources" in override:
            res = override["resources"]
            base.resources = ResourceConfig(
                max_container_memory_gb=res.get(
                    "max_container_memory_gb",
                    base.resources.max_container_memory_gb
                ),
                max_container_cpu_cores=res.get(
                    "max_container_cpu_cores",
                    base.resources.max_container_cpu_cores
                ),
                max_container_disk_gb=res.get(
                    "max_container_disk_gb",
                    base.resources.max_container_disk_gb
                ),
                max_execution_time_minutes=res.get(
                    "max_execution_time_minutes",
                    base.resources.max_execution_time_minutes
                ),
            )
        
        return base
