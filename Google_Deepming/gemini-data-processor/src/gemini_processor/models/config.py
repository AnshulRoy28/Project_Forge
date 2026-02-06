"""Configuration models for the Gemini Data Processor."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CLIConfig:
    """Configuration from command-line arguments."""
    input_file: str
    output_directory: Optional[str] = None
    verbose: bool = False
    quiet: bool = False
    max_file_size_gb: float = 1.0
    docker_timeout: int = 300  # 5 minutes
    enable_progress_indicators: bool = True
    concurrent_containers: int = 1


@dataclass
class GeminiConfig:
    """Configuration for Gemini AI integration."""
    api_key: str
    model_name: str = "gemini-2.0-flash"
    max_tokens: int = 8192
    temperature: float = 0.7
    rate_limit_requests_per_minute: int = 60
    enable_data_sanitization: bool = True
    max_context_size: int = 100000


@dataclass
class SecurityConfig:
    """Security configuration for the system."""
    enable_container_isolation: bool = True
    run_containers_as_non_root: bool = True
    enable_network_isolation: bool = True
    sanitize_sensitive_data: bool = True
    max_api_key_age_days: int = 90


@dataclass
class ResourceConfig:
    """Resource limits and configuration."""
    max_container_memory_gb: int = 4
    max_container_cpu_cores: int = 2
    max_container_disk_gb: int = 10
    max_execution_time_minutes: int = 30
    max_concurrent_containers: int = 1


@dataclass
class AppConfig:
    """Main application configuration combining all configs."""
    cli: CLIConfig = field(default_factory=lambda: CLIConfig(input_file=""))
    gemini: Optional[GeminiConfig] = None
    security: SecurityConfig = field(default_factory=SecurityConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    project_dir: str = "."
    
    @property
    def is_configured(self) -> bool:
        """Check if the application has been properly configured."""
        return self.gemini is not None and self.gemini.api_key != ""
