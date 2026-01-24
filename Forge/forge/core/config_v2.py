"""
Enhanced configuration system for model-aware preprocessing pipeline.

This module provides data classes and utilities for managing Forge configuration
with model awareness, hardware optimization, and preprocessing parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from pathlib import Path
import yaml
from enum import Enum


class ChatTemplate(str, Enum):
    """Supported chat templates for different model architectures."""
    CHATML = "chatml"
    LLAMA = "llama"
    ALPACA = "alpaca"
    GEMMA = "gemma"
    VICUNA = "vicuna"
    CUSTOM = "custom"


class GPUArchitecture(str, Enum):
    """Supported GPU architectures."""
    BLACKWELL = "blackwell"
    ADA = "ada"
    AMPERE = "ampere"
    HOPPER = "hopper"
    BASE = "base"


@dataclass
class ModelConfig:
    """Configuration for the target model."""
    name: str
    architecture: str
    chat_template: ChatTemplate
    max_length: int = 2048
    special_tokens: Dict[str, str] = field(default_factory=dict)
    model_family: Optional[str] = None
    requires_system_prompt: bool = False
    
    def __post_init__(self):
        """Validate model configuration after initialization."""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        
        if isinstance(self.chat_template, str):
            self.chat_template = ChatTemplate(self.chat_template)


@dataclass
class HardwareConfig:
    """Configuration for hardware-specific optimizations."""
    gpu_arch: GPUArchitecture
    vram_gb: float
    compute_capability: float
    recommended_batch_size: int
    use_gpu_preprocessing: bool = True
    max_memory_usage: float = 0.8  # Fraction of available memory to use
    
    def __post_init__(self):
        """Validate hardware configuration after initialization."""
        if self.vram_gb <= 0:
            raise ValueError("vram_gb must be positive")
        
        if not 0 < self.max_memory_usage <= 1:
            raise ValueError("max_memory_usage must be between 0 and 1")
        
        if isinstance(self.gpu_arch, str):
            self.gpu_arch = GPUArchitecture(self.gpu_arch)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing parameters."""
    train_split: float = 0.9
    validation_split: float = 0.1
    test_split: float = 0.0
    chunk_size: int = 1000
    quality_checks: bool = True
    streaming_threshold_gb: float = 2.0
    remove_duplicates: bool = True
    min_text_length: int = 10
    max_text_length: Optional[int] = None
    stratify_column: Optional[str] = None
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate preprocessing configuration after initialization."""
        total_split = self.train_split + self.validation_split + self.test_split
        if abs(total_split - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_split}")
        
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.min_text_length < 0:
            raise ValueError("min_text_length must be non-negative")
        
        if self.max_text_length is not None and self.max_text_length <= self.min_text_length:
            raise ValueError("max_text_length must be greater than min_text_length")


@dataclass
class ContainerConfig:
    """Configuration for persistent Docker container management."""
    container_id: Optional[str] = None
    container_name: Optional[str] = None
    model_cached: bool = False
    last_used: Optional[str] = None
    gpu_arch: Optional[str] = None


@dataclass
class ForgeConfig:
    """Main configuration class combining all aspects of Forge setup."""
    model: ModelConfig
    hardware: HardwareConfig
    preprocessing: PreprocessingConfig
    container: ContainerConfig = None
    version: str = "2.0"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __post_init__(self):
        """Initialize container config if not provided."""
        if self.container is None:
            self.container = ContainerConfig()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ForgeConfig':
        """Create ForgeConfig from dictionary (e.g., loaded from YAML)."""
        model_data = data.get('model', {})
        hardware_data = data.get('hardware', {})
        preprocessing_data = data.get('preprocessing', {})
        container_data = data.get('container', {})
        
        return cls(
            model=ModelConfig(**model_data),
            hardware=HardwareConfig(**hardware_data),
            preprocessing=PreprocessingConfig(**preprocessing_data),
            container=ContainerConfig(
                container_id=container_data.get('container_id'),
                container_name=container_data.get('container_name'),
                model_cached=container_data.get('model_cached', False),
                last_used=container_data.get('last_used'),
                gpu_arch=container_data.get('gpu_arch'),
            ),
            version=data.get('version', '2.0'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ForgeConfig to dictionary for YAML serialization."""
        return {
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'model': {
                'name': self.model.name,
                'architecture': self.model.architecture,
                'chat_template': self.model.chat_template.value,
                'max_length': self.model.max_length,
                'special_tokens': self.model.special_tokens,
                'model_family': self.model.model_family,
                'requires_system_prompt': self.model.requires_system_prompt,
            },
            'hardware': {
                'gpu_arch': self.hardware.gpu_arch.value,
                'vram_gb': self.hardware.vram_gb,
                'compute_capability': self.hardware.compute_capability,
                'recommended_batch_size': self.hardware.recommended_batch_size,
                'use_gpu_preprocessing': self.hardware.use_gpu_preprocessing,
                'max_memory_usage': self.hardware.max_memory_usage,
            },
            'preprocessing': {
                'train_split': self.preprocessing.train_split,
                'validation_split': self.preprocessing.validation_split,
                'test_split': self.preprocessing.test_split,
                'chunk_size': self.preprocessing.chunk_size,
                'quality_checks': self.preprocessing.quality_checks,
                'streaming_threshold_gb': self.preprocessing.streaming_threshold_gb,
                'remove_duplicates': self.preprocessing.remove_duplicates,
                'min_text_length': self.preprocessing.min_text_length,
                'max_text_length': self.preprocessing.max_text_length,
                'stratify_column': self.preprocessing.stratify_column,
                'random_seed': self.preprocessing.random_seed,
            },
            'container': {
                'container_id': self.container.container_id,
                'container_name': self.container.container_name,
                'model_cached': self.container.model_cached,
                'last_used': self.container.last_used,
                'gpu_arch': self.container.gpu_arch,
            }
        }
    
    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import datetime
        
        # Update timestamp
        self.updated_at = datetime.datetime.now().isoformat()
        if self.created_at is None:
            self.created_at = self.updated_at
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ForgeConfig':
        """Load configuration from YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    def validate(self) -> List[str]:
        """Validate the entire configuration and return list of errors."""
        errors = []
        
        # Model validation
        try:
            if not self.model.name:
                errors.append("Model name cannot be empty")
            
            if not self.model.architecture:
                errors.append("Model architecture cannot be empty")
                
        except Exception as e:
            errors.append(f"Model configuration error: {e}")
        
        # Hardware validation
        try:
            if self.hardware.vram_gb <= 0:
                errors.append("VRAM must be positive")
                
            if self.hardware.recommended_batch_size <= 0:
                errors.append("Batch size must be positive")
                
        except Exception as e:
            errors.append(f"Hardware configuration error: {e}")
        
        # Preprocessing validation
        try:
            total_split = (self.preprocessing.train_split + 
                          self.preprocessing.validation_split + 
                          self.preprocessing.test_split)
            if abs(total_split - 1.0) > 0.001:
                errors.append(f"Split ratios must sum to 1.0, got {total_split}")
                
        except Exception as e:
            errors.append(f"Preprocessing configuration error: {e}")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


# Configuration utilities
def create_default_config(
    model_name: str,
    gpu_arch: GPUArchitecture,
    vram_gb: float,
    compute_capability: float
) -> ForgeConfig:
    """Create a default configuration with sensible defaults."""
    
    # Determine model family and template from name
    model_family, chat_template = _detect_model_info(model_name)
    
    # Calculate recommended batch size based on VRAM
    batch_size = _calculate_batch_size(vram_gb, compute_capability)
    
    # Set aggressive memory usage for high-end GPUs
    max_memory_usage = 0.95 if vram_gb >= 15 else 0.85 if vram_gb >= 12 else 0.8
    
    return ForgeConfig(
        model=ModelConfig(
            name=model_name,
            architecture=_detect_architecture(model_name),
            chat_template=chat_template,
            model_family=model_family,
            max_length=2048,
        ),
        hardware=HardwareConfig(
            gpu_arch=gpu_arch,
            vram_gb=vram_gb,
            compute_capability=compute_capability,
            recommended_batch_size=batch_size,
            max_memory_usage=max_memory_usage,
        ),
        preprocessing=PreprocessingConfig()
    )


def _detect_model_info(model_name: str) -> tuple[Optional[str], ChatTemplate]:
    """Detect model family and appropriate chat template from model name."""
    model_lower = model_name.lower()
    
    if "llama" in model_lower:
        return "llama", ChatTemplate.LLAMA
    elif "gemma" in model_lower:
        return "gemma", ChatTemplate.GEMMA
    elif "vicuna" in model_lower:
        return "vicuna", ChatTemplate.VICUNA
    elif "alpaca" in model_lower:
        return "alpaca", ChatTemplate.ALPACA
    elif "dialogpt" in model_lower or "gpt" in model_lower:
        return "gpt", ChatTemplate.CHATML
    else:
        return None, ChatTemplate.CHATML  # Default fallback


def _detect_architecture(model_name: str) -> str:
    """Detect model architecture from model name."""
    model_lower = model_name.lower()
    
    if "llama" in model_lower:
        return "llama"
    elif "gemma" in model_lower:
        return "gemma"
    elif "gpt" in model_lower:
        return "gpt"
    elif "t5" in model_lower:
        return "t5"
    else:
        return "transformer"  # Generic fallback


def _calculate_batch_size(vram_gb: float, compute_capability: float) -> int:
    """Calculate recommended batch size based on hardware for maximum training speed."""
    # Aggressive batch size calculation for high-end GPUs
    # Goal: finish training as fast as possible by maximizing GPU utilization
    
    if vram_gb >= 24:
        return 32  # RTX 4090/5090/H100 - use very large batches
    elif vram_gb >= 15:  # RTX 5080 has 15.92GB - treat as high-end
        return 24  # RTX 5080/4080 - maximize utilization for speed
    elif vram_gb >= 12:
        return 16  # RTX 4070 Ti/3080 Ti - aggressive batching
    elif vram_gb >= 10:
        return 12  # RTX 4070/3080 - push harder
    elif vram_gb >= 8:
        return 8   # RTX 4060 Ti/3070 - still aggressive
    else:
        return 4   # Lower-end GPUs - conservative


# Configuration validation utilities
def validate_config_file(path: Path) -> tuple[bool, List[str]]:
    """Validate a configuration file and return status and errors."""
    try:
        config = ForgeConfig.load(path)
        errors = config.validate()
        return len(errors) == 0, errors
    except Exception as e:
        return False, [f"Failed to load configuration: {e}"]


def migrate_config(old_path: Path, new_path: Path) -> bool:
    """Migrate old configuration format to new format."""
    try:
        # This would implement migration logic for older config versions
        # For now, just copy if it's already in the new format
        if old_path.exists():
            config = ForgeConfig.load(old_path)
            config.save(new_path)
            return True
        return False
    except Exception:
        return False