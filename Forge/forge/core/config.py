"""
Configuration management for Forge.

Handles forge.yaml parsing, environment variables, and .forge/ directory.
"""

import os
from pathlib import Path
from typing import Optional, Any

import yaml
from pydantic import BaseModel, Field


# Default paths
FORGE_DIR_NAME = ".forge"
CONFIG_FILE_NAME = "forge.yaml"
CHECKPOINTS_DIR = "checkpoints"
OUTPUT_DIR = "output"
DATA_DIR = "data"


class LoRAConfig(BaseModel):
    """LoRA/QLoRA configuration."""
    
    rank: int = Field(default=16, ge=1, le=256, description="LoRA rank (r)")
    alpha: int = Field(default=32, ge=1, description="LoRA alpha for scaling")
    dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout")
    target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="Modules to apply LoRA to",
    )


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    
    # Model
    base_model: str = Field(default="unsloth/gemma-2b", description="Base model to fine-tune")
    quantization: Optional[str] = Field(default="4bit", description="Quantization: 4bit, 8bit, or null")
    max_seq_length: int = Field(default=2048, ge=128, le=32768, description="Maximum sequence length")
    
    # LoRA
    lora: LoRAConfig = Field(default_factory=LoRAConfig, description="LoRA configuration")
    
    # Training
    num_epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=4, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(default=4, ge=1, description="Gradient accumulation steps")
    learning_rate: float = Field(default=2e-4, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay")
    warmup_ratio: float = Field(default=0.03, ge=0, le=1, description="Warmup ratio")
    lr_scheduler: str = Field(default="linear", description="LR scheduler type")
    
    # Optimization
    use_gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing")
    optim: str = Field(default="adamw_8bit", description="Optimizer")
    
    # Checkpointing
    save_steps: int = Field(default=100, ge=1, description="Save checkpoint every N steps")
    logging_steps: int = Field(default=10, ge=1, description="Log every N steps")


class DataConfig(BaseModel):
    """Dataset configuration."""
    
    path: str = Field(description="Path to training data")
    format: str = Field(default="auto", description="Data format: csv, json, jsonl, txt, auto")
    text_column: str = Field(default="text", description="Column name for text data")
    train_split: float = Field(default=0.9, ge=0.1, le=1.0, description="Train/val split ratio")
    shuffle: bool = Field(default=True, description="Shuffle dataset")
    seed: int = Field(default=42, description="Random seed")
    max_samples: Optional[int] = Field(default=None, description="Maximum samples to use (for quick validation)")


class OutputConfig(BaseModel):
    """Output configuration."""
    
    # Docker mount paths (training always runs in Docker container)
    dir: str = Field(default="/output", description="Output directory (Docker: /output)")
    checkpoint_dir: str = Field(default="/checkpoints", description="Checkpoint directory (Docker: /checkpoints)")
    export_formats: list[str] = Field(
        default=["lora", "merged"],
        description="Export formats: lora, merged, gguf, ollama",
    )


class ForgeConfig(BaseModel):
    """Complete Forge configuration (forge.yaml schema)."""
    
    name: str = Field(default="forge-project", description="Project name")
    goal: str = Field(default="", description="Training goal in natural language")
    
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: Optional[DataConfig] = Field(default=None)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    # Metadata
    created_by: str = Field(default="gemini", description="Config created by: gemini or user")
    version: str = Field(default="1.0", description="Config version")


def get_forge_dir(base_path: Optional[Path] = None) -> Path:
    """Get the .forge directory path."""
    if base_path is None:
        base_path = Path.cwd()
    return base_path / FORGE_DIR_NAME


def ensure_forge_dir(base_path: Optional[Path] = None) -> Path:
    """Ensure .forge directory exists and return path."""
    forge_dir = get_forge_dir(base_path)
    forge_dir.mkdir(exist_ok=True)
    return forge_dir


def get_config_path(base_path: Optional[Path] = None) -> Path:
    """Get the forge.yaml config file path."""
    if base_path is None:
        base_path = Path.cwd()
    return base_path / CONFIG_FILE_NAME


def load_config(config_path: Optional[Path] = None) -> ForgeConfig:
    """Load configuration from forge.yaml."""
    if config_path is None:
        config_path = get_config_path()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    return ForgeConfig.model_validate(data)


def save_config(config: ForgeConfig, config_path: Optional[Path] = None) -> Path:
    """Save configuration to forge.yaml."""
    if config_path is None:
        config_path = get_config_path()
    
    data = config.model_dump(exclude_none=True)
    
    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    
    return config_path


def get_env_config() -> dict[str, Any]:
    """Get configuration from environment variables."""
    return {
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "deep_audit": os.getenv("FORGE_DEEP_AUDIT", "false").lower() == "true",
        "data_dir": os.getenv("FORGE_DATA_DIR", f"./{DATA_DIR}"),
        "checkpoint_dir": os.getenv("FORGE_CHECKPOINT_DIR", f"./{CHECKPOINTS_DIR}"),
        "output_dir": os.getenv("FORGE_OUTPUT_DIR", f"./{OUTPUT_DIR}"),
        "log_level": os.getenv("FORGE_LOG_LEVEL", "INFO"),
    }
