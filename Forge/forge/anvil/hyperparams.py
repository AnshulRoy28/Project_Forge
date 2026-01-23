"""
VRAM-aware hyperparameter tuning for Forge.
"""

from dataclasses import dataclass
from typing import Optional

from forge.core.hardware import HardwareProfile, detect_hardware


@dataclass
class OptimalHyperparams:
    """Optimal hyperparameters based on hardware."""
    
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lora_rank: int
    lora_alpha: int
    max_seq_length: int
    quantization: Optional[str]
    use_gradient_checkpointing: bool
    estimated_vram_usage: float  # GB


def calculate_optimal_hyperparams(
    dataset_size: int,
    avg_seq_length: int,
    target_epochs: int = 3,
    hardware: Optional[HardwareProfile] = None,
) -> OptimalHyperparams:
    """
    Calculate optimal hyperparameters based on hardware and dataset.
    
    Args:
        dataset_size: Number of training samples
        avg_seq_length: Average sequence length in tokens
        target_epochs: Target number of training epochs
        hardware: Hardware profile (auto-detected if None)
    """
    if hardware is None:
        hardware = detect_hardware()
    
    vram_gb = hardware.gpu.vram_total_gb if hardware.gpu else 0
    
    # Base configurations by VRAM tier
    if vram_gb < 8:
        # Very limited VRAM
        config = OptimalHyperparams(
            batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            lora_rank=4,
            lora_alpha=8,
            max_seq_length=min(512, avg_seq_length * 2),
            quantization="4bit",
            use_gradient_checkpointing=True,
            estimated_vram_usage=6.0,
        )
    elif vram_gb < 12:
        # Limited VRAM (8-12GB)
        config = OptimalHyperparams(
            batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            lora_rank=8,
            lora_alpha=16,
            max_seq_length=min(1024, avg_seq_length * 2),
            quantization="4bit",
            use_gradient_checkpointing=True,
            estimated_vram_usage=10.0,
        )
    elif vram_gb < 24:
        # Mid-range VRAM (12-24GB)
        config = OptimalHyperparams(
            batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            lora_rank=32,
            lora_alpha=64,
            max_seq_length=min(2048, avg_seq_length * 2),
            quantization="4bit",
            use_gradient_checkpointing=True,
            estimated_vram_usage=20.0,
        )
    elif vram_gb < 48:
        # High VRAM (24-48GB)
        config = OptimalHyperparams(
            batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            lora_rank=64,
            lora_alpha=128,
            max_seq_length=min(4096, avg_seq_length * 2),
            quantization="8bit",
            use_gradient_checkpointing=False,
            estimated_vram_usage=40.0,
        )
    else:
        # Very high VRAM (48GB+)
        config = OptimalHyperparams(
            batch_size=16,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            lora_rank=128,
            lora_alpha=256,
            max_seq_length=min(8192, avg_seq_length * 2),
            quantization=None,
            use_gradient_checkpointing=False,
            estimated_vram_usage=60.0,
        )
    
    # Adjust based on dataset size
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = dataset_size // effective_batch
    
    # If too few steps per epoch, reduce batch size
    if steps_per_epoch < 10 and config.batch_size > 1:
        config.batch_size = max(1, config.batch_size // 2)
        config.gradient_accumulation_steps = config.gradient_accumulation_steps * 2
    
    # Adjust learning rate for larger datasets
    if dataset_size > 10000:
        config.learning_rate = config.learning_rate * 0.5
    elif dataset_size < 1000:
        config.learning_rate = config.learning_rate * 1.5
    
    return config


def estimate_training_time(
    dataset_size: int,
    hyperparams: OptimalHyperparams,
    epochs: int = 3,
    hardware: Optional[HardwareProfile] = None,
) -> float:
    """
    Estimate training time in minutes.
    
    Returns approximate time based on hardware and configuration.
    """
    if hardware is None:
        hardware = detect_hardware()
    
    effective_batch = hyperparams.batch_size * hyperparams.gradient_accumulation_steps
    total_steps = (dataset_size // effective_batch) * epochs
    
    # Rough estimates: seconds per step based on VRAM tier
    vram_gb = hardware.gpu.vram_total_gb if hardware.gpu else 0
    
    if vram_gb >= 24:
        seconds_per_step = 0.5
    elif vram_gb >= 12:
        seconds_per_step = 1.0
    elif vram_gb >= 8:
        seconds_per_step = 2.0
    else:
        seconds_per_step = 5.0  # CPU or very limited
    
    # Adjust for quantization
    if hyperparams.quantization:
        seconds_per_step *= 1.2  # Slight overhead for quantized training
    
    total_seconds = total_steps * seconds_per_step
    return total_seconds / 60  # Return minutes
