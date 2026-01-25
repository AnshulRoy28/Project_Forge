"""
Hardware detection and monitoring for Forge.

Detects GPU architecture, VRAM, and provides hardware-aware training recommendations
optimized for different GPU generations (Ampere, Ada Lovelace, Blackwell, etc.)
"""

import platform
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

import psutil

# Try to import GPU monitoring libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
        try:
            import pynvml  # type: ignore[import]
        except ImportError:
            # Try the new package name
            import nvidia_ml_py as pynvml  # type: ignore[import]
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUArchitecture(Enum):
    """NVIDIA GPU architectures with their compute capabilities."""
    UNKNOWN = "unknown"
    PASCAL = "pascal"           # GTX 10xx, compute 6.x
    TURING = "turing"           # RTX 20xx, compute 7.5
    AMPERE = "ampere"           # RTX 30xx, compute 8.x
    ADA_LOVELACE = "ada"        # RTX 40xx, compute 8.9
    BLACKWELL = "blackwell"     # RTX 50xx, compute 12.x
    HOPPER = "hopper"           # H100, compute 9.0
    

def get_gpu_architecture(compute_capability: tuple) -> GPUArchitecture:
    """Determine GPU architecture from compute capability."""
    major, minor = compute_capability
    
    if major >= 12:
        return GPUArchitecture.BLACKWELL
    elif major == 9:
        return GPUArchitecture.HOPPER
    elif major == 8 and minor >= 9:
        return GPUArchitecture.ADA_LOVELACE
    elif major == 8:
        return GPUArchitecture.AMPERE
    elif major == 7 and minor >= 5:
        return GPUArchitecture.TURING
    elif major >= 6:
        return GPUArchitecture.PASCAL
    else:
        return GPUArchitecture.UNKNOWN


@dataclass
class GPUInfo:
    """Information about the detected GPU."""
    
    name: str
    vram_total_gb: float
    vram_free_gb: float
    vram_used_gb: float
    cuda_version: str
    driver_version: str
    temperature: int  # Celsius
    compute_capability: tuple
    architecture: GPUArchitecture
    
    @property
    def vram_usage_pct(self) -> float:
        """Get VRAM usage percentage."""
        if self.vram_total_gb == 0:
            return 0.0
        return (self.vram_used_gb / self.vram_total_gb) * 100
    
    @property
    def supports_bf16(self) -> bool:
        """Check if GPU supports BFloat16 (Ampere+)."""
        return self.architecture in [
            GPUArchitecture.AMPERE,
            GPUArchitecture.ADA_LOVELACE,
            GPUArchitecture.BLACKWELL,
            GPUArchitecture.HOPPER,
        ]
    
    @property
    def supports_fp8(self) -> bool:
        """Check if GPU supports FP8 (Ada Lovelace+ for inference, Blackwell+ for training)."""
        return self.architecture in [
            GPUArchitecture.ADA_LOVELACE,
            GPUArchitecture.BLACKWELL,
            GPUArchitecture.HOPPER,
        ]
    
    @property
    def supports_flash_attention(self) -> bool:
        """Check if GPU supports Flash Attention (Ampere+)."""
        return self.compute_capability[0] >= 8


@dataclass
class SystemInfo:
    """Information about the system."""
    
    os_name: str
    os_version: str
    cpu_name: str
    cpu_cores: int
    ram_total_gb: float
    ram_free_gb: float
    ram_used_gb: float
    
    @property
    def ram_usage_pct(self) -> float:
        """Get RAM usage percentage."""
        if self.ram_total_gb == 0:
            return 0.0
        return (self.ram_used_gb / self.ram_total_gb) * 100


@dataclass
class TrainingConfig:
    """Recommended training configuration based on hardware."""
    
    # Precision settings
    quantization: Optional[str]  # "4bit", "8bit", or None for full
    use_bf16: bool
    use_fp16: bool
    use_tf32: bool
    
    # LoRA settings
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    
    # Training settings
    batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    use_gradient_checkpointing: bool
    optimizer: str
    
    # Model recommendation
    model_recommendation: str
    model_id: str  # Actual Unsloth model ID
    ollama_model: str  # Corresponding Ollama model name for testing
    
    # Architecture-specific notes
    notes: str


@dataclass
class HardwareProfile:
    """Complete hardware profile for the system."""
    
    gpu: Optional[GPUInfo]
    system: SystemInfo
    has_cuda: bool
    has_rocm: bool
    
    def recommend_config(self) -> TrainingConfig:
        """Generate hardware-optimized training configuration."""
        if not self.gpu or not self.has_cuda:
            return self._cpu_config()
        
        arch = self.gpu.architecture
        vram = self.gpu.vram_total_gb
        
        # Architecture-specific optimizations
        if arch == GPUArchitecture.BLACKWELL:
            return self._blackwell_config(vram)
        elif arch == GPUArchitecture.ADA_LOVELACE:
            return self._ada_config(vram)
        elif arch == GPUArchitecture.AMPERE:
            return self._ampere_config(vram)
        elif arch == GPUArchitecture.HOPPER:
            return self._hopper_config(vram)
        else:
            return self._legacy_config(vram)
    
    def _cpu_config(self) -> TrainingConfig:
        """Configuration for CPU-only (not recommended)."""
        return TrainingConfig(
            quantization="4bit",
            use_bf16=False,
            use_fp16=False,
            use_tf32=False,
            lora_rank=4,
            lora_alpha=4,
            lora_dropout=0.05,
            batch_size=1,
            gradient_accumulation_steps=16,
            max_seq_length=512,
            use_gradient_checkpointing=True,
            optimizer="adamw_torch",
            model_recommendation="Gemma 2B (CPU mode - very slow)",
            model_id="unsloth/gemma-2-2b-it-bnb-4bit",
            ollama_model="gemma2:2b",
            notes="CPU training is extremely slow. Consider using a cloud GPU.",
        )
    
    def _blackwell_config(self, vram: float) -> TrainingConfig:
        """
        RTX 50-series (Blackwell) optimizations:
        - Native BF16 support (prevents NaN errors)
        - TF32 for 3x speedup on Tensor Cores
        - FP8 capable for future optimizations
        - High memory bandwidth (960 GB/s on 5080)
        - AGGRESSIVE batch sizes for maximum training speed
        """
        if vram >= 24:  # RTX 5090
            return TrainingConfig(
                quantization="4bit",
                use_bf16=True,
                use_fp16=False,
                use_tf32=True,
                lora_rank=32,
                lora_alpha=32,
                lora_dropout=0,
                batch_size=32,  # Maximize RTX 5090 utilization
                gradient_accumulation_steps=1,
                max_seq_length=4096,
                use_gradient_checkpointing=True,
                optimizer="adamw_8bit",
                model_recommendation="Gemma 9B or Llama 8B",
                model_id="unsloth/gemma-2-9b-it-bnb-4bit",
                ollama_model="gemma2:9b",
                notes="RTX 5090 Blackwell: BF16 native, TF32 enabled, maximum batch size for speed.",
            )
        else:  # RTX 5080 (16GB) or 5070
            return TrainingConfig(
                quantization="4bit",
                use_bf16=True,
                use_fp16=False,
                use_tf32=True,
                lora_rank=16,
                lora_alpha=16,
                lora_dropout=0,
                batch_size=24,  # RTX 5080: aggressive batch size for maximum speed
                gradient_accumulation_steps=1,
                max_seq_length=2048,
                use_gradient_checkpointing=True,
                optimizer="adamw_8bit",
                model_recommendation="Gemma 7B (4-bit) for optimal speed",
                model_id="unsloth/gemma-2-2b-it-bnb-4bit",
                ollama_model="gemma2:2b",
                notes="RTX 5080 Blackwell: BF16 native, TF32 enabled, batch size 24 for maximum training speed.",
            )
    
    def _ada_config(self, vram: float) -> TrainingConfig:
        """RTX 40-series (Ada Lovelace) optimizations - aggressive for speed."""
        if vram >= 24:  # RTX 4090
            return TrainingConfig(
                quantization="4bit",
                use_bf16=True,
                use_fp16=False,
                use_tf32=True,
                lora_rank=64,
                lora_alpha=64,
                lora_dropout=0,
                batch_size=32,  # Maximize RTX 4090 utilization
                gradient_accumulation_steps=1,
                max_seq_length=4096,
                use_gradient_checkpointing=False,
                optimizer="adamw_8bit",
                model_recommendation="Gemma 9B",
                model_id="unsloth/gemma-2-9b-it-bnb-4bit",
                ollama_model="gemma2:9b",
                notes="RTX 4090 Ada: Full BF16 support, maximum batch size for speed.",
            )
        elif vram >= 16:  # RTX 4080, 4070 Ti Super
            return TrainingConfig(
                quantization="4bit",
                use_bf16=True,
                use_fp16=False,
                use_tf32=True,
                lora_rank=32,
                lora_alpha=32,
                lora_dropout=0,
                batch_size=20,  # Aggressive batch size for RTX 4080
                gradient_accumulation_steps=1,
                max_seq_length=2048,
                use_gradient_checkpointing=True,
                optimizer="adamw_8bit",
                model_recommendation="Gemma 7B (4-bit)",
                model_id="unsloth/gemma-2-9b-it-bnb-4bit",
                ollama_model="gemma2:9b",
                notes="RTX 4080 Ada: BF16 enabled, aggressive batching for maximum speed.",
            )
        else:  # RTX 4070, 4060
            return TrainingConfig(
                quantization="4bit",
                use_bf16=True,
                use_fp16=False,
                use_tf32=True,
                lora_rank=16,
                lora_alpha=16,
                lora_dropout=0.05,
                batch_size=12,  # More aggressive for mid-range
                gradient_accumulation_steps=2,
                max_seq_length=1024,
                use_gradient_checkpointing=True,
                optimizer="adamw_8bit",
                model_recommendation="Gemma 2B",
                model_id="unsloth/gemma-2-2b-it-bnb-4bit",
                ollama_model="gemma2:2b",
                notes="RTX 4060/4070 Ada: Aggressive batching for faster training.",
            )
    
    def _ampere_config(self, vram: float) -> TrainingConfig:
        """RTX 30-series (Ampere) optimizations - aggressive for speed."""
        if vram >= 24:  # RTX 3090
            return TrainingConfig(
                quantization="4bit",
                use_bf16=True,
                use_fp16=False,
                use_tf32=True,
                lora_rank=32,
                lora_alpha=32,
                lora_dropout=0,
                batch_size=24,  # Aggressive for RTX 3090
                gradient_accumulation_steps=1,
                max_seq_length=2048,
                use_gradient_checkpointing=True,
                optimizer="adamw_8bit",
                model_recommendation="Gemma 9B (4-bit)",
                model_id="unsloth/gemma-2-9b-it-bnb-4bit",
                ollama_model="gemma2:9b",
                notes="RTX 3090 Ampere: Good BF16 support, TF32 enabled, aggressive batching.",
            )
        elif vram >= 10:  # RTX 3080
            return TrainingConfig(
                quantization="4bit",
                use_bf16=True,
                use_fp16=False,
                use_tf32=True,
                lora_rank=16,
                lora_alpha=16,
                lora_dropout=0.05,
                batch_size=12,  # More aggressive for RTX 3080
                gradient_accumulation_steps=2,
                max_seq_length=1024,
                use_gradient_checkpointing=True,
                optimizer="adamw_8bit",
                model_recommendation="Gemma 2B",
                model_id="unsloth/gemma-2-2b-it-bnb-4bit",
                ollama_model="gemma2:2b",
                notes="RTX 3080 Ampere: Aggressive batching for faster training.",
            )
        else:  # RTX 3070, 3060
            return TrainingConfig(
                quantization="4bit",
                use_bf16=False,
                use_fp16=True,
                use_tf32=True,
                lora_rank=8,
                lora_alpha=8,
                lora_dropout=0.1,
                batch_size=8,  # More aggressive than before
                gradient_accumulation_steps=4,
                max_seq_length=512,
                use_gradient_checkpointing=True,
                optimizer="adamw_8bit",
                model_recommendation="Gemma 2B",
                model_id="unsloth/gemma-2-2b-it-bnb-4bit",
                ollama_model="gemma2:2b",
                notes="RTX 3060/3070: FP16 for stability, aggressive batching for speed.",
            )
    
    def _hopper_config(self, vram: float) -> TrainingConfig:
        """H100/H200 (Hopper) optimizations."""
        return TrainingConfig(
            quantization=None,  # Full precision for datacenter
            use_bf16=True,
            use_fp16=False,
            use_tf32=True,
            lora_rank=128,
            lora_alpha=128,
            lora_dropout=0,
            batch_size=16,
            gradient_accumulation_steps=1,
            max_seq_length=8192,
            use_gradient_checkpointing=False,
            optimizer="adamw_torch",
            model_recommendation="Gemma 9B (full precision) or larger",
            model_id="google/gemma-2-9b-it",
            ollama_model="gemma2:9b",
            notes="H100 Hopper: Massive VRAM & bandwidth, full precision training.",
        )
    
    def _legacy_config(self, vram: float) -> TrainingConfig:
        """Legacy GPU (Turing, Pascal) or unknown architecture."""
        return TrainingConfig(
            quantization="4bit",
            use_bf16=False,
            use_fp16=True,  # FP16 for older GPUs
            use_tf32=False,
            lora_rank=8,
            lora_alpha=8,
            lora_dropout=0.1,
            batch_size=1,
            gradient_accumulation_steps=16,
            max_seq_length=512,
            use_gradient_checkpointing=True,
            optimizer="adamw_torch",
            model_recommendation="Gemma 2B (4-bit)",
            model_id="unsloth/gemma-2-2b-it-bnb-4bit",
            ollama_model="gemma2:2b",
            notes="Older GPU: Use FP16, smaller model, and aggressive memory savings.",
        )
    
    def to_yaml_dict(self) -> Dict[str, Any]:
        """Convert recommended config to YAML-compatible dict for forge.yaml."""
        config = self.recommend_config()
        
        return {
            "training": {
                "base_model": config.model_id,
                "quantization": config.quantization or "none",
                "max_seq_length": config.max_seq_length,
                "lora": {
                    "rank": config.lora_rank,
                    "alpha": config.lora_alpha,
                    "dropout": config.lora_dropout,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
                "num_epochs": 3,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "learning_rate": 2e-4,
                "weight_decay": 0.01,
                "warmup_ratio": 0.03,
                "lr_scheduler": "linear",
                "use_gradient_checkpointing": config.use_gradient_checkpointing,
                "optim": config.optimizer,
                "save_steps": 100,
                "logging_steps": 10,
            },
            "output": {
                "dir": "./output",
                "checkpoint_dir": "./checkpoints",
                "export_formats": ["lora", "merged"],
            },
        }


def _get_cpu_name() -> str:
    """Get the CPU name."""
    if platform.system() == "Windows":
        try:
            output = subprocess.check_output(
                ["wmic", "cpu", "get", "name"],
                stderr=subprocess.DEVNULL,
            ).decode()
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            if len(lines) > 1:
                return lines[1]
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    
    return platform.processor() or "Unknown CPU"


def detect_gpu() -> Optional[GPUInfo]:
    """Detect GPU and return information."""
    if not TORCH_AVAILABLE:
        return None
    
    if not torch.cuda.is_available():
        return None
    
    try:
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        
        # Get memory info
        total_mem = torch.cuda.get_device_properties(device).total_memory
        vram_total_gb = total_mem / (1024**3)
        
        # Current memory usage
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        vram_used_gb = reserved / (1024**3)
        vram_free_gb = vram_total_gb - vram_used_gb
        
        # CUDA version
        cuda_version = torch.version.cuda or "Unknown"
        
        # Compute capability
        props = torch.cuda.get_device_properties(device)
        compute_capability = (props.major, props.minor)
        
        # Determine architecture
        architecture = get_gpu_architecture(compute_capability)
        
        # Driver version and temperature via pynvml
        driver_version = "Unknown"
        temperature = 0
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                driver_version = pynvml.nvmlSystemGetDriverVersion()
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                
                # More accurate memory info from nvml
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_total_gb = mem_info.total / (1024**3)
                vram_used_gb = mem_info.used / (1024**3)
                vram_free_gb = mem_info.free / (1024**3)
                
                pynvml.nvmlShutdown()
            except Exception:
                pass
        
        return GPUInfo(
            name=gpu_name,
            vram_total_gb=vram_total_gb,
            vram_free_gb=vram_free_gb,
            vram_used_gb=vram_used_gb,
            cuda_version=cuda_version,
            driver_version=driver_version,
            temperature=temperature,
            compute_capability=compute_capability,
            architecture=architecture,
        )
    except Exception:
        return None


def detect_system() -> SystemInfo:
    """Detect system information."""
    mem = psutil.virtual_memory()
    
    return SystemInfo(
        os_name=platform.system(),
        os_version=platform.version(),
        cpu_name=_get_cpu_name(),
        cpu_cores=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=mem.total / (1024**3),
        ram_free_gb=mem.available / (1024**3),
        ram_used_gb=(mem.total - mem.available) / (1024**3),
    )


def detect_hardware() -> HardwareProfile:
    """Detect all hardware and return a complete profile."""
    gpu = detect_gpu()
    system = detect_system()
    
    has_cuda = TORCH_AVAILABLE and torch.cuda.is_available()
    has_rocm = False
    
    # Check for ROCm (AMD GPUs)
    if TORCH_AVAILABLE:
        try:
            has_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        except Exception:
            pass
    
    return HardwareProfile(
        gpu=gpu,
        system=system,
        has_cuda=has_cuda,
        has_rocm=has_rocm,
    )


def get_current_vram_usage() -> tuple:
    """Get current VRAM usage (used_gb, total_gb)."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return (0.0, 0.0)
    
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return (mem_info.used / (1024**3), mem_info.total / (1024**3))
        except Exception:
            pass
    
    # Fallback to torch
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    used = torch.cuda.memory_reserved(device) / (1024**3)
    return (used, total)


def get_gpu_temperature() -> int:
    """Get current GPU temperature in Celsius."""
    if not PYNVML_AVAILABLE:
        return 0
    
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return temp
    except Exception:
        return 0
