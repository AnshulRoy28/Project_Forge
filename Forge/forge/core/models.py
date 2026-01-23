"""
Model catalog for Forge.

Defines supported models organized by size and their HuggingFace IDs for fine-tuning.
Models are selected for Ollama compatibility (can be exported to GGUF format).
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class ModelFamily(Enum):
    """Supported model families."""
    GEMMA = "gemma"
    LLAMA = "llama"
    QWEN = "qwen"
    PHI = "phi"
    SMOLLM = "smollm"
    TINYLLAMA = "tinyllama"
    MISTRAL = "mistral"


class ModelSize(Enum):
    """Model size categories."""
    TINY = "tiny"       # < 500M params
    SMALL = "small"     # 500M - 2B params
    MEDIUM = "medium"   # 2B - 7B params
    LARGE = "large"     # 7B - 13B params


@dataclass
class ModelSpec:
    """Specification for a fine-tunable model."""
    
    id: str                     # HuggingFace model ID
    name: str                   # Human-readable name
    family: ModelFamily
    size: ModelSize
    params_billions: float
    min_vram_gb: float          # Minimum VRAM needed with 4-bit
    recommended_vram_gb: float  # Recommended VRAM
    supports_unsloth: bool      # Optimized Unsloth support
    ollama_name: Optional[str]  # Corresponding Ollama model name
    notes: str
    
    @property
    def unsloth_id(self) -> Optional[str]:
        """Get Unsloth-optimized variant if available."""
        if self.supports_unsloth:
            # Check for pre-quantized Unsloth variants
            unsloth_variants = {
                "google/gemma-2-2b-it": "unsloth/gemma-2-2b-it-bnb-4bit",
                "google/gemma-2-9b-it": "unsloth/gemma-2-9b-it-bnb-4bit",
                "meta-llama/Llama-3.2-1B-Instruct": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
                "meta-llama/Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
                "Qwen/Qwen2.5-1.5B-Instruct": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
                "Qwen/Qwen2.5-3B-Instruct": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                "Qwen/Qwen2.5-7B-Instruct": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
                "microsoft/Phi-3-mini-4k-instruct": "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
            }
            return unsloth_variants.get(self.id)
        return None


# ============================================================================
# MODEL CATALOG
# ============================================================================

MODELS: List[ModelSpec] = [
    # --- TINY Models (< 500M) - Great for testing pipelines ---
    ModelSpec(
        id="HuggingFaceTB/SmolLM-135M-Instruct",
        name="SmolLM 135M",
        family=ModelFamily.SMOLLM,
        size=ModelSize.TINY,
        params_billions=0.135,
        min_vram_gb=1.0,
        recommended_vram_gb=2.0,
        supports_unsloth=False,
        ollama_name="smollm:135m",
        notes="Ultra-small model, great for pipeline testing and edge devices."
    ),
    ModelSpec(
        id="HuggingFaceTB/SmolLM-360M-Instruct",
        name="SmolLM 360M",
        family=ModelFamily.SMOLLM,
        size=ModelSize.TINY,
        params_billions=0.36,
        min_vram_gb=2.0,
        recommended_vram_gb=4.0,
        supports_unsloth=False,
        ollama_name="smollm:360m",
        notes="Small but capable model for lightweight applications."
    ),
    
    # --- SMALL Models (500M - 2B) ---
    ModelSpec(
        id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        name="TinyLlama 1.1B",
        family=ModelFamily.TINYLLAMA,
        size=ModelSize.SMALL,
        params_billions=1.1,
        min_vram_gb=3.0,
        recommended_vram_gb=6.0,
        supports_unsloth=True,
        ollama_name="tinyllama",
        notes="Efficient small model based on Llama 2 architecture."
    ),
    ModelSpec(
        id="meta-llama/Llama-3.2-1B-Instruct",
        name="Llama 3.2 1B",
        family=ModelFamily.LLAMA,
        size=ModelSize.SMALL,
        params_billions=1.0,
        min_vram_gb=3.0,
        recommended_vram_gb=6.0,
        supports_unsloth=True,
        ollama_name="llama3.2:1b",
        notes="Latest Llama architecture, excellent for small deployments."
    ),
    ModelSpec(
        id="Qwen/Qwen2.5-1.5B-Instruct",
        name="Qwen 2.5 1.5B",
        family=ModelFamily.QWEN,
        size=ModelSize.SMALL,
        params_billions=1.5,
        min_vram_gb=4.0,
        recommended_vram_gb=6.0,
        supports_unsloth=True,
        ollama_name="qwen2.5:1.5b",
        notes="Excellent multilingual model, very capable for size."
    ),
    ModelSpec(
        id="HuggingFaceTB/SmolLM-1.7B-Instruct",
        name="SmolLM 1.7B",
        family=ModelFamily.SMOLLM,
        size=ModelSize.SMALL,
        params_billions=1.7,
        min_vram_gb=4.0,
        recommended_vram_gb=6.0,
        supports_unsloth=False,
        ollama_name="smollm:1.7b",
        notes="Largest SmolLM variant, good balance of size and capability."
    ),
    ModelSpec(
        id="google/gemma-2-2b-it",
        name="Gemma 2 2B",
        family=ModelFamily.GEMMA,
        size=ModelSize.SMALL,
        params_billions=2.0,
        min_vram_gb=4.0,
        recommended_vram_gb=8.0,
        supports_unsloth=True,
        ollama_name="gemma2:2b",
        notes="Google's efficient model, excellent instruction following."
    ),
    
    # --- MEDIUM Models (2B - 7B) ---
    ModelSpec(
        id="meta-llama/Llama-3.2-3B-Instruct",
        name="Llama 3.2 3B",
        family=ModelFamily.LLAMA,
        size=ModelSize.MEDIUM,
        params_billions=3.0,
        min_vram_gb=6.0,
        recommended_vram_gb=10.0,
        supports_unsloth=True,
        ollama_name="llama3.2:3b",
        notes="Sweet spot of size and capability for edge deployment."
    ),
    ModelSpec(
        id="Qwen/Qwen2.5-3B-Instruct",
        name="Qwen 2.5 3B",
        family=ModelFamily.QWEN,
        size=ModelSize.MEDIUM,
        params_billions=3.0,
        min_vram_gb=6.0,
        recommended_vram_gb=10.0,
        supports_unsloth=True,
        ollama_name="qwen2.5:3b",
        notes="Strong reasoning and coding capabilities."
    ),
    ModelSpec(
        id="microsoft/Phi-3-mini-4k-instruct",
        name="Phi-3 Mini 3.8B",
        family=ModelFamily.PHI,
        size=ModelSize.MEDIUM,
        params_billions=3.8,
        min_vram_gb=8.0,
        recommended_vram_gb=12.0,
        supports_unsloth=True,
        ollama_name="phi3:mini",
        notes="Microsoft's compact powerhouse, punches above its weight."
    ),
    ModelSpec(
        id="Qwen/Qwen2.5-7B-Instruct",
        name="Qwen 2.5 7B",
        family=ModelFamily.QWEN,
        size=ModelSize.MEDIUM,
        params_billions=7.0,
        min_vram_gb=10.0,
        recommended_vram_gb=16.0,
        supports_unsloth=True,
        ollama_name="qwen2.5:7b",
        notes="Full-featured model with excellent performance."
    ),
    ModelSpec(
        id="mistralai/Mistral-7B-Instruct-v0.3",
        name="Mistral 7B",
        family=ModelFamily.MISTRAL,
        size=ModelSize.MEDIUM,
        params_billions=7.0,
        min_vram_gb=10.0,
        recommended_vram_gb=16.0,
        supports_unsloth=True,
        ollama_name="mistral:7b",
        notes="Excellent general-purpose model with sliding window attention."
    ),
    
    # --- LARGE Models (7B - 13B) ---
    ModelSpec(
        id="google/gemma-2-9b-it",
        name="Gemma 2 9B",
        family=ModelFamily.GEMMA,
        size=ModelSize.LARGE,
        params_billions=9.0,
        min_vram_gb=12.0,
        recommended_vram_gb=24.0,
        supports_unsloth=True,
        ollama_name="gemma2:9b",
        notes="Powerful model for demanding tasks."
    ),
    ModelSpec(
        id="meta-llama/Llama-3.1-8B-Instruct",
        name="Llama 3.1 8B",
        family=ModelFamily.LLAMA,
        size=ModelSize.LARGE,
        params_billions=8.0,
        min_vram_gb=12.0,
        recommended_vram_gb=24.0,
        supports_unsloth=True,
        ollama_name="llama3.1:8b",
        notes="Industry-standard model with broad capabilities."
    ),
]


def get_model_by_id(model_id: str) -> Optional[ModelSpec]:
    """Get model spec by HuggingFace ID."""
    for model in MODELS:
        if model.id == model_id or (model.unsloth_id and model.unsloth_id == model_id):
            return model
    return None


def get_models_for_vram(vram_gb: float, prefer_unsloth: bool = True) -> List[ModelSpec]:
    """Get models that fit within the given VRAM budget."""
    suitable = [m for m in MODELS if m.min_vram_gb <= vram_gb]
    
    if prefer_unsloth:
        # Prioritize models with Unsloth support
        suitable.sort(key=lambda m: (not m.supports_unsloth, -m.params_billions))
    else:
        # Sort by size (larger first)
        suitable.sort(key=lambda m: -m.params_billions)
    
    return suitable


def get_recommended_model(vram_gb: float, prefer_unsloth: bool = True) -> Optional[ModelSpec]:
    """Get the best model for the given VRAM."""
    suitable = get_models_for_vram(vram_gb, prefer_unsloth)
    
    if not suitable:
        return None
    
    # Find the largest model that fits comfortably
    for model in suitable:
        if model.recommended_vram_gb <= vram_gb:
            return model
    
    # Fallback to smallest that fits at all
    return suitable[-1] if suitable else None


def get_models_by_family(family: ModelFamily) -> List[ModelSpec]:
    """Get all models from a specific family."""
    return [m for m in MODELS if m.family == family]


def get_models_by_size(size: ModelSize) -> List[ModelSpec]:
    """Get all models of a specific size category."""
    return [m for m in MODELS if m.size == size]
