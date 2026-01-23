"""
forge plan - Generate model-aware training configuration from natural language.

Enhanced version that creates comprehensive configuration for model-aware preprocessing
and hardware-optimized training.
"""

import re
from pathlib import Path
from typing import Optional
import datetime

import typer
import yaml

from forge.ui.console import console, print_success, print_error, print_info, print_gemini, print_warning
from forge.ui.panels import create_gemini_panel, display_panel
from forge.ui.progress import thinking_spinner
from forge.core.config_v2 import (
    ForgeConfig, ModelConfig, HardwareConfig, PreprocessingConfig,
    ChatTemplate, GPUArchitecture, create_default_config
)
from forge.core.templates import ModelDetector, ChatTemplateRegistry
from forge.core.hardware import detect_hardware, HardwareProfile


def plan_command(
    goal: str = typer.Argument(..., help="Your training goal in natural language"),
    data_path: Optional[Path] = typer.Option(None, "--data", "-d", help="Path to dataset"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output config path"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Force specific model (e.g., llama-7b, gemma-2b)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show config without saving"),
    auto: bool = typer.Option(False, "--auto", help="Skip Gemini, use hardware-optimal defaults"),
):
    """
    Generate a model-aware training configuration from natural language.
    
    This enhanced version creates comprehensive configuration including:
    - Model selection based on hardware and task
    - Chat template detection for preprocessing
    - Hardware-optimized training parameters
    - Preprocessing configuration
    
    Example:
        forge plan "Make a helpful coding assistant using Llama"
        forge plan "Create a customer service chatbot" --data ./data/support.csv
    """
    console.print(f"\n[bold]📝 Planning model-aware training for:[/] \"{goal}\"\n")
    
    # Detect hardware
    with thinking_spinner("Analyzing your hardware..."):
        hardware = detect_hardware()
    
    # Display hardware info
    _display_hardware_info(hardware)
    
    # Get dataset metadata if provided
    dataset_metadata = None
    if data_path and data_path.exists():
        try:
            from forge.cli.study_cmd import _collect_metadata
            dataset_metadata = _collect_metadata(data_path)
            console.print(f"  Dataset: [cyan]{data_path.name}[/] ({dataset_metadata.get('num_samples', 0)} samples)")
            console.print()
        except Exception:
            pass
    
    # Generate model-aware config
    if auto:
        # Use pure hardware-based config without Gemini
        config = _generate_local_config(goal, hardware, data_path, model)
    else:
        # Use Gemini with hardware context for intelligent model selection
        config = _generate_gemini_config(goal, hardware, data_path, dataset_metadata, model)
    
    if not config:
        raise typer.Exit(1)
    
    # Display configuration
    _display_config(config)
    
    # Show model and template info
    _display_model_info(config)
    
    # Save config
    if not dry_run:
        config_path = output or Path("forge.yaml")
        config.save(config_path)
        print_success(f"Configuration saved to: {config_path}")
        
        console.print()
        console.print("[dim]Next steps:[/]")
        console.print("  1. Review the configuration above")
        console.print("  2. Run [cyan]forge prepare ./data/your_data.csv[/] to preprocess data")
        console.print("  3. Run [cyan]forge train[/] to start training")
    else:
        print_info("Dry run - configuration not saved.")
    
    console.print()


def _display_hardware_info(hardware: HardwareProfile):
    """Display detected hardware information."""
    if hardware.gpu:
        gpu = hardware.gpu
        console.print(f"  GPU: [cyan]{gpu.name}[/]")
        console.print(f"  VRAM: [cyan]{gpu.vram_total_gb:.1f} GB[/]")
        console.print(f"  CUDA: [cyan]{gpu.cuda_version}[/]")
        console.print(f"  Compute: [cyan]{gpu.compute_capability[0]}.{gpu.compute_capability[1]}[/] ({gpu.architecture.value})")
        
        if gpu.supports_bf16:
            console.print(f"  BF16: [green]✓ Supported[/]")
        if gpu.supports_fp8:
            console.print(f"  FP8: [green]✓ Supported[/]")
    else:
        console.print("  GPU: [yellow]None detected[/]")
        console.print("  [dim]Training will be CPU-only (very slow)[/]")


def _generate_local_config(
    goal: str, 
    hardware: HardwareProfile, 
    data_path: Optional[Path],
    model_override: Optional[str] = None
) -> Optional[ForgeConfig]:
    """Generate config using local hardware analysis only."""
    
    # Determine model based on goal and hardware
    model_name = _select_optimal_model(goal, hardware, model_override)
    
    # Map hardware to our new GPU architecture enum
    gpu_arch = _map_gpu_architecture(hardware)
    
    # Create configuration using our new system
    config = create_default_config(
        model_name=model_name,
        gpu_arch=gpu_arch,
        vram_gb=hardware.gpu.vram_total_gb if hardware.gpu else 8.0,
        compute_capability=float(f"{hardware.gpu.compute_capability[0]}.{hardware.gpu.compute_capability[1]}") if hardware.gpu else 7.5
    )
    
    # Customize preprocessing based on dataset
    if data_path and data_path.exists():
        # Adjust preprocessing based on file size
        file_size_gb = data_path.stat().st_size / (1024**3)
        if file_size_gb > 2.0:
            config.preprocessing.streaming_threshold_gb = 1.0
            config.preprocessing.chunk_size = 500
        
        # Adjust splits based on dataset size
        try:
            from forge.cli.study_cmd import _collect_metadata
            metadata = _collect_metadata(data_path)
            num_samples = metadata.get('num_samples', 1000)
            
            if num_samples < 100:
                config.preprocessing.validation_split = 0.2  # Larger val split for small datasets
            elif num_samples > 10000:
                config.preprocessing.validation_split = 0.05  # Smaller val split for large datasets
        except:
            pass
    
    return config


def _generate_gemini_config(
    goal: str,
    hardware: HardwareProfile,
    data_path: Optional[Path],
    dataset_metadata: Optional[dict],
    model_override: Optional[str] = None,
) -> Optional[ForgeConfig]:
    """Generate config using Gemini with hardware context."""
    
    # Start with hardware-optimized baseline
    baseline_config = _generate_local_config(goal, hardware, data_path, model_override)
    
    if not baseline_config:
        return None
    
    # Build comprehensive context for Gemini
    hardware_context = {
        "gpu_name": hardware.gpu.name if hardware.gpu else "None",
        "vram_gb": hardware.gpu.vram_total_gb if hardware.gpu else 0,
        "ram_gb": hardware.system.ram_total_gb,
        "architecture": hardware.gpu.architecture.value if hardware.gpu else "cpu",
        "compute_capability": f"{hardware.gpu.compute_capability[0]}.{hardware.gpu.compute_capability[1]}" if hardware.gpu else "0.0",
        "supports_bf16": hardware.gpu.supports_bf16 if hardware.gpu else False,
        "baseline_model": baseline_config.model.name,
        "baseline_template": baseline_config.model.chat_template.value,
        "baseline_batch_size": baseline_config.hardware.recommended_batch_size,
    }
    
    try:
        with thinking_spinner("Gemini is optimizing your model selection and configuration..."):
            from forge.brain.client import create_brain
            
            brain = create_brain()
            
            # Create enhanced prompt for model-aware planning
            prompt = _create_model_aware_prompt(goal, hardware_context, dataset_metadata, baseline_config)
            
            response = brain.reason_sync(prompt)
        
        # Try to extract and apply Gemini's recommendations
        recommendations = _parse_gemini_recommendations(response.text)
        
        if recommendations:
            # Apply Gemini's recommendations to baseline config
            enhanced_config = _apply_recommendations(baseline_config, recommendations)
            
            # Show Gemini's explanation
            explanation = recommendations.get('explanation', '')
            if explanation:
                console.print()
                print_gemini(explanation)
            
            return enhanced_config
        else:
            print_warning("Could not parse Gemini recommendations. Using hardware defaults.")
            return baseline_config
            
    except Exception as e:
        if "API key" in str(e) or "authenticate" in str(e).lower():
            print_error("Gemini API key not configured. Run 'forge init' first.")
            print_info("Using hardware-optimized defaults instead...")
        else:
            print_warning(f"Gemini unavailable: {e}. Using hardware defaults.")
        
        return baseline_config


def _select_optimal_model(goal: str, hardware: HardwareProfile, model_override: Optional[str] = None) -> str:
    """Select optimal model based on goal, hardware, and user preference."""
    
    if model_override:
        # User specified a model, try to find full name
        model_mappings = {
            "llama": "meta-llama/Llama-2-7b-chat-hf",
            "llama-7b": "meta-llama/Llama-2-7b-chat-hf", 
            "llama-13b": "meta-llama/Llama-2-13b-chat-hf",
            "codellama": "codellama/CodeLlama-7b-Instruct-hf",
            "gemma": "google/gemma-7b-it",
            "gemma-2b": "google/gemma-2b-it",
            "gemma-7b": "google/gemma-7b-it",
            "vicuna": "lmsys/vicuna-7b-v1.5",
            "alpaca": "tatsu-lab/alpaca-7b-wdiff",
            "dialogpt": "microsoft/DialoGPT-medium",
        }
        
        return model_mappings.get(model_override.lower(), model_override)
    
    # Auto-select based on hardware and goal
    vram_gb = hardware.gpu.vram_total_gb if hardware.gpu else 8.0
    goal_lower = goal.lower()
    
    # Goal-based selection with better logic
    if any(word in goal_lower for word in ["cod", "program", "develop", "script", "function", "debug"]):
        # Coding tasks - prefer code-specific models
        if vram_gb >= 16:
            return "codellama/CodeLlama-7b-Instruct-hf"
        elif vram_gb >= 12:
            return "google/gemma-7b-it"  # Good general model that can handle code
        else:
            return "google/gemma-2b-it"
    
    elif any(word in goal_lower for word in ["chat", "conversation", "assistant", "help", "support"]):
        # Conversational tasks
        if vram_gb >= 24:
            return "meta-llama/Llama-2-13b-chat-hf"
        elif vram_gb >= 16:
            return "meta-llama/Llama-2-7b-chat-hf"
        elif vram_gb >= 12:
            return "google/gemma-7b-it"
        else:
            return "google/gemma-2b-it"
    
    elif any(word in goal_lower for word in ["instruct", "follow", "task", "command"]):
        # Instruction following
        if vram_gb >= 16:
            return "tatsu-lab/alpaca-7b-wdiff"
        else:
            return "google/gemma-2b-it"
    
    else:
        # General purpose - use Gemma as safe default
        if vram_gb >= 16:
            return "google/gemma-7b-it"
        else:
            return "google/gemma-2b-it"


def _map_gpu_architecture(hardware: HardwareProfile) -> GPUArchitecture:
    """Map hardware GPU architecture to our enum."""
    if not hardware.gpu:
        return GPUArchitecture.BASE
    
    arch_name = hardware.gpu.architecture.value.lower()
    
    if "blackwell" in arch_name:
        return GPUArchitecture.BLACKWELL
    elif "ada" in arch_name or "lovelace" in arch_name:
        return GPUArchitecture.ADA
    elif "ampere" in arch_name:
        return GPUArchitecture.AMPERE
    elif "hopper" in arch_name:
        return GPUArchitecture.HOPPER
    else:
        return GPUArchitecture.BASE


def _create_model_aware_prompt(goal: str, hardware_context: dict, dataset_metadata: Optional[dict], baseline_config: ForgeConfig) -> str:
    """Create enhanced prompt for Gemini model selection and optimization."""
    
    prompt = f"""You are an expert ML engineer helping optimize model selection and training configuration.

GOAL: {goal}

HARDWARE CONTEXT:
- GPU: {hardware_context['gpu_name']} ({hardware_context['vram_gb']:.1f} GB VRAM)
- Architecture: {hardware_context['architecture']}
- Compute Capability: {hardware_context['compute_capability']}
- BF16 Support: {hardware_context['supports_bf16']}

BASELINE CONFIGURATION (HARDWARE-OPTIMIZED):
- Model: {baseline_config.model.name}
- Chat Template: {baseline_config.model.chat_template.value}
- Batch Size: {baseline_config.hardware.recommended_batch_size} (OPTIMIZED FOR MAXIMUM SPEED - DO NOT REDUCE)
- Max Length: {baseline_config.model.max_length}

DATASET INFO:
{f"- Samples: {dataset_metadata.get('num_samples', 'unknown')}" if dataset_metadata else "- No dataset provided"}
{f"- Format: {dataset_metadata.get('format', 'unknown')}" if dataset_metadata else ""}
{f"- Columns: {dataset_metadata.get('columns', [])}" if dataset_metadata else ""}

IMPORTANT: The baseline batch size ({baseline_config.hardware.recommended_batch_size}) is already optimized for this high-end GPU to maximize training speed. 
DO NOT recommend reducing it unless there's a specific technical reason (like model architecture limitations).

Please analyze this setup and provide recommendations for:

1. MODEL SELECTION: Is the baseline model optimal for this goal and hardware? Consider:
   - Task suitability (coding, chat, instruction-following, etc.)
   - Hardware constraints (VRAM, compute capability)
   - Model size vs performance tradeoffs

2. CHAT TEMPLATE: Is the detected template appropriate for the selected model?

3. TRAINING OPTIMIZATIONS: Any adjustments to:
   - Batch size (only increase or keep current hardware-optimized value)
   - Sequence length
   - Quantization settings
   - LoRA parameters

4. PREPROCESSING OPTIMIZATIONS: Based on the dataset characteristics:
   - Train/validation split ratios
   - Text length limits
   - Quality filtering settings

Respond with your analysis and specific recommendations. If the baseline is already optimal, explain why.
Focus on practical improvements that will meaningfully impact training quality or efficiency.
REMEMBER: Keep the aggressive batch size for maximum training speed on this high-end hardware.
"""
    
    return prompt


def _parse_gemini_recommendations(response_text: str) -> Optional[dict]:
    """Parse Gemini's recommendations from response text."""
    # This is a simplified parser - in practice, you'd want more robust parsing
    recommendations = {}
    
    # Look for specific recommendation patterns
    lines = response_text.lower().split('\n')
    
    for line in lines:
        if 'model:' in line and 'recommend' in line:
            # Extract model recommendation
            model_match = re.search(r'model[:\s]+([a-zA-Z0-9/_-]+)', line)
            if model_match:
                recommendations['model'] = model_match.group(1)
        
        elif 'batch' in line and ('size' in line or 'recommend' in line):
            # Extract batch size recommendation
            batch_match = re.search(r'(\d+)', line)
            if batch_match:
                recommendations['batch_size'] = int(batch_match.group(1))
        
        elif 'template' in line and ('recommend' in line or 'use' in line):
            # Extract template recommendation
            for template in ['llama', 'gemma', 'chatml', 'alpaca', 'vicuna']:
                if template in line:
                    recommendations['template'] = template
                    break
    
    # Extract explanation (first paragraph)
    paragraphs = response_text.split('\n\n')
    if paragraphs:
        recommendations['explanation'] = paragraphs[0].strip()
    
    return recommendations if recommendations else None


def _apply_recommendations(baseline_config: ForgeConfig, recommendations: dict) -> ForgeConfig:
    """Apply Gemini's recommendations to the baseline configuration."""
    
    # Apply model recommendation
    if 'model' in recommendations:
        baseline_config.model.name = recommendations['model']
        # Update template based on new model
        new_template = ModelDetector.detect_template(recommendations['model'])
        baseline_config.model.chat_template = new_template
    
    # Apply template recommendation
    if 'template' in recommendations:
        try:
            baseline_config.model.chat_template = ChatTemplate(recommendations['template'])
        except ValueError:
            pass  # Invalid template, keep baseline
    
    # Apply batch size recommendation - but never go below hardware-optimized minimum
    if 'batch_size' in recommendations:
        recommended_batch = recommendations['batch_size']
        hardware_optimized = baseline_config.hardware.recommended_batch_size
        
        # Use the larger of Gemini's recommendation or hardware optimization
        # This ensures we never use conservative batch sizes on high-end hardware
        final_batch_size = max(recommended_batch, hardware_optimized)
        
        if final_batch_size > recommended_batch:
            print_info(f"Using hardware-optimized batch size {final_batch_size} instead of Gemini's {recommended_batch}")
        
        baseline_config.hardware.recommended_batch_size = final_batch_size
    
    return baseline_config


def _display_model_info(config: ForgeConfig):
    """Display model and template information."""
    console.print(f"\n[bold]🤖 Model Configuration:[/]")
    console.print(f"  Model: [cyan]{config.model.name}[/]")
    console.print(f"  Architecture: [cyan]{config.model.architecture}[/]")
    console.print(f"  Chat Template: [cyan]{config.model.chat_template.value}[/]")
    console.print(f"  Max Length: [cyan]{config.model.max_length}[/]")
    
    # Show template example
    template_config = ChatTemplateRegistry.get_template(config.model.chat_template)
    console.print(f"  Template Format: [dim]{template_config['description']}[/]")
    
    console.print(f"\n[bold]⚙️ Hardware Optimization:[/]")
    console.print(f"  GPU Architecture: [cyan]{config.hardware.gpu_arch.value}[/]")
    console.print(f"  VRAM: [cyan]{config.hardware.vram_gb:.1f} GB[/]")
    console.print(f"  Recommended Batch Size: [cyan]{config.hardware.recommended_batch_size}[/]")
    console.print(f"  GPU Preprocessing: [cyan]{'Enabled' if config.hardware.use_gpu_preprocessing else 'Disabled'}[/]")
    
    console.print(f"\n[bold]📊 Preprocessing Settings:[/]")
    console.print(f"  Train/Val Split: [cyan]{config.preprocessing.train_split:.1%}/{config.preprocessing.validation_split:.1%}[/]")
    console.print(f"  Chunk Size: [cyan]{config.preprocessing.chunk_size}[/]")
    console.print(f"  Quality Checks: [cyan]{'Enabled' if config.preprocessing.quality_checks else 'Disabled'}[/]")
    console.print(f"  Streaming Threshold: [cyan]{config.preprocessing.streaming_threshold_gb:.1f} GB[/]")


def _display_config(config: ForgeConfig):
    """Display the generated configuration."""
    from rich.syntax import Syntax
    from rich.panel import Panel
    
    console.print("\n[bold green]✨ Generated Configuration:[/]\n")
    
    # Convert to dictionary for YAML display
    config_dict = config.to_dict()
    
    clean_yaml = yaml.safe_dump(
        config_dict, 
        default_flow_style=False, 
        sort_keys=False,
        indent=2
    )
    syntax = Syntax(clean_yaml, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, border_style="green"))


def _slugify(text: str) -> str:
    """Convert text to a slug for config name."""
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[\s_-]+', '-', slug)
    return slug.strip('-')


def _extract_yaml(text: str) -> Optional[str]:
    """Extract YAML from markdown code blocks or raw text."""
    # Try to find YAML in code blocks
    yaml_pattern = r"```(?:yaml)?\\s*\\n(.*?)\\n```"
    matches = re.findall(yaml_pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Try simpler pattern
    yaml_pattern2 = r"```(?:yaml)?\s*(.*?)```"
    matches = re.findall(yaml_pattern2, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Try to find YAML-like content (starts with common keys)
    lines = text.split("\n")
    yaml_lines = []
    in_yaml = False
    
    for line in lines:
        if line.strip().startswith(("name:", "training:", "goal:", "data:", "output:")):
            in_yaml = True
        
        if in_yaml:
            # Stop at explanation text
            if line.strip() and not line.startswith(" ") and ":" not in line and not line.startswith("-"):
                break
            yaml_lines.append(line)
    
    if yaml_lines:
        return "\n".join(yaml_lines).strip()
    
    return None


def _extract_explanation(text: str) -> Optional[str]:
    """Extract explanation text after YAML block."""
    # Find end of YAML block
    parts = re.split(r"```\s*\n", text)
    
    if len(parts) > 1:
        # Get text after last code block
        after_yaml = parts[-1].strip()
        if after_yaml and len(after_yaml) > 20:
            return after_yaml
    
    return None
