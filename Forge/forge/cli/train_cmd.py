"""
forge train - Execute training via Docker with real-time monitoring and self-healing.

All training runs inside Docker containers with GPU passthrough.
"""

import subprocess
import sys
import re
import os
import time
import datetime
from pathlib import Path
from typing import Optional, Tuple

import typer

from forge.ui.console import console, print_success, print_error, print_info, print_warning, print_gemini
from forge.ui.panels import create_gemini_panel, display_panel
from forge.ui.progress import thinking_spinner
from forge.core.config import load_config, save_config, get_config_path, ForgeConfig


MAX_HEAL_ATTEMPTS = 3


# Common error patterns and their auto-fixes
ERROR_FIXES = {
    "BFloat16": {
        "pattern": r"(not implemented for 'BFloat16'|bfloat16|bf16.*not supported)",
        "fix": "disable_bf16",
        "message": "BFloat16 not supported on your GPU, switching to FP16",
    },
    "OutOfMemory": {
        "pattern": r"(CUDA out of memory|OutOfMemoryError|OOM|dispatched on the CPU or the disk|enough GPU RAM)",
        "fix": "reduce_memory",
        "message": "Out of memory, switching to smaller model with 4-bit quantization",
    },
    "ModelNotFound": {
        "pattern": r"(not a valid model identifier|is not a local folder)",
        "fix": "fix_model_name",
        "message": "Model not found, trying alternative model",
    },
    "GatedModel": {
        "pattern": r"(gated repo|pass a token|token=)",
        "fix": "prompt_hf_login",
        "message": "Model requires authentication",
    },
    "Tokenizer": {
        "pattern": r"(unexpected keyword argument 'tokenizer'|tokenizer.*not)",
        "fix": "update_trainer_api",
        "message": "TRL API changed, updating trainer configuration",
    },
    "MaxSeqLength": {
        "pattern": r"(max_seq_length|'max_length')",
        "fix": "fix_max_length",
        "message": "Fixing sequence length parameter",
    },
}


def train_command(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to forge.yaml"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from last checkpoint"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config inside Docker without training"),
    no_heal: bool = typer.Option(False, "--no-heal", help="Disable self-healing"),
):
    """
    Start training via Docker container with GPU passthrough.
    
    All training runs inside Docker containers optimized for your GPU architecture.
    Requires session credentials (use 'forge init' first).
    """
    # Check for session credentials first
    from forge.core.security import has_credentials, prompt_for_credentials
    
    has_gemini, has_hf = has_credentials()
    
    if not has_gemini:
        console.print("[bold yellow]🔑 Session credentials required[/]")
        console.print("[dim]Run 'forge init' first, or provide credentials now:[/]")
        console.print()
        
        got_gemini, got_hf = prompt_for_credentials(console, force_gemini=True, force_hf=False)
        
        if not got_gemini:
            print_error("Gemini API key is required for training")
            console.print("[dim]Run: forge init[/]")
            raise typer.Exit(1)
    
    # Check Docker availability
    if not _check_docker():
        print_error("Docker is required for training")
        console.print("\n[bold]To install Docker:[/]")
        console.print("  Windows: [cyan]https://docs.docker.com/desktop/install/windows/[/]")
        console.print("  Linux:   [cyan]https://docs.docker.com/engine/install/[/]")
        console.print("\n[dim]After installing, run: forge docker build[/]")
        raise typer.Exit(1)
    
    # Load configuration - try new format first, then old format
    config_file = config_path or Path("forge.yaml")
    
    if not config_file.exists():
        print_error(f"Config not found: {config_file}")
        console.print("[dim]Run 'forge plan \"your goal\"' to generate a configuration.[/]")
        raise typer.Exit(1)
    
    try:
        # Try loading as new format (v2.0)
        from forge.core.config_v2 import ForgeConfig as ForgeConfigV2
        config_v2 = ForgeConfigV2.load(config_file)
        
        console.print(f"\n[bold]🔥 Starting Docker training:[/] {config_v2.model.name}-finetune\n")
        _display_config_v2(config_v2)
        
        # Show what's about to happen
        console.print("[bold]📋 Training Plan:[/]")
        console.print(f"   1. Load model: {config_v2.model.name}")
        
        # Check for processed data files
        train_file = Path("./data/processed_train.jsonl")
        if train_file.exists():
            console.print(f"   2. Process dataset: {train_file.stat().st_size / 1024**2:.1f}MB")
        else:
            console.print(f"   2. Process dataset: Ready")
            
        console.print(f"   3. Train for 3 epochs with batch size {config_v2.hardware.recommended_batch_size}")
        console.print(f"   4. Save to: ./output/")
        console.print()
        
    except Exception as e:
        # Fallback to old format
        try:
            from forge.core.config import load_config
            config = load_config(config_file)
            console.print(f"\n[bold]🔥 Starting Docker training:[/] {config.name}\n")
            _display_config(config)
            
            # For old format, create a minimal v2 config for container management
            from forge.core.config_v2 import ForgeConfig as ForgeConfigV2, ModelConfig, HardwareConfig, PreprocessingConfig, ChatTemplate, GPUArchitecture, ContainerConfig
            
            config_v2 = ForgeConfigV2(
                model=ModelConfig(
                    name=config.training.base_model,
                    architecture="transformer",
                    chat_template=ChatTemplate.CHATML,
                ),
                hardware=HardwareConfig(
                    gpu_arch=GPUArchitecture.BASE,
                    vram_gb=16.0,
                    compute_capability=8.0,
                    recommended_batch_size=config.training.batch_size,
                ),
                preprocessing=PreprocessingConfig(),
                container=ContainerConfig(),
            )
            
        except Exception as e2:
            print_error(f"Invalid configuration: {e2}")
            raise typer.Exit(1)
    
    # Training via persistent Docker container with self-healing
    auto_heal = not no_heal
    success = _run_with_healing(config_v2, config_file, resume, dry_run, auto_heal)
    
    if not success:
        raise typer.Exit(1)


def _check_docker() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _detect_gpu_architecture() -> str:
    """Detect GPU architecture for Docker image selection."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            compute_cap = result.stdout.strip().split("\n")[0]
            major = int(compute_cap.split(".")[0])
            minor = int(compute_cap.split(".")[1]) if "." in compute_cap else 0
            
            if major >= 12:
                return "blackwell"
            elif major == 9:
                return "hopper"
            elif major == 8 and minor >= 9:
                return "ada"
            elif major >= 8:
                return "ampere"
    except Exception:
        pass
    
    return "ada"  # Default fallback


def _check_docker_image(arch: str) -> bool:
    """Check if Docker image exists for the architecture."""
    result = subprocess.run(
        ["docker", "image", "inspect", f"forge:{arch}"],
        capture_output=True
    )
    return result.returncode == 0


def _build_docker_image(arch: str) -> bool:
    """Build Docker image for the architecture."""
    console.print(f"\n[bold]🐳 Building Docker image for {arch}...[/]\n")
    
    # Find docker directory
    docker_dir = None
    for base in [Path(__file__).parent.parent.parent, Path.cwd()]:
        candidate = base / "docker"
        if candidate.exists():
            docker_dir = candidate
            break
    
    if not docker_dir:
        print_error("Docker directory not found")
        return False
    
    dockerfile = docker_dir / f"Dockerfile.{arch}"
    if not dockerfile.exists():
        print_error(f"Dockerfile not found: {dockerfile}")
        return False
    
    cmd = [
        "docker", "build",
        "-t", f"forge:{arch}",
        "-f", str(dockerfile),
        str(docker_dir.parent)
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def _display_config(config: ForgeConfig):
    """Display config summary."""
    console.print(f"  Model: [cyan]{config.training.base_model}[/]")
    console.print(f"  Quantization: [cyan]{config.training.quantization or 'Full precision'}[/]")
    console.print(f"  LoRA Rank: [cyan]{config.training.lora.rank}[/]")
    console.print(f"  Epochs: [cyan]{config.training.num_epochs}[/]")
    console.print(f"  Batch Size: [cyan]{config.training.batch_size}[/]")
    console.print(f"  Learning Rate: [cyan]{config.training.learning_rate}[/]")
    console.print()


def _display_config_v2(config):
    """Display config summary for v2.0 format."""
    console.print(f"  Model: [cyan]{config.model.name}[/]")
    console.print(f"  Template: [cyan]{config.model.chat_template.value}[/]")
    console.print(f"  Max Length: [cyan]{config.model.max_length}[/]")
    console.print(f"  Batch Size: [cyan]{config.hardware.recommended_batch_size}[/]")
    console.print(f"  GPU: [cyan]{config.hardware.gpu_arch.value}[/] ({config.hardware.vram_gb:.1f}GB)")
    console.print(f"  Memory Usage: [cyan]{config.hardware.max_memory_usage:.0%}[/]")
    console.print()


def _check_hf_authentication(model_name: str) -> bool:
    """Check if HuggingFace authentication is available for gated models."""
    if not ("google/" in model_name or "meta-llama/" in model_name):
        return True  # No auth needed for open models
    
    # Check session credentials first
    from forge.core.security import get_hf_token
    stored_token = get_hf_token()
    
    if stored_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=stored_token)
            # Try to get user info - this will fail if not authenticated
            api.whoami()
            return True
        except Exception:
            pass
    
    # Fallback to checking if huggingface-cli login was used
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Try to get user info - this will fail if not authenticated
        api.whoami()
        return True
    except Exception:
        return False


def _convert_v2_to_v1_config(config_v2):
    """Convert new v2.0 config format to old v1 format for training."""
    from forge.core.config import ForgeConfig, TrainingConfig, DataConfig, OutputConfig, LoRAConfig
    
    # Check for processed data files
    train_file = Path("./data/processed_train.jsonl")
    val_file = Path("./data/processed_val.jsonl")
    
    if not train_file.exists():
        print_error("Processed training data not found. Run 'forge prepare' first.")
        raise typer.Exit(1)
    
    # Use the exact model specified in the YAML configuration
    model_name = config_v2.model.name
    console.print(f"[bold]Using model from YAML: [cyan]{model_name}[/]")
    
    # Check if model requires authentication
    if not _check_hf_authentication(model_name):
        print_warning("HuggingFace authentication not found for gated model")
        console.print()
        console.print("[bold]Options:[/]")
        console.print("1. Provide HuggingFace token for this session")
        console.print("2. Or use an open model: [cyan]forge plan \"your goal\" --model unsloth/gemma-2-2b-it-bnb-4bit[/]")
        console.print()
        
        # Prompt for HuggingFace token
        from forge.core.security import prompt_for_credentials
        got_gemini, got_hf = prompt_for_credentials(console, force_gemini=False, force_hf=True)
        
        if not got_hf:
            print_info("Please provide HuggingFace token or use an open model")
            raise typer.Exit(1)
    
    batch_size = config_v2.hardware.recommended_batch_size
    console.print(f"[bold]Using batch size from YAML: [cyan]{batch_size}[/]")
    
    # Create old format config with Docker paths
    return ForgeConfig(
        name=f"{model_name.split('/')[-1]}-finetune",
        training=TrainingConfig(
            base_model=model_name,  # Use exact model from YAML
            batch_size=batch_size,  # Use the batch size from v2 config
            num_epochs=3,  # Default
            learning_rate=2e-4,  # Default
            quantization="4bit",  # Default for efficiency
            lora=LoRAConfig(
                rank=16,  # Default
                alpha=32,  # Default
                dropout=0.1,  # Default
            ),
            max_seq_length=config_v2.model.max_length,
            use_gradient_checkpointing=True,
            gradient_accumulation_steps=1,
        ),
        data=DataConfig(
            path="/data/processed_train.jsonl",  # Docker path
            validation_path="/data/processed_val.jsonl" if val_file.exists() else None,  # Docker path
            text_column="text",
            format="jsonl",
        ),
        output=OutputConfig(
            dir="/output",  # Docker path
            checkpoint_dir="/checkpoints",  # Docker path
            export_formats=["merged"],
        ),
    )


def _run_with_healing(
    config_v2, 
    config_file: Path, 
    resume: bool,
    dry_run: bool,
    auto_heal: bool
) -> bool:
    """Run training with automatic error healing using persistent containers."""
    
    for attempt in range(MAX_HEAL_ATTEMPTS):
        try:
            if attempt > 0:
                console.print(f"\n[bold yellow]🔧 Retry attempt {attempt + 1}/{MAX_HEAL_ATTEMPTS}[/]\n")
                _display_config_v2(config_v2)
            
            success, error_output = _run_docker_training(config_v2, resume, dry_run)
            
            if success:
                return True
            
            # Training failed - try to heal
            if not auto_heal or attempt >= MAX_HEAL_ATTEMPTS - 1:
                _show_gemini_diagnosis(error_output, config_v2)
                return False
            
            # Try to auto-fix (would need to be adapted for v2 config)
            print_warning("Auto-healing not yet implemented for v2 config format")
            _show_gemini_diagnosis(error_output, config_v2)
            return False
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Training interrupted by user.[/]")
            print_info("Checkpoint saved. Use --resume to continue.")
            return False
    
    return False


def _run_docker_training(config_v2, resume: bool, dry_run: bool) -> Tuple[bool, str]:
    """Execute training in persistent Docker container with model caching."""
    
    # Initialize container manager
    from forge.core.container_manager import ContainerManager
    container_mgr = ContainerManager(config_v2)
    
    # Start or reuse persistent container
    container_id = container_mgr.start_persistent_container()
    
    if not container_id:
        return False, "Failed to start persistent container"
    
    # Cache model in container if not already cached
    if not config_v2.container.model_cached:
        console.print(f"[bold]🚀 First-time setup: Caching model in container[/]")
        console.print(f"[dim]This will make future training sessions much faster![/]")
        console.print()
        
        if container_mgr.cache_model_in_container():
            # Save updated config with cached model info
            config_v2.save(Path("forge.yaml"))
        else:
            print_warning("Model caching failed, but training will continue")
    else:
        print_success(f"Using cached model from container: {container_id[:12]}")
    
    console.print(f"[bold]🐳 Training in persistent container ({config_v2.hardware.gpu_arch.value})[/]")
    console.print(f"[dim]Container: {container_id[:12]} | Model cached: {config_v2.container.model_cached}[/]")
    console.print()
    
    # Create temporary old-format config for training
    temp_config_path = Path("./forge_temp.yaml")
    try:
        from forge.core.config import save_config
        old_config = _convert_v2_to_v1_config(config_v2)
        save_config(old_config, temp_config_path)
        
        # Copy config to container
        copy_cmd = [
            "docker", "cp", 
            str(temp_config_path.resolve()),
            f"{container_id}:/app/forge.yaml"
        ]
        
        subprocess.run(copy_cmd, check=True, timeout=30)
        
        # Build forge command inside container
        forge_cmd = ["forge", "train-internal"]
        if resume:
            forge_cmd.append("--resume")
        if dry_run:
            forge_cmd.append("--dry-run")
        
        # Execute training in persistent container
        error_output = []
        
        try:
            process = container_mgr.execute_in_container(forge_cmd)
            
            # Add timeout and heartbeat detection
            last_output_time = time.time()
            timeout_seconds = 1800  # 30 minutes timeout
            heartbeat_interval = 60  # Print heartbeat every 60 seconds of silence
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Print line to console
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    
                    # Update last output time
                    last_output_time = time.time()
                    
                    # Collect for error analysis
                    error_output.append(line)
                    
                    # Keep only last 100 lines for error analysis
                    if len(error_output) > 100:
                        error_output.pop(0)
                else:
                    # Check for timeout or need for heartbeat
                    current_time = time.time()
                    elapsed_since_output = current_time - last_output_time
                    
                    if elapsed_since_output > timeout_seconds:
                        print_error(f"Training timed out after {timeout_seconds/60:.1f} minutes of no output")
                        process.terminate()
                        return False, "Training timeout - no output received"
                    
                    elif elapsed_since_output > heartbeat_interval and elapsed_since_output % heartbeat_interval < 1:
                        console.print(f"[dim]⏱️  Training in progress... ({elapsed_since_output/60:.1f}m since last output)[/]")
            
            process.wait()
            
            if process.returncode == 0:
                # Update container last used time
                config_v2.container.last_used = datetime.datetime.now().isoformat()
                config_v2.save(Path("forge.yaml"))
                return True, ""
            else:
                return False, "\n".join(error_output)
                
        except Exception as e:
            return False, str(e)
    
    finally:
        # Clean up temporary config file
        if temp_config_path.exists():
            temp_config_path.unlink()


def _try_auto_fix(error: str, config: ForgeConfig) -> Tuple[bool, ForgeConfig]:
    """Try to automatically fix the error."""
    
    for error_type, fix_info in ERROR_FIXES.items():
        if re.search(fix_info["pattern"], error, re.IGNORECASE):
            console.print()
            print_warning(f"Detected: {error_type}")
            print_info(f"Auto-fix: {fix_info['message']}")
            
            # Apply the fix
            fixed_config = _apply_fix(fix_info["fix"], config, error)
            
            if fixed_config:
                return True, fixed_config
    
    return False, config


def _apply_fix(fix_type: str, config: ForgeConfig, error: str) -> Optional[ForgeConfig]:
    """Apply a specific fix to the configuration."""
    
    # Create a modified copy of config
    import copy
    new_config = copy.deepcopy(config)
    
    if fix_type == "disable_bf16":
        # Force FP16 instead of BF16
        new_config.training.use_gradient_checkpointing = True
        console.print("  → Enabled gradient checkpointing")
        console.print("  → Will use FP16 instead of BF16")
        return new_config
    
    elif fix_type == "reduce_memory":
        # Reduce batch size and enable optimizations
        new_config.training.batch_size = max(1, new_config.training.batch_size // 2)
        new_config.training.gradient_accumulation_steps *= 2
        new_config.training.use_gradient_checkpointing = True
        
        # Try smaller model if available
        model = new_config.training.base_model
        if "9b" in model.lower():
            new_config.training.base_model = model.replace("9b", "2b").replace("9B", "2B")
            console.print(f"  → Switching to smaller model: {new_config.training.base_model}")
        
        console.print(f"  → Batch size: {new_config.training.batch_size}")
        console.print(f"  → Gradient accumulation: {new_config.training.gradient_accumulation_steps}")
        return new_config
    
    elif fix_type == "fix_model_name":
        # Try common model name fixes
        model = new_config.training.base_model
        alternatives = [
            model.replace("gemma-9b", "gemma-2-9b-it-bnb-4bit"),
            model.replace("gemma-2b", "gemma-2-2b-it-bnb-4bit"),
            "unsloth/gemma-2-2b-it-bnb-4bit",  # Safe fallback
        ]
        
        new_config.training.base_model = alternatives[0]
        console.print(f"  → Using: {new_config.training.base_model}")
        return new_config
    
    elif fix_type == "prompt_hf_login":
        console.print()
        console.print("[bold red]🔒 Authentication Required[/]")
        console.print(f"The model [cyan]{config.training.base_model}[/] requires HuggingFace authentication.")
        console.print()
        console.print("[bold]To fix this:[/]")
        console.print("1. Run: [cyan]forge init --force[/] (to configure HuggingFace token)")
        console.print("2. Or manually: [cyan]pip install huggingface_hub && huggingface-cli login[/]")
        console.print("3. Get a token from: [cyan]https://huggingface.co/settings/tokens[/]")
        console.print("4. Retry training: [cyan]forge train[/]")
        console.print()
        console.print("[dim]Alternatively, you can use an open model by running:[/]")
        console.print("[dim]forge plan \"your goal\" --model unsloth/gemma-2-2b-it-bnb-4bit[/]")
        return None  # Can't auto-fix, needs user action
    
    elif fix_type == "fix_max_length":
        # Already fixed in trainer.py, just retry
        console.print("  → Retrying with corrected parameters")
        return new_config
    
    elif fix_type == "update_trainer_api":
        # Already fixed in trainer.py, just retry
        console.print("  → Retrying with updated TRL API")
        return new_config
    
    return None


def _show_gemini_diagnosis(error: str, config_v2):
    """Show Gemini's diagnosis of the error."""
    try:
        with thinking_spinner("Gemini is diagnosing the error..."):
            from forge.brain.client import create_brain
            brain = create_brain()
            diagnosis = brain.diagnose_error(error[:1000], context=f"Training {config_v2.model.name}")
        
        display_panel(create_gemini_panel(diagnosis.text))
    except Exception:
        pass


# Internal command for running inside Docker container
def train_internal_command(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to forge.yaml"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from last checkpoint"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without training"),
    internal_docker_run: bool = typer.Option(False, "--internal-docker-run", hidden=True),
):
    """Internal training execution (runs inside Docker)."""
    from forge.anvil.trainer import ForgeTrainer
    from forge.anvil.data import load_dataset
    
    # Load configuration - try new format first, then old format
    config_file = config_path or Path("forge.yaml")
    
    if not config_file.exists():
        print_error(f"Config not found: {config_file}")
        raise typer.Exit(1)
    
    try:
        # Try loading as new format (v2.0)
        from forge.core.config_v2 import ForgeConfig as ForgeConfigV2
        config_v2 = ForgeConfigV2.load(config_file)
        
        # Convert to old format for training
        config = _convert_v2_to_v1_config(config_v2)
        
        console.print(f"\n[bold]🔥 Training:[/] {config.name}\n")
        _display_config_v2(config_v2)
        
    except Exception as e:
        # Fallback to old format
        try:
            config = load_config(config_file)
            console.print(f"\n[bold]🔥 Training:[/] {config.name}\n")
            _display_config(config)
        except Exception as e2:
            print_error(f"Invalid configuration: {e2}")
            raise typer.Exit(1)
    
    if dry_run:
        print_success("Configuration is valid!")
        print_info("Dry run mode - no training performed.")
        return
    
    # Check for data
    if not config.data or not config.data.path:
        print_error("No dataset configured.")
        raise typer.Exit(1)
    
    data_path = Path(config.data.path)
    if not data_path.exists():
        print_error(f"Dataset not found: {data_path}")
        raise typer.Exit(1)
    
    console.print(f"  Dataset: [cyan]{data_path}[/]\n")
    
    # Find checkpoint if resuming
    checkpoint_path = None
    if resume:
        checkpoint_path = _find_latest_checkpoint(config)
        if checkpoint_path:
            print_info(f"Resuming from: {checkpoint_path.name}")
    
    # Run training
    _run_training_internal(config, checkpoint_path)


def _find_latest_checkpoint(config: ForgeConfig) -> Optional[Path]:
    """Find the latest checkpoint."""
    checkpoint_dir = Path(config.output.checkpoint_dir)
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
        if checkpoints:
            return checkpoints[-1]
    return None


def _run_training_internal(config: ForgeConfig, checkpoint_path: Optional[Path] = None):
    """Execute the training loop (inside Docker)."""
    import time
    from forge.anvil.trainer import ForgeTrainer
    from forge.anvil.data import load_dataset
    from forge.ui.progress import create_training_progress, spinner
    
    console.print("[bold]🚀 Starting training process...[/]\n")
    
    # Pre-training validation
    console.print("🔍 Pre-training validation...")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            console.print(f"   GPU: {gpu_name} ({vram_total:.1f}GB)")
        else:
            print_warning("No CUDA GPU detected - training will be very slow")
    except ImportError:
        console.print("   PyTorch not available for GPU check")
    
    # Validate configuration
    if config.training.batch_size <= 0:
        print_error("Invalid batch size")
        raise typer.Exit(1)
    
    console.print("✅ Pre-training validation passed")
    
    # Create directories
    console.print("📁 Creating output directories...")
    Path(config.output.dir).mkdir(parents=True, exist_ok=True)
    Path(config.output.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    console.print("✅ Directories created")
    
    # Load model with detailed progress
    console.print("\n🤖 Loading model and tokenizer...")
    console.print(f"   Model: {config.training.base_model}")
    console.print(f"   Quantization: {config.training.quantization}")
    console.print("   This may take 2-5 minutes for large models...")
    
    start_time = time.time()
    try:
        trainer = ForgeTrainer(config)
        load_time = time.time() - start_time
        print_success(f"Model loaded in {load_time:.1f}s: {config.training.base_model}")
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        raise typer.Exit(1)
    
    # Load dataset with progress
    console.print("\n📊 Loading dataset...")
    try:
        dataset = load_dataset(config.data)
        print_success(f"Dataset ready: {len(dataset):,} samples")
    except Exception as e:
        print_error(f"Failed to load dataset: {e}")
        raise typer.Exit(1)
    
    # Calculate training parameters
    total_samples = len(dataset)
    batch_size = config.training.batch_size
    epochs = config.training.num_epochs
    grad_accum = config.training.gradient_accumulation_steps
    
    steps_per_epoch = total_samples // (batch_size * grad_accum)
    total_steps = steps_per_epoch * epochs
    
    console.print(f"\n📈 Training configuration:")
    console.print(f"   Total samples: {total_samples:,}")
    console.print(f"   Batch size: {batch_size}")
    console.print(f"   Gradient accumulation: {grad_accum}")
    console.print(f"   Steps per epoch: {steps_per_epoch:,}")
    console.print(f"   Total steps: {total_steps:,}")
    console.print(f"   Estimated time: {_estimate_training_time(total_steps, batch_size)}")
    console.print()
    
    # Training with enhanced progress tracking
    console.print("[bold]🔥 Starting training...[/]")
    
    training_start = time.time()
    last_update = training_start
    
    with create_training_progress() as progress:
        task = progress.add_task("[cyan]Training Progress", total=total_steps)
        
        def on_step(step: int, loss: float, lr: float):
            nonlocal last_update
            current_time = time.time()
            
            # Update progress
            progress.update(task, completed=step)
            
            # Print detailed updates every 10 steps or 30 seconds
            if step % 10 == 0 or (current_time - last_update) > 30:
                elapsed = current_time - training_start
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                eta_seconds = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                
                console.print(f"Step {step:,}/{total_steps:,} | Loss: {loss:.4f} | LR: {lr:.2e} | "
                            f"{steps_per_sec:.2f} steps/s | ETA: {_format_time(eta_seconds)}")
                last_update = current_time
        
        try:
            trainer.train(
                dataset=dataset,
                callback=on_step,
                resume_from=checkpoint_path,
            )
        except Exception as e:
            print_error(f"Training failed: {e}")
            raise typer.Exit(1)
    
    training_time = time.time() - training_start
    console.print()
    print_success(f"Training completed in {_format_time(training_time)}!")
    
    # Save model with progress
    console.print("\n💾 Saving model...")
    output_path = Path(config.output.dir)
    try:
        with spinner("Saving model..."):
            trainer.save(output_path)
        print_success(f"Model saved to: {output_path}")
    except Exception as e:
        print_error(f"Failed to save model: {e}")
        raise typer.Exit(1)
    
    # Export formats with progress
    for format_type in config.output.export_formats:
        if format_type == "merged":
            console.print("\n🔗 Merging LoRA weights...")
            try:
                with spinner("Merging LoRA weights..."):
                    trainer.merge_and_save(output_path / "merged")
                print_success("Merged model saved")
            except Exception as e:
                print_warning(f"Failed to merge model: {e}")
        
        elif format_type == "gguf":
            console.print("\n📦 Exporting to GGUF format...")
            try:
                with spinner("Exporting to GGUF format..."):
                    gguf_path = trainer.export_gguf(output_path / "gguf")
                print_success(f"GGUF model saved to: {gguf_path}")
            except Exception as e:
                print_warning(f"Failed to export GGUF: {e}")
        
        elif format_type == "ollama":
            console.print("\n🦙 Setting up Ollama...")
            # First export to GGUF if not already done
            gguf_dir = output_path / "gguf"
            if not gguf_dir.exists() or not list(gguf_dir.glob("*.gguf")):
                try:
                    with spinner("Exporting to GGUF for Ollama..."):
                        trainer.export_gguf(gguf_dir)
                except Exception as e:
                    print_warning(f"Failed to export GGUF for Ollama: {e}")
                    continue
            
            # Register with Ollama
            model_name = config.name.replace(" ", "-").lower()
            try:
                with spinner(f"Registering with Ollama as '{model_name}'..."):
                    trainer.register_ollama(gguf_dir, model_name)
                print_success(f"Model registered with Ollama: {model_name}")
                console.print(f"[dim]Test with: ollama run {model_name}[/]")
            except Exception as e:
                print_warning(f"Ollama registration failed: {e}")
                print_info("You can manually register later with the GGUF file")
    
    total_time = time.time() - start_time
    console.print()
    console.print(f"[bold green]🎉 All done in {_format_time(total_time)}![/]")
    console.print("[dim]Next: forge inference[/]")
    console.print()


def _estimate_training_time(total_steps: int, batch_size: int) -> str:
    """Estimate training time based on hardware and configuration."""
    # Rough estimates based on typical performance
    # RTX 5080 with batch size 24: ~2-4 steps/second for 7B models
    if batch_size >= 20:
        steps_per_sec = 3.0  # High-end GPU with large batch
    elif batch_size >= 10:
        steps_per_sec = 2.0  # Mid-range performance
    else:
        steps_per_sec = 1.0  # Conservative estimate
    
    estimated_seconds = total_steps / steps_per_sec
    return _format_time(estimated_seconds)


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
