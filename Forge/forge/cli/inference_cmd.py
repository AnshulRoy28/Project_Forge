"""
forge inference - Interactive model inference testing via Docker.

All inference runs inside Docker containers for consistency.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional

import typer
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown

from forge.ui.console import console, print_success, print_error, print_info, print_warning
from forge.ui.progress import spinner, thinking_spinner
from forge.core.config import load_config, get_config_path


def inference_command(
    model_path: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to model"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to forge.yaml"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Maximum response tokens"),
    use_ollama: bool = typer.Option(False, "--ollama", help="Use Ollama for inference (requires Ollama installed)"),
    ollama_model: Optional[str] = typer.Option(None, "--ollama-model", help="Ollama model name (e.g., gemma2:2b, llama3.2:3b)"),
    local: bool = typer.Option(False, "--local", help="Run inference locally instead of Docker (requires GPU support)"),
):
    """
    Run inference on your trained model interactively.
    
    By default, runs inference inside Docker container for GPU compatibility.
    Use --ollama to test with Ollama, or --local to run directly on host.
    """
    # Ollama mode - test with Ollama API
    if use_ollama:
        model_name = ollama_model or "gemma2:2b"
        console.print(f"\n[bold]💬 Inference with Ollama:[/] {model_name}\n")
        _run_ollama_chat(model_name, system_prompt, max_tokens)
        return
    
    # Local mode (legacy) - run on host
    if local:
        console.print("\n[bold]💬 Running local inference[/]\n")
        _run_local_inference(model_path, config_path, system_prompt, max_tokens)
        return
    
    # Docker mode (default) - run inside container
    _run_docker_inference(model_path, config_path, system_prompt, max_tokens)


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


def _run_docker_inference(
    model_path: Optional[Path],
    config_path: Optional[Path],
    system_prompt: Optional[str],
    max_tokens: int
):
    """Run inference inside Docker container."""
    
    # Check Docker availability
    if not _check_docker():
        print_error("Docker is required for inference")
        console.print("\n[dim]Alternatives:[/]")
        console.print("  [cyan]forge inference --ollama[/] - Use Ollama")
        console.print("  [cyan]forge inference --local[/]  - Run on host (if GPU supported)")
        raise typer.Exit(1)
    
    arch = _detect_gpu_architecture()
    image = f"forge:{arch}"
    
    # Check if image exists
    check = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True
    )
    
    if check.returncode != 0:
        print_warning(f"Docker image {image} not found")
        console.print("[dim]Build with: forge docker build[/]")
        console.print("[dim]Or use: forge inference --ollama[/]")
        raise typer.Exit(1)
    
    console.print(f"\n[bold]💬 Inference via Docker ({arch})[/]\n")
    
    # Build docker run command
    cmd = [
        "docker", "run",
        "--gpus", "all",
        "--rm",
        "-it",  # Interactive for chat
        "-v", f"{Path('./output').resolve()}:/output",
    ]
    
    # Mount config if exists
    config_file = Path("./forge.yaml")
    if config_file.exists():
        cmd.extend(["-v", f"{config_file.resolve()}:/app/forge.yaml:ro"])
    
    # Pass GEMINI_API_KEY if set
    if "GEMINI_API_KEY" in os.environ:
        cmd.extend(["-e", f"GEMINI_API_KEY={os.environ['GEMINI_API_KEY']}"])
    
    cmd.append(image)
    
    # Build forge command inside container
    forge_cmd = ["inference", "--local"]  # Use --local inside container
    if system_prompt:
        forge_cmd.extend(["--system", system_prompt])
    if max_tokens != 512:
        forge_cmd.extend(["--max-tokens", str(max_tokens)])
    
    cmd.extend(forge_cmd)
    
    console.print(f"[dim]$ docker run --gpus all ... forge inference --local[/]\n")
    
    # Run interactively
    subprocess.run(cmd)


def _run_local_inference(
    model_path: Optional[Path],
    config_path: Optional[Path],
    system_prompt: Optional[str],
    max_tokens: int
):
    """Run inference locally on host."""
    
    # Determine model path
    if not model_path:
        config_file = config_path or get_config_path()
        
        if config_file.exists():
            config = load_config(config_file)
            base_output = Path(config.output.dir)
        else:
            base_output = Path("./output")
        
        # Prefer merged model if available (has full weights)
        merged_path = base_output / "merged"
        if merged_path.exists() and (merged_path / "config.json").exists():
            model_path = merged_path
            print_info("Using merged model (full weights)")
        elif base_output.exists():
            model_path = base_output
            print_info("Using LoRA adapter model")
        else:
            model_path = base_output
    
    if not model_path.exists():
        print_error(f"Model not found: {model_path}")
        console.print("[dim]Train a model first with 'forge train'[/]")
        raise typer.Exit(1)
    
    console.print(f"[bold]💬 Testing model:[/] {model_path}\n")
    
    # Load model
    try:
        with spinner("Loading model for inference..."):
            model, tokenizer = _load_model(model_path)
        
        print_success("Model loaded successfully!")
        console.print()
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        raise typer.Exit(1)
    
    # Display instructions
    console.print("[dim]Type your message and press Enter. Commands:[/]")
    console.print("[dim]  /clear  - Clear chat history[/]")
    console.print("[dim]  /system - Change system prompt[/]")
    console.print("[dim]  /quit   - Exit[/]")
    console.print()
    
    # Chat loop
    history = []
    current_system = system_prompt or "You are a helpful assistant."
    
    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/]")
        except (KeyboardInterrupt, EOFError):
            break
        
        if not user_input.strip():
            continue
        
        # Handle commands
        if user_input.strip().startswith("/"):
            cmd = user_input.strip().lower()
            
            if cmd == "/quit" or cmd == "/exit":
                break
            elif cmd == "/clear":
                history = []
                console.print("[dim]Chat history cleared.[/]\n")
                continue
            elif cmd.startswith("/system"):
                new_system = user_input[7:].strip()
                if new_system:
                    current_system = new_system
                    console.print("[dim]System prompt updated.[/]\n")
                else:
                    console.print(f"[dim]Current system: {current_system}[/]\n")
                continue
            else:
                console.print(f"[dim]Unknown command: {cmd}[/]\n")
                continue
        
        # Generate response
        try:
            with spinner("Generating..."):
                response = _generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    message=user_input,
                    history=history,
                    system_prompt=current_system,
                    max_tokens=max_tokens,
                )
            
            # Display response
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold magenta]Assistant[/]",
                border_style="magenta",
            ))
            console.print()
            
            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            print_error(f"Generation failed: {e}")
    
    console.print("\n[dim]Goodbye![/]\n")


def _load_model(model_path: Path):
    """Load model and tokenizer for inference."""
    try:
        # Try Unsloth first for efficiency
        from unsloth import FastLanguageModel
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=2048,
            load_in_4bit=True,
        )
        
        FastLanguageModel.for_inference(model)
        return model, tokenizer
        
    except ImportError:
        # Fallback to transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        return model, tokenizer


def _generate_response(
    model,
    tokenizer,
    message: str,
    history: list,
    system_prompt: str,
    max_tokens: int,
) -> str:
    """Generate a response from the model."""
    import torch
    
    # Build conversation
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})
    
    # Format for model
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback formatting
        prompt = f"### System:\n{system_prompt}\n\n"
        for msg in history:
            role = "### User:" if msg["role"] == "user" else "### Assistant:"
            prompt += f"{role}\n{msg['content']}\n\n"
        prompt += f"### User:\n{message}\n\n### Assistant:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    return response.strip()


def _run_ollama_chat(model_name: str, system_prompt: Optional[str], max_tokens: int):
    """Run interactive chat using Ollama."""
    
    # Check if Ollama is available
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            print_error("Ollama is not running. Start it with 'ollama serve'")
            return
    except FileNotFoundError:
        print_error("Ollama not found. Install from: https://ollama.ai")
        return
    except subprocess.TimeoutExpired:
        print_error("Ollama is not responding")
        return
    
    # Check if model is available
    if model_name not in result.stdout:
        print_info(f"Pulling model {model_name}...")
        pull_result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=False,
        )
        if pull_result.returncode != 0:
            print_error(f"Failed to pull model: {model_name}")
            return
    
    print_success(f"Connected to Ollama - {model_name}")
    
    # Display instructions
    console.print()
    console.print("[dim]Type your message and press Enter. Commands:[/]")
    console.print("[dim]  /clear  - Clear chat history[/]")
    console.print("[dim]  /quit   - Exit[/]")
    console.print()
    
    history = []
    current_system = system_prompt or "You are a helpful assistant."
    
    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/]")
        except (KeyboardInterrupt, EOFError):
            break
        
        if not user_input.strip():
            continue
        
        # Handle commands
        if user_input.strip().startswith("/"):
            cmd = user_input.strip().lower()
            if cmd == "/quit" or cmd == "/exit":
                break
            elif cmd == "/clear":
                history = []
                console.print("[dim]Chat history cleared.[/]\n")
                continue
        
        # Build messages
        messages = [{"role": "system", "content": current_system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        
        # Call Ollama API
        try:
            with spinner("Generating..."):
                result = subprocess.run(
                    ["ollama", "run", model_name, user_input],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                response = result.stdout.strip()
            
            # Display response
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold magenta]Assistant[/]",
                border_style="magenta",
            ))
            console.print()
            
            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
        except subprocess.TimeoutExpired:
            print_error("Response timed out")
        except Exception as e:
            print_error(f"Error: {e}")
    
    console.print("\n[dim]Goodbye![/]\n")
