"""
forge test - Interactive model testing.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown

from forge.ui.console import console, print_success, print_error, print_info
from forge.ui.progress import spinner, thinking_spinner
from forge.core.config import load_config, get_config_path


def test_command(
    model_path: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to model"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to forge.yaml"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Maximum response tokens"),
    use_ollama: bool = typer.Option(False, "--ollama", help="Use Ollama for inference (requires Ollama installed)"),
    ollama_model: Optional[str] = typer.Option(None, "--ollama-model", help="Ollama model name (e.g., gemma2:2b, llama3.2:3b)"),
):
    """
    Test your trained model interactively.
    
    Chat with your fine-tuned model to evaluate its behavior.
    
    Use --ollama to test with Ollama instead of loading weights directly.
    """
    # Ollama mode - test with Ollama API
    if use_ollama:
        model_name = ollama_model or "gemma2:2b"
        console.print(f"\n[bold]💬 Testing with Ollama:[/] {model_name}\n")
        _run_ollama_chat(model_name, system_prompt, max_tokens)
        return
    
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
    
    console.print(f"\n[bold]💬 Testing model:[/] {model_path}\n")
    
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
                    console.print(f"[dim]System prompt updated.[/]\n")
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
    import subprocess
    import json
    
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
