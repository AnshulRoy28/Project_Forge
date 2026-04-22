"""Stage 6: Code Generation + Mock Run."""

from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from nnb.models.project_spec import MockRunResult
from nnb.gemini_brain.client import GeminiClient
from nnb.docker_runtime.container import get_container
from nnb.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


def generate_training_code(project: "Project") -> str:  # noqa: F821
    """Generate training code using Gemini."""
    spec = project._spec
    
    # Read conversation for context
    conversation_file = project.project_dir / "conversation.txt"
    conversation = conversation_file.read_text() if conversation_file.exists() else ""
    
    prompt = f"""Generate a complete PyTorch training script for the following project:

Project Type: {spec.project_type}
Task: {spec.task}
Framework: {spec.framework}
Dataset Source: {spec.dataset_source}
Compute Budget: {spec.compute_budget}

User Description:
{conversation}

Requirements:
1. Use PyTorch and torchvision
2. Download MNIST dataset automatically to /data directory
3. Create a simple CNN model for digit classification
4. Include training loop with progress bars
5. Save model checkpoints to /workspace/checkpoints
6. Log metrics to /workspace/logs
7. Use appropriate batch size for {spec.compute_budget} compute budget
8. Include validation loop
9. Save final model to /workspace/model.pth
10. Print training progress and final accuracy

Generate a complete, runnable train.py script. Include:
- Imports
- Model definition (simple CNN)
- Data loading with transforms
- Training loop
- Validation loop
- Checkpoint saving
- Logging

Output ONLY the Python code, no explanations or markdown formatting."""

    gemini = GeminiClient()
    code = gemini.generate(prompt, temperature=0.3)
    
    # Clean up any markdown formatting
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    
    return code.strip()


def generate_mock_script() -> str:
    """Generate a simple mock training script for quick validation."""
    return """#!/usr/bin/env python3
\"\"\"Mock training script - validates environment setup.\"\"\"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys

print("=" * 60)
print("MOCK TRAINING RUN - Environment Validation")
print("=" * 60)

# Check PyTorch
print(f"\\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")

# Simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    
    def forward(self, x):
        return self.fc(x.view(-1, 784))

print("\\n✓ Model definition successful")

# Create model
model = SimpleNet()
print("✓ Model instantiation successful")

# Test forward pass
dummy_input = torch.randn(2, 1, 28, 28)
try:
    output = model(dummy_input)
    print(f"✓ Forward pass successful: {output.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test backward pass
try:
    loss = output.sum()
    loss.backward()
    print("✓ Backward pass successful")
except Exception as e:
    print(f"✗ Backward pass failed: {e}")
    sys.exit(1)

# Test optimizer
try:
    optimizer = optim.Adam(model.parameters())
    optimizer.step()
    print("✓ Optimizer step successful")
except Exception as e:
    print(f"✗ Optimizer step failed: {e}")
    sys.exit(1)

# Test data loading (small subset)
print("\\n✓ Testing data loading...")
try:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST(
        '/data',
        train=True,
        download=True,
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print(f"✓ Data loading successful: batch shape {batch[0].shape}")
    
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    sys.exit(1)

print("\\n" + "=" * 60)
print("✓ MOCK RUN PASSED - Environment is ready!")
print("=" * 60)
"""


def run_mock_training(project: "Project") -> MockRunResult:  # noqa: F821
    """Run mock training to validate environment."""
    try:
        console.print("\\n🧪 Running mock training pass...")
        console.print("[dim]This validates the environment setup[/dim]\\n")
        
        # Generate mock script
        mock_script = generate_mock_script()
        
        # Save mock script
        workspace_dir = project.project_dir / "workspace"
        workspace_dir.mkdir(exist_ok=True)
        
        mock_file = workspace_dir / "mock_train.py"
        mock_file.write_text(mock_script,encoding="utf-8")
        logger.info(f"Generated mock script: {mock_file}")
        
        # Get container
        container = get_container(project.project_id)
        
        # Prepare directories
        data_dir = str(project.project_dir / "data")
        workspace_dir_str = str(workspace_dir)
        
        (project.project_dir / "data").mkdir(exist_ok=True)
        
        console.print("🐳 Starting container...")
        
        # Start container with volume mounts
        container_instance = container.start(
            command="python /workspace/mock_train.py",
            detach=False,
            workspace_dir=workspace_dir_str,
            data_dir=data_dir,
            remove=True,
        )
        
        # Get logs
        console.print("\\n📋 Mock run output:\\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running mock training...", total=None)
            
            # Wait for container to finish
            result = container_instance.wait()
            progress.update(task, completed=True)
        
        # Get logs
        logs = container_instance.logs().decode('utf-8')
        console.print(logs)
        
        # Check exit code
        exit_code = result['StatusCode']
        
        if exit_code == 0:
            console.print("\\n[green]✓ Mock run passed![/green]")
            console.print("\\n💡 Next step:")
            console.print("  Run: [cyan]nnb train[/cyan]")
            
            return MockRunResult(
                succeeded=True,
                forward_pass_succeeded=True,
                backward_pass_succeeded=True,
                optimizer_step_succeeded=True,
            )
        else:
            console.print(f"\\n[red]✗ Mock run failed with exit code {exit_code}[/red]")
            
            return MockRunResult(
                succeeded=False,
                error_message=f"Mock run failed with exit code {exit_code}",
            )
            
    except Exception as e:
        logger.error(f"Mock run failed: {e}", exc_info=True)
        console.print(f"\\n[red]✗ Mock run failed: {e}[/red]")
        
        return MockRunResult(
            succeeded=False,
            error_message=str(e),
        )
