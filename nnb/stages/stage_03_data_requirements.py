"""Stage 3: Training Data Requirements."""

from rich.console import Console

from nnb.gemini_brain.client import GeminiClient
from nnb.orchestrator.state import State
from nnb.utils.logging import get_logger
from nnb.utils.steering_loader import SteeringLoader

console = Console()
logger = get_logger(__name__)


DATA_REQUIREMENTS_PROMPT = """
{steering_context}

================================================================================
YOUR TASK: DATA REQUIREMENTS SPECIFICATION
================================================================================

You are a data requirements specialist.

Project specification:
{spec}

Dataset source: {dataset_source}

Generate a comprehensive Data Requirements Report covering:

IF dataset_source is "torchvision" or a specific dataset name (like "MNIST", "CIFAR10"):
1. Confirm the dataset will be automatically downloaded from torchvision.datasets
2. Specify the dataset name and any required parameters
3. Note that no manual data validation is needed
4. Provide example code for loading the dataset
5. Mention that data will be automatically downloaded to /data in the container

IF dataset_source is "custom":
1. Format (images, CSV, parquet, JSONL, audio, etc.)
2. Minimum dataset size
3. Recommended dataset size
4. Required folder/label structure with examples
5. Expected file naming conventions
6. Any format-specific requirements (image resolution, CSV columns, etc.)

REMEMBER:
- For custom data: Data will be mounted READ-ONLY at /data in Docker container
- For torchvision data: Data will be auto-downloaded to /data
- Focus on FORMAT and STRUCTURE validation (data is pre-cleaned)
- Provide clear, actionable requirements
- Include examples of correct structure

Output as clear Markdown with sections and examples.
"""


def generate_data_requirements(project: "Project") -> None:  # noqa: F821
    """Generate data requirements document."""
    
    try:
        console.print("\n🔍 Generating data requirements...")
        
        # Get requirements from Gemini
        gemini = GeminiClient()
        
        # Include steering context
        steering_context = SteeringLoader.format_for_prompt("data_requirements")
        
        spec_str = "\n".join(f"{k}: {v}" for k, v in project._spec.dict().items())
        dataset_source = project._spec.dataset_source
        
        prompt = DATA_REQUIREMENTS_PROMPT.format(
            steering_context=steering_context,
            spec=spec_str,
            dataset_source=dataset_source
        )
        
        requirements = gemini.generate(prompt, temperature=0.3)
        
        # Save to file
        project.data_requirements_file.write_text(requirements,encoding="utf-8")
        
        # Transition state
        project.transition_to(State.DATA_REQUIRED)
        
        console.print("\n[green]✓ Data requirements generated[/green]")
        console.print(f"📄 Saved to: {project.data_requirements_file}")
        
        # Display requirements
        console.print("\n[bold]📋 Data Requirements:[/bold]\n")
        console.print(requirements)
        
        # Show next step based on dataset source
        console.print("\n[bold]Next step:[/bold]")
        if dataset_source in ["torchvision", "MNIST", "CIFAR10", "CIFAR100", "FashionMNIST"]:
            console.print("  [cyan]No data validation needed - dataset will be auto-downloaded[/cyan]")
            console.print("  Run: [cyan]nnb env build[/cyan] to set up the environment")
        else:
            console.print("  Run: [cyan]nnb data validate --path <your-data-directory>[/cyan]")
        
    except Exception as e:
        logger.error(f"Error generating data requirements: {e}", exc_info=True)
        console.print(f"[red]❌ Error: {e}[/red]")
        raise
