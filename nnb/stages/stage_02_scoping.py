"""Stage 2: Scoping Questions."""

from rich.console import Console
from rich.prompt import Prompt, Confirm
import yaml

from nnb.gemini_brain.client import GeminiClient
from nnb.models.project_spec import ProjectSpec
from nnb.orchestrator.state import State
from nnb.utils.logging import get_logger
from nnb.utils.steering_loader import SteeringLoader

console = Console()
logger = get_logger(__name__)


INTERVIEWER_PROMPT = """
{steering_context}

================================================================================
YOUR TASK: INTAKE INTERVIEW
================================================================================

You are an expert ML engineer conducting an intake interview.

The user has described their project:
{user_description}

Your task:
1. Categorize the project type (vision, nlp, tabular, time-series, multimodal, rl, other)
2. Identify the most likely ML framework (pytorch, tensorflow, jax, scikit-learn)
3. Determine the specific task (classification, regression, detection, segmentation, etc.)
4. Identify if they want to use a standard dataset (like MNIST, CIFAR10 from torchvision) or custom data
5. Identify 3-5 critical questions that need answers to proceed

REMEMBER:
- Follow the three-layer architecture (Orchestrator, Docker, Gemini)
- All training will run in Docker containers
- Data can be from torchvision/standard datasets OR custom data mounted read-only
- Keep questions focused and actionable

Output format (JSON):
{{
  "project_type": "...",
  "framework": "...",
  "task": "...",
  "dataset_source": "torchvision|custom|<specific_dataset_name>",
  "complexity_estimate": "low|medium|high",
  "questions": [
    {{"topic": "...", "question": "...", "why_needed": "..."}}
  ]
}}
"""


def ask_scoping_questions(project: "Project", user_description: str) -> None:  # noqa: F821
    """Ask targeted scoping questions."""
    
    try:
        # Get questions from Gemini
        gemini = GeminiClient()
        
        # Include steering context
        steering_context = SteeringLoader.format_for_prompt("scoping")
        
        prompt = INTERVIEWER_PROMPT.format(
            steering_context=steering_context,
            user_description=user_description
        )
        
        response = gemini.generate_json(prompt)
        
        logger.info(f"Gemini analysis: {response['project_type']}, {response['framework']}")
        
        # Ask questions
        console.print("\n[bold]📋 I have a few questions to clarify:[/bold]\n")
        
        answers = {}
        for q in response["questions"]:
            console.print(f"[cyan]Q: {q['question']}[/cyan]")
            console.print(f"[dim]   (Needed for: {q['why_needed']})[/dim]")
            answer = Prompt.ask("Your answer")
            answers[q["topic"]] = answer
            console.print()
        
        # Build specification
        spec_data = {
            "project_type": response["project_type"],
            "framework": response["framework"],
            "task": response["task"],
            "dataset_source": response.get("dataset_source", "custom"),
            "compute_budget": "medium",
            "latency_priority": "balanced",
            "pretrained": False,
        }
        
        # Extract specific details from answers
        for topic, answer in answers.items():
            if "class" in topic.lower():
                try:
                    spec_data["num_classes"] = int(answer)
                except ValueError:
                    pass
            elif "shape" in topic.lower() or "size" in topic.lower():
                if "input" in topic.lower():
                    spec_data["input_shape"] = answer
                elif "output" in topic.lower():
                    spec_data["output_shape"] = answer
            elif "pretrain" in topic.lower():
                spec_data["pretrained"] = "yes" in answer.lower()
            elif "export" in topic.lower():
                spec_data["export_format"] = answer
        
        # Show specification
        console.print("\n[bold]📄 Project Specification:[/bold]\n")
        for key, value in spec_data.items():
            console.print(f"  {key}: [green]{value}[/green]")
        
        # Confirm
        console.print()
        confirmed = Confirm.ask("Does this look correct?", default=True)
        
        if not confirmed:
            console.print("[yellow]⚠️  Please restart and provide more details[/yellow]")
            return
        
        # Save specification
        spec = ProjectSpec(**spec_data)
        project._spec = spec
        
        # Save to YAML
        with open(project.spec_file, "w") as f:
            yaml.dump(spec.dict(), f, default_flow_style=False)
        
        # Transition state
        project.transition_to(State.SPEC_CONFIRMED)
        
        console.print("\n[green]✓ Specification confirmed[/green]")
        
        # Move to Stage 3
        from nnb.stages.stage_03_data_requirements import generate_data_requirements
        generate_data_requirements(project)
        
    except Exception as e:
        logger.error(f"Error in scoping: {e}", exc_info=True)
        console.print(f"[red]❌ Error: {e}[/red]")
        raise
