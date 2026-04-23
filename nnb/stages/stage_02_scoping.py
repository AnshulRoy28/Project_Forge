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

IMP_POINTS_PROMPT = """
You are an expert ML engineer summarizing key architectural and implementation decisions.

The user described their project:
{user_description}

Confirmed specification:
{spec}

Scoping Q&A:
{qa_summary}

Generate a structured "Important Points" document that captures ALL key decisions
that downstream code generation will need to reference. This document is the
single source of truth for:

1. **Architecture** — Model architecture choice and why (e.g., ResNet-18, simple CNN, LSTM)
2. **Dataset** — Source (torchvision/custom), name, path, format, preprocessing needed
3. **Training Strategy** — Optimizer, learning rate schedule, batch size, epochs, loss function
4. **Input/Output** — Exact shapes, data types, normalization
5. **Compute** — GPU/CPU preference, memory constraints, mixed precision
6. **Export** — Target format if any (ONNX, TFLite, etc.)
7. **Key Constraints** — Anything unusual the user mentioned
8. **Checkpointing** — Frequency, what to save

Format as clean Markdown with headers and bullet points. Be specific and actionable —
this document will be directly referenced during code generation.

Output ONLY the Markdown content, no code fences.
"""


def _generate_imp_points(
    project: "Project",  # noqa: F821
    gemini: "GeminiClient",  # noqa: F821
    user_description: str,
    spec_data: dict,
    answers: dict,
) -> None:
    """Generate imp_points.md — key decisions for downstream stages."""
    
    spec_str = "\n".join(f"- {k}: {v}" for k, v in spec_data.items())
    qa_str = "\n".join(f"- **{topic}**: {answer}" for topic, answer in answers.items())
    
    prompt = IMP_POINTS_PROMPT.format(
        user_description=user_description,
        spec=spec_str,
        qa_summary=qa_str,
    )
    
    try:
        imp_points = gemini.generate(prompt, temperature=0.3)
        
        imp_file = project.project_dir / "imp_points.md"
        imp_file.write_text(imp_points, encoding="utf-8")
        
        logger.info(f"Generated imp_points.md ({len(imp_points)} chars)")
        console.print("[green]✓ Key decisions saved to imp_points.md[/green]")
        
    except Exception as e:
        logger.warning(f"Failed to generate imp_points.md: {e}")
        console.print("[yellow]⚠️  Could not generate decisions summary, continuing...[/yellow]")


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
        
        # Generate imp_points.md — key decisions for downstream stages
        console.print("\n📝 Generating key decisions summary...")
        _generate_imp_points(project, gemini, user_description, spec_data, answers)
        
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
