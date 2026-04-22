"""Stage 1: User Conversation."""

from rich.console import Console
from rich.prompt import Prompt

from nnb.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


def start_conversation(project: "Project") -> None:  # noqa: F821
    """Start interactive conversation with user."""
    
    console.print("\n[bold]🎯 Let's build your neural network![/bold]\n")
    console.print("Tell me about your project. What are you trying to build?")
    console.print("(Be as detailed as you like - domain, goal, constraints, etc.)\n")
    
    # Collect user description
    description_lines = []
    console.print("[dim]Type your description (press Enter twice when done):[/dim]\n")
    
    empty_count = 0
    while empty_count < 2:
        line = input()
        if line.strip():
            description_lines.append(line)
            empty_count = 0
        else:
            empty_count += 1
    
    user_description = "\n".join(description_lines).strip()
    
    if not user_description:
        console.print("[yellow]⚠️  No description provided. Please try again.[/yellow]")
        return
    
    # Save conversation
    conversation_file = project.project_dir / "conversation.txt"
    with open(conversation_file, "w") as f:
        f.write(f"User Description:\n{user_description}\n\n")
    
    logger.info(f"Saved user description ({len(user_description)} chars)")
    
    console.print("\n[green]✓ Description saved[/green]")
    console.print("\n🤖 Analyzing your project...")
    
    # Move to Stage 2: Scoping Questions
    from nnb.stages.stage_02_scoping import ask_scoping_questions
    ask_scoping_questions(project, user_description)
