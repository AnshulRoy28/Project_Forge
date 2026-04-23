"""Stage 6: Code Generation + Mock Run."""

import json
import re
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from nnb.models.project_spec import MockRunResult
from nnb.gemini_brain.client import GeminiClient
from nnb.docker_runtime.container import get_container
from nnb.utils.logging import get_logger
from nnb.utils.steering_loader import SteeringLoader

console = Console()
logger = get_logger(__name__)

MAX_RETRIES = 3


# =============================================================================
# CODE GENERATION PROMPT
# =============================================================================

CODE_GENERATION_PROMPT = """
{steering_context}

================================================================================
YOUR TASK: CODE GENERATION
================================================================================

You are an expert ML engineer writing production-quality training code.

Project specification:
{spec}

Key architectural decisions (imp_points.md):
{imp_points}

Data manifest / requirements:
{data_context}

User's original project description:
{conversation}

Generate a COMPLETE, RUNNABLE training codebase as separate files.
Each file must be immediately runnable inside a Docker container with:
- Data at /data (read-only)
- Workspace at /workspace (read-write)
- Checkpoints saved to /workspace/checkpoints/

Files to generate:

1. **model.py** — Model architecture definition with clear docstrings
2. **dataset.py** — Data loading and preprocessing pipeline
3. **train.py** — Complete training loop with:
   - Configurable hyperparameters from config.yaml
   - Validation loop
   - Checkpoint saving every N epochs
   - Progress logging (loss, accuracy, ETA)
   - Final model save to /workspace/model.pth
4. **config.yaml** — ALL hyperparameters externalized (never hardcoded)
5. **requirements.txt** — Minimal pip dependencies with pinned versions

CRITICAL RULES:
- Follow the steering document conventions strictly
- Include comprehensive error handling in all files
- Log all important events
- No hardcoded paths — use /data and /workspace
- Code must be immediately runnable: python /workspace/train.py
- Use the EXACT dataset source specified (torchvision dataset OR custom path)

OUTPUT FORMAT:
Output each file using this EXACT delimiter format. Do NOT use JSON.
Do NOT wrap files in markdown code fences.

=== FILE: model.py ===
...complete file content here...

=== FILE: dataset.py ===
...complete file content here...

=== FILE: train.py ===
...complete file content here...

=== FILE: config.yaml ===
...complete file content here...

=== FILE: requirements.txt ===
...complete file content here...

=== SUMMARY ===
Brief description of what was generated.
"""


# =============================================================================
# MOCK RUN DEBUG PROMPT
# =============================================================================

MOCK_DEBUG_PROMPT = """
{steering_context}

================================================================================
YOUR TASK: DEBUG FAILED MOCK RUN
================================================================================

You are debugging a failed mock training run.

Generated code files:
{code_files}

Error traceback:
{traceback}

Attempt {retry_count} of {max_retries}.

Your task:
1. Identify the root cause of the failure
2. Generate corrected file contents for ONLY the files that need fixing
3. Do NOT change files that are working correctly

CRITICAL: The fix must address the SPECIFIC error shown in the traceback.
Do not refactor unrelated code.

Respond with valid JSON in this EXACT format:
{{
  "root_cause": "Brief description of what went wrong",
  "fix_explanation": "What the fix does",
  "files": {{
    "filename.py": "...complete corrected file content..."
  }}
}}

Only include files that need changes in the "files" dict.
Respond with valid JSON only, no markdown formatting.
"""


# =============================================================================
# CODE GENERATION
# =============================================================================


def generate_code(project: "Project") -> None:  # noqa: F821
    """Generate training code using Gemini with full steering context."""

    console.print("\n[bold]🔧 Generating training code...[/bold]\n")

    try:
        gemini = GeminiClient()

        # Load steering context for code generation stage
        steering_context = SteeringLoader.format_for_prompt(
            "code_generation", include_code_rules=True
        )

        # Load project spec
        spec = project._spec
        spec_str = "\n".join(f"- {k}: {v}" for k, v in spec.dict().items())

        # Load imp_points.md (key decisions from scoping)
        imp_file = project.project_dir / "imp_points.md"
        imp_points = ""
        if imp_file.exists():
            imp_points = imp_file.read_text(encoding="utf-8")
        else:
            logger.warning("imp_points.md not found — generating without it")
            console.print("[yellow]⚠️  imp_points.md not found, generating with spec only[/yellow]")

        # Load data context (manifest or requirements)
        data_context = _load_data_context(project)

        # Load conversation history
        conversation_file = project.project_dir / "conversation.txt"
        conversation = ""
        if conversation_file.exists():
            conversation = conversation_file.read_text(encoding="utf-8")

        # Build prompt
        prompt = CODE_GENERATION_PROMPT.format(
            steering_context=steering_context,
            spec=spec_str,
            imp_points=imp_points,
            data_context=data_context,
            conversation=conversation,
        )

        console.print("🤖 Asking Gemini to generate code...")

        raw_response = gemini.generate(prompt, temperature=0.3)

        # Parse delimiter-based response
        files, summary = _parse_delimited_response(raw_response)

        if not files:
            raise ValueError("Gemini returned no files — response may be malformed")

        # Write files to workspace
        workspace = project.project_dir / "workspace"
        workspace.mkdir(exist_ok=True)

        console.print("\n📁 Writing generated files:\n")

        for filename, content in files.items():
            filepath = workspace / filename
            filepath.write_text(content, encoding="utf-8")
            line_count = content.count("\n") + 1
            console.print(f"  [green]✓[/green] {filename} ({line_count} lines)")

        logger.info(f"Code generation complete: {summary}")
        console.print(f"\n📝 {summary}")

        console.print("\n[green]✓ Code generation complete![/green]")
        console.print("\n💡 Next step:")
        console.print("  Run: [cyan]nnb mock-run[/cyan] to validate the generated code")

    except Exception as e:
        logger.error(f"Code generation failed: {e}", exc_info=True)
        console.print(f"\n[red]❌ Code generation failed: {e}[/red]")
        console.print("💡 Try running [cyan]nnb generate[/cyan] again")
        raise



def _parse_delimited_response(raw: str) -> tuple:
    """Parse Gemini's delimiter-based file output.
    
    Expected format:
        === FILE: model.py ===
        ...content...
        
        === FILE: dataset.py ===
        ...content...
        
        === SUMMARY ===
        Brief description.
    
    Returns (files_dict, summary_str).
    """
    
    files = {}
    summary = "Code generated successfully"
    
    # Split on === FILE: xxx === or === SUMMARY === markers
    pattern = r'===\s*FILE:\s*(.+?)\s*===|===\s*SUMMARY\s*==='
    parts = re.split(pattern, raw)
    
    # parts alternates: [preamble, filename1, content1, filename2, content2, ..., None, summary]
    # re.split with groups: non-matching text alternates with captured group
    i = 0
    while i < len(parts):
        part = parts[i]
        
        if part is None:
            # This is the SUMMARY marker (captured group is None for non-FILE pattern)
            if i + 1 < len(parts):
                summary = parts[i + 1].strip()
            i += 2
            continue
        
        # Check if this looks like a filename (captured group from FILE pattern)
        if part and '.' in part and len(part) < 50 and '===' not in part:
            # This is a filename captured group
            if i + 1 < len(parts):
                content = parts[i + 1] if parts[i + 1] else ""
                # Strip leading/trailing whitespace and code fences
                content = content.strip()
                content = _strip_code_fences(content)
                if content:
                    files[part.strip()] = content
            i += 2
        else:
            i += 1
    
    if not files:
        # Fallback: try to extract from markdown code blocks
        files = _parse_markdown_fallback(raw)
    
    return files, summary


def _strip_code_fences(content: str) -> str:
    """Remove markdown code fences if present."""
    lines = content.split('\n')
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    return '\n'.join(lines)


def _parse_markdown_fallback(raw: str) -> dict:
    """Fallback parser: extract files from markdown code blocks with filename headers."""
    files = {}
    
    # Match patterns like: ### model.py\n```python\n...\n```
    pattern = r'(?:#{1,4}\s*)?(\w+\.(?:py|yaml|yml|txt))\s*\n```\w*\n(.*?)```'
    matches = re.findall(pattern, raw, re.DOTALL)
    
    for filename, content in matches:
        files[filename.strip()] = content.strip()
    
    return files


def _load_data_context(project: "Project") -> str:  # noqa: F821
    """Load data manifest or requirements for prompt context."""

    # Try data manifest first (generated after validation)
    if project.data_manifest_file.exists():
        try:
            with open(project.data_manifest_file) as f:
                manifest = json.load(f)
            return json.dumps(manifest, indent=2)
        except Exception:
            pass

    # Fall back to data requirements
    if project.data_requirements_file.exists():
        return project.data_requirements_file.read_text(encoding="utf-8")

    # Fall back to spec dataset_source
    return f"Dataset source: {project._spec.dataset_source}"


# =============================================================================
# MOCK RUN
# =============================================================================


def run_mock_training(project: "Project") -> MockRunResult:  # noqa: F821
    """Run mock training to validate generated code in Docker container."""

    try:
        console.print("\n[bold]🧪 Running mock training pass...[/bold]")
        console.print("[dim]This validates the generated code in a Docker container[/dim]\n")

        # Verify generated code exists
        workspace = project.project_dir / "workspace"
        train_file = workspace / "train.py"

        if not train_file.exists():
            console.print("[red]❌ No train.py found in workspace[/red]")
            console.print("💡 Run [cyan]nnb generate[/cyan] first")
            return MockRunResult(
                succeeded=False,
                error_message="train.py not found — run 'nnb generate' first",
            )

        # Generate mock runner script
        mock_script = _generate_mock_runner(workspace)
        mock_file = workspace / "mock_runner.py"
        mock_file.write_text(mock_script, encoding="utf-8")

        # Run mock in container with retry loop
        result = _run_with_retries(project, workspace)
        return result

    except Exception as e:
        logger.error(f"Mock run failed: {e}", exc_info=True)
        console.print(f"\n[red]❌ Mock run failed: {e}[/red]")
        return MockRunResult(succeeded=False, error_message=str(e))


def _generate_mock_runner(workspace: Path) -> str:
    """Generate a mock runner that imports and validates the generated code."""

    # Check which files exist to build appropriate imports
    has_model = (workspace / "model.py").exists()
    has_dataset = (workspace / "dataset.py").exists()

    return f'''#!/usr/bin/env python3
"""Mock training runner — validates generated code end-to-end."""

import sys
import os
import traceback

os.chdir("/workspace")
sys.path.insert(0, "/workspace")

print("=" * 60)
print("MOCK TRAINING RUN — Code Validation")
print("=" * 60)

errors = []

# Step 1: Import checks
print("\\n--- Step 1: Import Checks ---")

try:
    import torch
    print(f"✓ PyTorch version: {{torch.__version__}}")
    print(f"✓ CUDA available: {{torch.cuda.is_available()}}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {{e}}")
    errors.append(f"PyTorch import: {{e}}")

{"" if not has_model else """
try:
    from model import *
    print("✓ model.py imports successfully")
except Exception as e:
    print(f"✗ model.py import failed: {{e}}")
    traceback.print_exc()
    errors.append(f"model.py import: {{e}}")
"""}

{"" if not has_dataset else """
try:
    from dataset import *
    print("✓ dataset.py imports successfully")
except Exception as e:
    print(f"✗ dataset.py import failed: {{e}}")
    traceback.print_exc()
    errors.append(f"dataset.py import: {{e}}")
"""}

# Step 2: Run train.py in mock mode (1 batch, 1 epoch)
print("\\n--- Step 2: Mock Training Pass ---")

try:
    # Set environment variable to signal mock mode
    os.environ["NNB_MOCK_RUN"] = "1"
    os.environ["NNB_MAX_BATCHES"] = "2"
    os.environ["NNB_MAX_EPOCHS"] = "1"

    # Execute train.py
    exec(open("/workspace/train.py").read())
    print("\\n✓ Training script executed successfully")
except SystemExit as e:
    if e.code == 0:
        print("\\n✓ Training script completed (exit 0)")
    else:
        print(f"\\n✗ Training script exited with code {{e.code}}")
        errors.append(f"train.py exit code: {{e.code}}")
except Exception as e:
    print(f"\\n✗ Training script failed: {{e}}")
    traceback.print_exc()
    errors.append(f"train.py execution: {{e}}")

# Summary
print("\\n" + "=" * 60)
if errors:
    print(f"✗ MOCK RUN FAILED — {{len(errors)}} error(s)")
    for err in errors:
        print(f"  - {{err}}")
    sys.exit(1)
else:
    print("✓ MOCK RUN PASSED — All checks green!")
    print("=" * 60)
    sys.exit(0)
'''


def _run_with_retries(project: "Project", workspace: Path) -> MockRunResult:  # noqa: F821
    """Run mock training with auto-retry and Gemini debugging."""

    for attempt in range(1, MAX_RETRIES + 1):
        console.print(f"\n🔄 Attempt {attempt}/{MAX_RETRIES}")

        # Run in container
        exit_code, logs = _execute_in_container(project, workspace)

        if exit_code == 0:
            console.print(f"\n[green]✓ Mock run passed on attempt {attempt}![/green]")
            console.print("\n💡 Next step:")
            console.print("  Run: [cyan]nnb train[/cyan]")

            return MockRunResult(
                succeeded=True,
                forward_pass_succeeded=True,
                backward_pass_succeeded=True,
                optimizer_step_succeeded=True,
                retry_count=attempt - 1,
            )

        # Failed — check if it's an infrastructure error (don't waste Gemini API calls)
        console.print(f"\n[yellow]⚠️  Mock run failed on attempt {attempt}[/yellow]")

        # Detect infrastructure errors (Docker issues, not code bugs)
        infra_patterns = ["ImageNotFound", "No such image", "Docker daemon", "docker.errors.APIError"]
        is_infra_error = any(pat in logs for pat in infra_patterns)

        if is_infra_error:
            console.print("\n[red]❌ Infrastructure error (not a code issue)[/red]")
            console.print("💡 Try rebuilding the environment:")
            console.print("  1. Run: [cyan]nnb env build[/cyan]")
            console.print("  2. Then: [cyan]nnb mock-run[/cyan]")
            break

        if attempt < MAX_RETRIES:
            console.print("🤖 Asking Gemini to diagnose and fix...")
            fix_applied = _debug_and_patch(project, workspace, logs, attempt)

            if not fix_applied:
                console.print("[red]❌ Gemini could not generate a fix[/red]")
                break
        else:
            console.print(f"\n[red]❌ Mock run failed after {MAX_RETRIES} attempts[/red]")

    # All retries exhausted
    console.print("\n💡 Manual intervention required:")
    console.print("  1. Check generated code in .nnb/<project>/workspace/")
    console.print("  2. Fix the errors manually")
    console.print("  3. Run: [cyan]nnb mock-run[/cyan]")

    return MockRunResult(
        succeeded=False,
        error_message=f"Mock run failed after {MAX_RETRIES} attempts",
        retry_count=MAX_RETRIES,
    )


def _execute_in_container(
    project: "Project", workspace: Path  # noqa: F821
) -> tuple:
    """Execute mock runner in Docker container. Returns (exit_code, logs).
    
    Uses detached execution + manual log capture to ensure we always
    get full stdout/stderr, even on failure.
    """

    container = get_container(project.project_id)

    data_dir = str(project.project_dir / "data")
    workspace_dir = str(workspace)

    (project.project_dir / "data").mkdir(exist_ok=True)

    console.print("🐳 Starting container...")

    docker_container = None
    try:
        # Start container in detached mode so we can always capture logs
        docker_container = container.start(
            command="python3 /workspace/mock_runner.py",
            detach=True,
            workspace_dir=workspace_dir,
            data_dir=data_dir,
            remove=False,  # Don't auto-remove — we need logs first
        )

        # Wait for container to finish
        console.print("⏳ Waiting for mock run to complete...")
        result = docker_container.wait(timeout=120)
        exit_code = result.get("StatusCode", 1)

        # Capture logs (stdout + stderr)
        logs = docker_container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")

        console.print("\n📋 Mock run output:\n")
        console.print(logs if logs else "[dim]No output captured[/dim]")

        return exit_code, logs

    except Exception as e:
        # Try to get logs even on exception
        logs = ""
        if docker_container:
            try:
                logs = docker_container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            except Exception:
                pass
        
        error_msg = str(e)
        logger.error(f"Container execution failed: {e}", exc_info=True)
        full_logs = f"{logs}\n\nContainer error: {error_msg}" if logs else f"Container error: {error_msg}"
        return 1, full_logs

    finally:
        # Clean up container
        if docker_container:
            try:
                docker_container.remove(force=True)
            except Exception:
                pass


def _debug_and_patch(
    project: "Project",  # noqa: F821
    workspace: Path,
    error_logs: str,
    attempt: int,
) -> bool:
    """Use Gemini to diagnose mock-run failure and patch code files."""

    try:
        gemini = GeminiClient()
        steering_context = SteeringLoader.format_for_prompt("code_generation")

        # Read current code files
        code_files = {}
        for filename in ["model.py", "dataset.py", "train.py", "config.yaml"]:
            filepath = workspace / filename
            if filepath.exists():
                code_files[filename] = filepath.read_text(encoding="utf-8")

        code_files_str = "\n\n".join(
            f"### {name}:\n```python\n{content}\n```"
            for name, content in code_files.items()
        )

        prompt = MOCK_DEBUG_PROMPT.format(
            steering_context=steering_context,
            code_files=code_files_str,
            traceback=error_logs,
            retry_count=attempt,
            max_retries=MAX_RETRIES,
        )

        response = gemini.generate_json(prompt, temperature=0.2)

        root_cause = response.get("root_cause", "Unknown")
        fix_explanation = response.get("fix_explanation", "")
        patched_files = response.get("files", {})

        console.print(f"\n🔍 Root cause: {root_cause}")
        console.print(f"🔧 Fix: {fix_explanation}")

        if not patched_files:
            logger.warning("Gemini returned no file patches")
            return False

        # Apply patches
        for filename, content in patched_files.items():
            filepath = workspace / filename
            filepath.write_text(content, encoding="utf-8")
            console.print(f"  [green]✓[/green] Patched {filename}")

        logger.info(
            f"Applied {len(patched_files)} patches (attempt {attempt}): {root_cause}"
        )

        return True

    except Exception as e:
        logger.error(f"Debug and patch failed: {e}", exc_info=True)
        console.print(f"[red]❌ Gemini debugging failed: {e}[/red]")
        return False
