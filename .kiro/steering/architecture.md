---
inclusion: auto
description: Architecture principles and patterns for the CLI Neural Network Builder project.
---

# Architecture Principles

## Three-Layer Architecture

The tool has three distinct layers that MUST remain decoupled:

### 1. Orchestrator (CLI)
- User-facing layer
- Manages conversation state
- Talks to Gemini
- Drives the pipeline forward stage by stage
- **NEVER** executes training code directly

### 2. Docker Runtime
- Isolated container per project
- All environment setup, code execution, mock runs, training, and inference
- The host machine is never touched beyond Docker itself
- Volume mounts as the integration boundary

### 3. Gemini Brain
- Not a single call but a stateful agent across the pipeline
- Each stage has its own prompt role:
  - Stage 1-2: Interviewer
  - Stage 3-4: Analyst/Validator
  - Stage 5: Environment Engineer
  - Stage 6: Code Engineer
  - Stage 7: Training Monitor
  - Stage 8: Packager

## State Machine Pattern

Every project lives in one of these states, tracked in `.nnb/state.json`:

```
INIT → SCOPING → SPEC_CONFIRMED → DATA_REQUIRED → DATA_VALIDATED
     → ENV_BUILDING → ENV_READY → MOCK_RUNNING → MOCK_PASSED
     → TRAINING → TRAINING_COMPLETE → INFERENCE_READY → DONE
```

**CRITICAL RULES:**
- Any command checks the current state first
- `nnb status` always tells the user exactly where they are
- No stage can be skipped
- Failed stages must be resolved before proceeding

## Container Lifecycle

- **One container per project**, not one per command
- Container is built once in Stage 5 and reused
- Rebuilding is explicit (`nnb env rebuild`)
- Container is fully reproducible via generated Dockerfile

## Volume Mounts as Integration Boundary

```
Host                          Container
.nnb/workspace/      ←→      /workspace/     (read-write)
./my-data/           ←→      /data/          (read-only)
```

- Host never runs Python or touches training code
- Container does all execution
- Files flow exclusively through mounted directories

## Gemini as Staged Agent

Each stage sends Gemini a purpose-built prompt with exactly the context it needs:

```python
# WRONG: Single long conversation
gemini.chat("Do everything for my project...")

# CORRECT: Staged prompts
gemini.interview(user_description)  # Stage 1-2
gemini.validate_data(data_snapshot)  # Stage 4
gemini.generate_code(spec, framework)  # Stage 6
```

## Steering Document as Living Artifact

- Auto-initialized in Stage 0
- Updated automatically as conversation progresses
- By Stage 6, it's a complete, structured description
- Gemini uses it as ground truth for every decision

## Resilience Patterns

### Checkpointing
- Save state after every stage completion
- Enable `nnb resume <project-id>` at any point
- Conversation history persisted after every turn

### Detached Execution
- Training runs as detached container process
- Closing terminal doesn't kill training
- `nnb attach <project-id>` to reattach

### Automatic Recovery
- Mock run failures trigger auto-fix (up to 3 retries)
- Training crashes invoke Gemini for diagnosis
- Clear error messages with actionable fixes

## Hard Gates

These are blocking checkpoints that prevent progression:

1. **Spec Confirmation** (Stage 2 → 3): User must confirm spec
2. **Data Validation** (Stage 4 → 5): Data must pass validation
3. **Container Health** (Stage 5 → 6): Container must be healthy
4. **Mock Run** (Stage 6 → 7): Mock run must pass
5. **Training Completion** (Stage 7 → 8): Training must complete successfully

## Separation of Concerns

| Concern | Owner | Location |
|---------|-------|----------|
| User interaction | Orchestrator | CLI commands |
| State management | Orchestrator | `.nnb/state.json` |
| LLM communication | Gemini Brain | `gemini_brain/` |
| Code execution | Docker Runtime | Container |
| Data storage | Host filesystem | `.nnb/` directory |
| Training artifacts | Container → Host | Volume mount |
