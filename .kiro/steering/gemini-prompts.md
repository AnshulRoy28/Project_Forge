---
inclusion: auto
description: Guidelines for crafting Gemini prompts for each stage of the pipeline.
---

# Gemini Prompt Engineering

## Core Principles

1. **Role-specific prompts** — Each stage has a distinct role
2. **Minimal context** — Only include what's needed for that stage
3. **Structured output** — Request specific formats (JSON, YAML, Markdown)
4. **Iterative refinement** — Build on previous stage outputs
5. **Error recovery** — Include traceback context for debugging

## Stage-Specific Prompt Patterns

### Stage 1-2: Interviewer Role

```python
INTERVIEWER_PROMPT = """
You are an expert ML engineer conducting an intake interview.

The user has described their project:
{user_description}

Your task:
1. Categorize the project type (vision, NLP, tabular, time-series, multimodal, RL, other)
2. Identify ambiguities that need clarification
3. Generate 3-5 focused questions to resolve gaps

Output format:
{
  "project_type": "...",
  "complexity_estimate": "low|medium|high",
  "questions": [
    {"topic": "...", "question": "...", "why_needed": "..."}
  ]
}
"""
```

### Stage 3: Data Requirements Analyst

```python
DATA_REQUIREMENTS_PROMPT = """
You are a data requirements specialist.

Project specification:
{confirmed_spec}

Generate a comprehensive Data Requirements Report covering:
1. Format (images, CSV, parquet, JSONL, audio, etc.)
2. Minimum dataset size
3. Recommended dataset size
4. Required folder/label structure
5. Preprocessing you'll handle automatically
6. Preprocessing the user must do
7. Gotchas (class imbalance thresholds, resolution, encoding)

Output as Markdown with clear sections and examples.
"""
```

### Stage 4: Data Validator

```python
DATA_VALIDATION_PROMPT = """
You are a data validation expert.

Data requirements:
{data_requirements}

Data sample:
{data_snapshot}

Validate:
1. Structure matches required format
2. Labels are present and consistent
3. No corrupt files (provide examples if found)
4. Class distribution (flag severe imbalance)
5. Estimated training time based on size

Output format:
{
  "status": "pass|warn|fail",
  "issues": [
    {"severity": "error|warning", "message": "...", "fix": "..."}
  ],
  "class_distribution": {...},
  "estimated_training_time": "..."
}
"""
```

### Stage 5: Environment Engineer

```python
DOCKERFILE_GENERATION_PROMPT = """
You are a Docker and ML environment specialist.

Project specification:
{confirmed_spec}

Generate a Dockerfile that:
1. Selects appropriate base image (PyTorch CUDA, TensorFlow, JAX)
2. Pins Python version
3. Installs ONLY required dependencies (no bloat)
4. Sets up working directory at /workspace
5. Expects data mount at /data

Output the complete Dockerfile as a code block.
"""
```

### Stage 6: Code Engineer

```python
CODE_GENERATION_PROMPT = """
You are an expert ML engineer writing production-quality training code.

Project specification:
{confirmed_spec}

Data manifest:
{data_manifest}

Steering document conventions:
{steering_doc}

Generate:
1. model.py — architecture definition with clear docstrings
2. dataset.py — data loading and preprocessing pipeline
3. train.py — training loop with checkpointing
4. config.yaml — all hyperparameters externalized
5. requirements.txt — minimal dependencies

Code must:
- Follow steering document conventions
- Include comprehensive error handling
- Log all important events
- Save checkpoints every N epochs
- Be immediately runnable

Output each file as a separate code block with filename header.
"""
```

### Stage 6: Mock Run Debugger

```python
MOCK_RUN_DEBUG_PROMPT = """
You are debugging a failed mock run.

Generated code:
{code_files}

Error traceback:
{traceback}

Your task:
1. Identify the root cause
2. Generate a fix
3. Specify which file(s) to patch

Output format:
{
  "root_cause": "...",
  "fix_explanation": "...",
  "patches": [
    {"file": "model.py", "old_code": "...", "new_code": "..."}
  ]
}
"""
```

### Stage 7: Training Monitor

```python
TRAINING_REVIEW_PROMPT = """
You are reviewing completed training.

Training logs:
{training_logs}

Final metrics:
{final_metrics}

Analyze:
1. Did training converge?
2. Any warning signs (overfitting, underfitting, instability)?
3. Recommendations for improvement

Output as a brief Markdown report (3-5 paragraphs).
"""
```

### Stage 8: Inference Packager

```python
INFERENCE_GENERATION_PROMPT = """
You are packaging a trained model for inference.

Project specification:
{confirmed_spec}

Best checkpoint:
{checkpoint_path}

Generate:
1. inference.py — standalone inference script with simple API
2. server.py — FastAPI wrapper (if deployment flagged)
3. Export to ONNX/TFLite (if flagged)
4. inference-requirements.txt — minimal dependencies

Code must:
- Load checkpoint correctly
- Handle input preprocessing
- Return predictions in expected format
- Include error handling

Output each file as a separate code block.
"""
```

## Prompt Best Practices

### DO:
- ✅ Specify exact output format (JSON schema, Markdown structure)
- ✅ Include relevant context from previous stages
- ✅ Request explanations alongside code
- ✅ Set clear success criteria
- ✅ Include examples when helpful

### DON'T:
- ❌ Include entire conversation history (too much context)
- ❌ Ask open-ended questions without structure
- ❌ Mix multiple roles in one prompt
- ❌ Assume Gemini remembers previous stages (always include context)
- ❌ Request code without specifying conventions

## Error Recovery Pattern

When Gemini needs to fix code:

```python
ERROR_RECOVERY_PROMPT = """
Previous attempt failed.

Original task:
{original_task}

Generated code:
{generated_code}

Error:
{error_traceback}

Attempt {retry_count} of 3.

Analyze the error and generate a corrected version.
Focus ONLY on fixing the specific error, don't refactor unrelated code.

Output the complete corrected file.
"""
```

## Context Management

### Minimal Context Principle

Only include what's needed:

| Stage | Required Context |
|-------|------------------|
| 1-2 | User description |
| 3 | Confirmed spec |
| 4 | Data requirements + data sample |
| 5 | Confirmed spec |
| 6 | Spec + data manifest + steering doc |
| 7 | Training logs + metrics |
| 8 | Spec + checkpoint path |

### Steering Document Integration

Always include relevant sections from the steering document:

```python
prompt = f"""
{base_prompt}

Project conventions (from steering document):
{steering_doc.get_relevant_sections(stage)}
"""
```

## Output Parsing

Always validate Gemini's output:

```python
def parse_gemini_response(response, expected_format):
    try:
        if expected_format == "json":
            return json.loads(response)
        elif expected_format == "yaml":
            return yaml.safe_load(response)
        else:
            return response
    except Exception as e:
        logger.error(f"Failed to parse Gemini response: {e}")
        raise GeminiOutputError(f"Invalid format: {e}")
```
