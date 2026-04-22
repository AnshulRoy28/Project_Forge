---
inclusion: auto
description: Core coding style rules for the CLI Neural Network Builder project including immutability, error handling, and code quality standards.
---

# Coding Style

## Immutability (CRITICAL)

ALWAYS create new objects, NEVER mutate existing ones:

```python
# WRONG: Mutation
def update_config(config, key, value):
    config[key] = value  # MUTATION!
    return config

# CORRECT: Immutability
def update_config(config, key, value):
    return {**config, key: value}
```

Rationale: Immutable data prevents hidden side effects, makes debugging easier, and enables safe concurrency in Docker environments.

## File Organization

MANY SMALL FILES > FEW LARGE FILES:
- High cohesion, low coupling
- 200-400 lines typical, 800 max
- Extract utilities from large modules
- Organize by stage/domain, not by type

### Recommended Structure
```
nnb/
├── orchestrator/      # CLI layer
├── docker_runtime/    # Container management
├── gemini_brain/      # LLM integration
├── stages/            # Stage implementations
├── utils/             # Shared utilities
└── schemas/           # Data validation schemas
```

## Error Handling

ALWAYS handle errors comprehensively:
- Handle errors explicitly at every stage
- Provide user-friendly error messages in CLI output
- Log detailed error context to `.nnb/logs/`
- Never silently swallow errors
- Always provide actionable next steps

```python
# CORRECT: Comprehensive error handling
try:
    result = validate_data(data_path)
except DataValidationError as e:
    logger.error(f"Data validation failed: {e}", exc_info=True)
    print(f"❌ Data validation failed: {e.user_message}")
    print(f"💡 Fix: {e.suggested_fix}")
    sys.exit(1)
```

## Input Validation

ALWAYS validate at system boundaries:
- Validate all user input before processing
- Use schema-based validation (Pydantic, Zod)
- Fail fast with clear error messages
- Never trust external data (user input, file content, LLM output)

## Code Quality Checklist

Before marking work complete:
- [ ] Code is readable and well-named
- [ ] Functions are small (<50 lines)
- [ ] Files are focused (<800 lines)
- [ ] No deep nesting (>4 levels)
- [ ] Proper error handling with user-friendly messages
- [ ] No hardcoded values (use constants or config)
- [ ] No mutation (immutable patterns used)
- [ ] All paths logged to `.nnb/logs/`

## Console Output

- Use rich formatting for CLI output (colors, emojis, progress bars)
- Clear visual hierarchy: ✓ success, ❌ error, 💡 tip, ⚠️ warning
- Show progress for long-running operations
- Always indicate current stage and next action
