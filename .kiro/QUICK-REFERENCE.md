# CLI Neural Network Builder - Quick Reference

> Fast lookup for common patterns and decisions

## 🏗️ Architecture at a Glance

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator (CLI)                    │
│  • User interaction                                      │
│  • State management                                      │
│  • Command routing                                       │
└────────────────┬────────────────────────────────────────┘
                 │
                 ├──────────────┐
                 │              │
        ┌────────▼────────┐    ┌▼──────────────────┐
        │  Gemini Brain   │    │  Docker Runtime   │
        │  • Interviewer  │    │  • Container mgmt │
        │  • Validator    │    │  • Code execution │
        │  • Engineer     │    │  • Training       │
        │  • Monitor      │    │  • Inference      │
        └─────────────────┘    └───────────────────┘
```

## 📊 State Machine

```
INIT → SCOPING → SPEC_CONFIRMED → DATA_REQUIRED → DATA_VALIDATED
     → ENV_BUILDING → ENV_READY → MOCK_RUNNING → MOCK_PASSED
     → TRAINING → TRAINING_COMPLETE → INFERENCE_READY → DONE
```

## 🎯 Hard Gates (Cannot Skip)

1. **Spec Confirmation** (Stage 2 → 3)
2. **Data Validation** (Stage 4 → 5)
3. **Container Health** (Stage 5 → 6)
4. **Mock Run Pass** (Stage 6 → 7)
5. **Training Complete** (Stage 7 → 8)

## 🐳 Docker Patterns

### Container Lifecycle
```python
# ONE container per project (not per command)
container = docker.get_or_create(f"nnb-{project_id}")
```

### Volume Mounts
```python
volumes = {
    str(data_path): {"bind": "/data", "mode": "ro"},      # READ-ONLY
    str(workspace): {"bind": "/workspace", "mode": "rw"}  # READ-WRITE
}
```

### Health Check
```python
checks = [
    "Python version",
    "GPU availability",
    "Framework import",
    "Data mount accessible"
]
```

## 🤖 Gemini Prompt Pattern

```python
# Stage-specific, minimal context
prompt = f"""
You are a {role} for this stage.

Relevant context:
{minimal_context_only}

Your task:
{specific_task}

Output format:
{expected_format}
"""
```

## ❌ Error Handling

### Error Categories
```python
UserError          # Recoverable, user's fault
SystemError        # Potentially recoverable, retry
InternalError      # Not recoverable, bug
```

### Error Response Pattern
```python
print(f"❌ {error.message}")
print(f"💡 Fix: {error.fix}")
logger.error(f"Details: {error}", exc_info=True)
sys.exit(1)
```

## 🧪 Testing

### Coverage Requirement
```bash
pytest --cov=nnb --cov-report=html
# Must be ≥ 80%
```

### Test Types
- **Unit**: Individual functions
- **Integration**: Stage transitions, Docker ops
- **E2E**: Complete pipeline runs

### TDD Workflow
```
1. Write test (RED)
2. Run test - should FAIL
3. Write code (GREEN)
4. Run test - should PASS
5. Refactor (IMPROVE)
```

## 🔒 Security Checklist

```python
# Before ANY commit:
✓ No hardcoded API keys
✓ All inputs validated
✓ Path traversal prevented
✓ Command injection prevented
✓ Data mount is read-only
✓ Secrets in env vars
✓ Logs sanitized
```

## 📝 Code Style

### Immutability
```python
# WRONG
def update(config, key, val):
    config[key] = val  # MUTATION!
    return config

# CORRECT
def update(config, key, val):
    return {**config, key: val}
```

### File Size
- Target: 200-400 lines
- Maximum: 800 lines
- Extract utilities if larger

### Error Messages
```python
# User-facing
print("❌ Data validation failed")
print("💡 Fix: Check file format")

# Logging
logger.error("Validation failed", exc_info=True)
```

## 🎨 CLI Output

```python
✓  Success
❌ Error
⚠️  Warning
💡 Tip
🔧 Working
📋 Info
📁 File/Path
🆔 ID
```

## 📂 Project Structure

```
.nnb/
├── state.json              # Current state
├── steering-doc.yaml       # Project specification
├── data-requirements.md    # Data requirements
├── data-manifest.json      # Validated data info
├── training-report.md      # Training results
├── workspace/              # Generated code & artifacts
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── config.yaml
│   └── checkpoints/
└── logs/                   # All logs
    ├── orchestrator.log
    ├── docker.log
    └── training.log
```

## 🚀 Common Commands

```bash
# Start new project
nnb start

# Resume existing
nnb resume <project-id>

# Check status
nnb status

# Validate data
nnb data validate --path ./data

# Build environment
nnb env build

# Run mock
nnb mock-run

# Start training
nnb train

# Attach to training
nnb attach

# Check progress
nnb check

# Test inference
nnb inference test
```

## 🔍 Debugging

### Check State
```bash
cat .nnb/state.json
```

### Check Logs
```bash
tail -f .nnb/logs/orchestrator.log
tail -f .nnb/logs/training.log
```

### Container Shell
```bash
nnb env shell
```

### Container Logs
```bash
docker logs nnb-<project-id>
```

## 📚 Document Quick Links

| Need | Document |
|------|----------|
| Architecture overview | architecture.md |
| Stage details | stage-workflow.md |
| Docker help | docker-patterns.md |
| Gemini prompts | gemini-prompts.md |
| Code standards | coding-style.md |
| Error handling | error-handling.md |
| Security | security.md |
| Testing | testing.md |
| Project patterns | lessons-learned.md |

## 💡 Common Pitfalls

### 1. Skipping Mock Run
**Problem**: Training fails with cryptic errors  
**Solution**: Always run mock-run before training

### 2. Mutable Data Structures
**Problem**: Hidden side effects, hard-to-debug issues  
**Solution**: Use immutable patterns everywhere

### 3. Generic Error Messages
**Problem**: Users don't know how to fix issues  
**Solution**: Always include actionable fix instructions

### 4. Missing State Validation
**Problem**: Commands run in wrong state  
**Solution**: Always check state before executing

### 5. Hardcoded Values
**Problem**: Not configurable, hard to test  
**Solution**: Use config files or environment variables

## 🎓 Learning Path

### Day 1: Understand Architecture
1. Read architecture.md
2. Review stage-workflow.md
3. Understand state machine

### Day 2: Learn Docker Patterns
1. Read docker-patterns.md
2. Practice container operations
3. Understand volume mounts

### Day 3: Master Gemini Integration
1. Read gemini-prompts.md
2. Study prompt patterns
3. Practice response parsing

### Day 4: Code Quality
1. Read coding-style.md
2. Read error-handling.md
3. Read testing.md

### Day 5: Security & Best Practices
1. Read security.md
2. Review lessons-learned.md
3. Start contributing!

---

**Pro Tip**: Keep this file open while coding for quick reference!
