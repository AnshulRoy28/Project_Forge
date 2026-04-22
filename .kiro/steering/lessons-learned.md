---
inclusion: auto
description: Project-specific patterns, preferences, and lessons learned over time (user-editable)
---

# Lessons Learned

This file captures project-specific patterns, coding preferences, common pitfalls, and architectural decisions that emerge during development of the CLI Neural Network Builder.

**How to use this file:**
1. Document patterns unique to this project
2. Capture team conventions and preferences
3. Record common pitfalls and how to avoid them
4. Track architectural decisions and their rationale
5. Keep it focused on project-specific insights

---

## Project-Specific Patterns

*Document patterns unique to the CLI Neural Network Builder.*

### State Machine Transitions

Always validate state before executing commands:

```python
def validate_data(self):
    if self.state != State.DATA_REQUIRED:
        raise InvalidStateError(
            f"Cannot validate data in state {self.state}. "
            f"Expected state: {State.DATA_REQUIRED}"
        )
    # ... validation logic ...
```

### Gemini Prompt Context

Keep prompts focused on the current stage only:

```python
# WRONG: Include entire conversation history
prompt = f"{full_conversation_history}\n{current_task}"

# CORRECT: Include only relevant context
prompt = f"{confirmed_spec}\n{data_requirements}\n{current_task}"
```

---

## Code Style Preferences

*Document team preferences beyond standard linting rules.*

### CLI Output Formatting

Use consistent emoji and color scheme:

```python
# Success
print("✓ Data validation passed")

# Error
print("❌ Mock run failed")

# Warning
print("⚠️  Class imbalance detected")

# Info
print("💡 Tip: Use --verbose for detailed output")

# Progress
print("🔧 Building Docker container...")
```

### File Organization

Organize by stage, not by type:

```
nnb/
├── stages/
│   ├── stage_01_conversation.py
│   ├── stage_02_scoping.py
│   ├── stage_03_data_requirements.py
│   └── ...
```

---

## Common Pitfalls

*Document mistakes that have been made and how to avoid them.*

### Docker Volume Mount Timing

**Pitfall:** Mounting volumes after container creation fails silently.

**Solution:** Always specify volumes during container creation:

```python
# WRONG: Try to mount after creation
container = docker.create_container(image)
container.mount_volume(data_path)  # Doesn't work!

# CORRECT: Specify volumes at creation
container = docker.create_container(
    image,
    volumes={str(data_path): {"bind": "/data", "mode": "ro"}}
)
```

### Gemini Response Parsing

**Pitfall:** Gemini sometimes returns markdown code blocks instead of raw JSON.

**Solution:** Strip markdown formatting before parsing:

```python
def parse_gemini_json(response: str) -> dict:
    # Remove markdown code blocks
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]  # Remove ```json
    if response.startswith("```"):
        response = response[3:]  # Remove ```
    if response.endswith("```"):
        response = response[:-3]  # Remove ```
    
    return json.loads(response.strip())
```

### State File Corruption

**Pitfall:** Concurrent writes to `state.json` can corrupt the file.

**Solution:** Use file locking:

```python
import fcntl

def save_state(state):
    with open(".nnb/state.json", "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json.dump(state, f, indent=2)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

---

## Architecture Decisions

*Document key architectural decisions and their rationale.*

### Decision: One Container Per Project

**Rationale:** 
- Simplifies lifecycle management
- Enables detached training
- Preserves environment consistency
- Reduces Docker overhead

**Trade-offs:**
- Slightly longer initial build time
- Container must be rebuilt for dependency changes
- But: Much simpler than managing ephemeral containers

### Decision: Volume Mounts Over COPY

**Rationale:**
- Generated files appear on host immediately
- No need to extract artifacts from container
- Enables live editing during development
- Simplifies debugging

**Trade-offs:**
- Requires Docker volume support
- But: All modern Docker installations support this

### Decision: Gemini as Staged Agent

**Rationale:**
- Each stage has focused, specific prompts
- Reduces context window usage
- Produces better outputs than single long conversation
- Enables stage-specific retry logic

**Trade-offs:**
- More complex prompt management
- But: Much better results and easier debugging

### Decision: Hard Gates Between Stages

**Rationale:**
- Prevents cascading failures
- Forces validation at each step
- Makes debugging easier (know exactly where it failed)
- Provides clear user feedback

**Trade-offs:**
- Slightly slower for experienced users
- But: Much safer and more reliable

---

## Performance Optimizations

*Document performance improvements discovered during development.*

### Data Validation Sampling

Instead of validating entire dataset, sample representative subset:

```python
def sample_data(data_path, sample_size=100):
    """Sample data for validation without loading everything."""
    all_files = list(data_path.rglob("*"))
    
    if len(all_files) <= sample_size:
        return all_files
    
    # Stratified sampling by directory
    samples = []
    for dir in data_path.iterdir():
        if dir.is_dir():
            dir_files = list(dir.glob("*"))
            n_samples = max(1, sample_size // len(list(data_path.iterdir())))
            samples.extend(random.sample(dir_files, min(n_samples, len(dir_files))))
    
    return samples[:sample_size]
```

### Lazy Docker Image Loading

Don't pull base images until actually needed:

```python
def build_container(self):
    # Check if base image exists locally
    try:
        docker.images.get(self.base_image)
    except docker.errors.ImageNotFound:
        print(f"📥 Pulling base image {self.base_image}...")
        docker.images.pull(self.base_image)
    
    # Now build
    docker.images.build(...)
```

---

## Testing Insights

*Document testing strategies that work well for this project.*

### Mock Gemini Responses

Use fixtures for consistent Gemini responses in tests:

```python
@pytest.fixture
def mock_gemini_validation_pass():
    return {
        "status": "pass",
        "issues": [],
        "class_distribution": {"class_0": 50, "class_1": 50},
        "estimated_training_time": "30 minutes"
    }
```

### Docker Test Cleanup

Always clean up test containers:

```python
@pytest.fixture
def test_container():
    container = create_test_container()
    yield container
    container.stop()
    container.remove()
```

---

## Notes

- Keep entries concise and actionable
- Remove patterns that are no longer relevant
- Update patterns as the project evolves
- Focus on what's unique to this project
- Add new sections as needed

---

## Future Considerations

*Ideas for future improvements.*

- [ ] Support for distributed training across multiple GPUs
- [ ] Integration with experiment tracking (MLflow, Weights & Biases)
- [ ] Support for model quantization and optimization
- [ ] Cloud deployment integration (AWS SageMaker, GCP AI Platform)
- [ ] Support for AutoML hyperparameter tuning
