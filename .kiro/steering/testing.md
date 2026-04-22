---
inclusion: auto
description: Testing requirements for the CLI Neural Network Builder including unit tests, integration tests, and mock run validation.
---

# Testing Requirements

## Test Coverage: 80% Minimum

Test Types (ALL required):
1. **Unit Tests** — Individual functions, utilities, validators
2. **Integration Tests** — Stage transitions, Docker operations, Gemini interactions
3. **E2E Tests** — Complete pipeline runs from start to inference

## Test-Driven Development

MANDATORY workflow for new features:
1. Write test first (RED)
2. Run test - it should FAIL
3. Write minimal implementation (GREEN)
4. Run test - it should PASS
5. Refactor (IMPROVE)
6. Verify coverage (80%+)

## Testing by Layer

### Orchestrator Tests

```python
# Test state transitions
def test_state_transition_from_init_to_scoping():
    project = Project.create()
    assert project.state == State.INIT
    
    project.start_conversation()
    assert project.state == State.SCOPING

# Test command validation
def test_command_requires_correct_state():
    project = Project.create()
    
    with pytest.raises(InvalidStateError):
        project.validate_data()  # Can't validate before spec confirmed
```

### Docker Runtime Tests

```python
# Test container lifecycle
def test_container_build_and_health_check():
    runtime = DockerRuntime(project_id="test-123")
    
    runtime.build_container(dockerfile_path)
    assert runtime.container_exists()
    
    health = runtime.check_health()
    assert health.all_passed()

# Test volume mounts
def test_data_mount_is_read_only():
    runtime = DockerRuntime(project_id="test-123")
    
    with pytest.raises(PermissionError):
        runtime.exec("touch /data/new-file.txt")
```

### Gemini Brain Tests

```python
# Test prompt generation
def test_interviewer_prompt_includes_user_description():
    brain = GeminiBrain()
    prompt = brain.generate_interviewer_prompt(user_desc="Build image classifier")
    
    assert "image classifier" in prompt.lower()
    assert "categorize" in prompt.lower()

# Test output parsing
def test_parse_data_validation_response():
    response = '{"status": "pass", "issues": []}'
    result = GeminiBrain.parse_validation_response(response)
    
    assert result.status == ValidationStatus.PASS
    assert len(result.issues) == 0
```

## Mock Run as Test Gate

The mock run in Stage 6 serves as a critical integration test:

```python
def test_mock_run_validates_end_to_end():
    """Mock run must validate the entire training pipeline."""
    project = Project.load("test-project")
    
    # Generate code
    project.generate_code()
    
    # Run mock
    result = project.run_mock()
    
    assert result.forward_pass_succeeded
    assert result.backward_pass_succeeded
    assert result.optimizer_step_succeeded
    assert not math.isnan(result.loss)
    assert not math.isinf(result.loss)
```

## Integration Test Patterns

### Stage Transition Tests

```python
def test_complete_pipeline_happy_path():
    """Test entire pipeline from init to inference."""
    project = Project.create()
    
    # Stage 0-2
    project.start_conversation()
    project.answer_scoping_questions(answers)
    project.confirm_spec()
    
    # Stage 3-4
    project.generate_data_requirements()
    project.validate_data(data_path)
    
    # Stage 5
    project.build_environment()
    
    # Stage 6
    project.generate_code()
    project.run_mock()
    
    # Stage 7
    project.start_training()
    project.wait_for_training()
    
    # Stage 8
    project.setup_inference()
    
    assert project.state == State.DONE
```

### Error Recovery Tests

```python
def test_mock_run_auto_retry_on_failure():
    """Test that mock run retries on failure."""
    project = Project.load("test-project")
    
    # Inject a fixable error
    inject_shape_mismatch_error(project)
    
    result = project.run_mock(max_retries=3)
    
    assert result.succeeded
    assert result.retry_count > 0
    assert result.retry_count <= 3
```

## Test Data Management

### Synthetic Test Data

```python
def create_synthetic_image_dataset(num_classes=2, samples_per_class=10):
    """Create synthetic image dataset for testing."""
    data_dir = Path("test-data")
    
    for class_idx in range(num_classes):
        class_dir = data_dir / f"class_{class_idx}"
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for sample_idx in range(samples_per_class):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(img).save(class_dir / f"sample_{sample_idx}.jpg")
    
    return data_dir
```

### Test Fixtures

```python
@pytest.fixture
def sample_project():
    """Create a sample project for testing."""
    project = Project.create()
    project.spec = ProjectSpec(
        project_type="vision",
        framework="pytorch",
        task="classification",
        num_classes=2
    )
    yield project
    project.cleanup()

@pytest.fixture
def mock_gemini():
    """Mock Gemini API for testing."""
    with patch("gemini_brain.GeminiClient") as mock:
        mock.return_value.generate.return_value = {
            "status": "pass",
            "issues": []
        }
        yield mock
```

## Performance Tests

```python
def test_data_validation_completes_within_timeout():
    """Ensure data validation doesn't hang."""
    project = Project.load("test-project")
    
    start = time.time()
    project.validate_data(large_dataset_path)
    duration = time.time() - start
    
    assert duration < 60  # Should complete within 1 minute
```

## Test Organization

```
tests/
├── unit/
│   ├── test_orchestrator.py
│   ├── test_docker_runtime.py
│   ├── test_gemini_brain.py
│   └── test_validators.py
├── integration/
│   ├── test_stage_transitions.py
│   ├── test_error_recovery.py
│   └── test_docker_integration.py
├── e2e/
│   ├── test_complete_pipeline.py
│   └── test_resume_workflow.py
└── fixtures/
    ├── sample_data.py
    └── mock_responses.py
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nnb --cov-report=html

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with verbose output
pytest -v

# Run specific test
pytest tests/unit/test_orchestrator.py::test_state_transition
```

## CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: pytest --cov=nnb --cov-report=xml
      
      - name: Check coverage
        run: |
          coverage report --fail-under=80
```

## Best Practices

### DO:
- ✅ Write tests before implementation (TDD)
- ✅ Test error paths, not just happy paths
- ✅ Use fixtures for common setup
- ✅ Mock external dependencies (Gemini API, Docker)
- ✅ Test state transitions explicitly
- ✅ Verify coverage meets 80% threshold

### DON'T:
- ❌ Skip tests for "simple" functions
- ❌ Test implementation details
- ❌ Write flaky tests that sometimes fail
- ❌ Ignore test failures
- ❌ Commit code without running tests
- ❌ Mock everything (integration tests need real Docker)
