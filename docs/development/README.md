# Development Documentation

This folder contains implementation notes, summaries, and technical details for developers working on the CLI Neural Network Builder.

## Implementation Summaries

- **[API-KEY-FEATURE-SUMMARY.md](API-KEY-FEATURE-SUMMARY.md)** - API key management implementation
- **[DATASET-SOURCE-UPDATE.md](DATASET-SOURCE-UPDATE.md)** - Torchvision dataset integration
- **[ENVIRONMENT-BUILD-IMPLEMENTATION.md](ENVIRONMENT-BUILD-IMPLEMENTATION.md)** - Docker environment building (Stage 5)
- **[MOCK-RUN-IMPLEMENTATION.md](MOCK-RUN-IMPLEMENTATION.md)** - Mock training validation (Stage 6)
- **[FIX-SUMMARY.md](FIX-SUMMARY.md)** - Bug fixes and troubleshooting
- **[TORCHVISION-DATASET-EXAMPLE.md](TORCHVISION-DATASET-EXAMPLE.md)** - Torchvision dataset examples

## Steering Documents

- **[STEERING-DOCS-SUMMARY.md](STEERING-DOCS-SUMMARY.md)** - Overview of steering system
- **[STEERING-REINFORCEMENT-SUMMARY.md](STEERING-REINFORCEMENT-SUMMARY.md)** - Steering reinforcement patterns

## Architecture

See the main steering documents in `.kiro/steering/` for:
- Architecture principles
- Coding standards
- Docker patterns
- Security guidelines
- Error handling
- Testing requirements

## Contributing

When adding new features:

1. Follow the architecture in `.kiro/steering/architecture.md`
2. Adhere to coding standards in `.kiro/steering/coding-style.md`
3. Add tests (see `.kiro/steering/testing.md`)
4. Update relevant documentation
5. Create an implementation summary in this folder

## Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=nnb --cov-report=html

# Format code
black nnb tests

# Lint
ruff check nnb tests

# Type check
mypy nnb
```

## Project Structure

```
nnb/
├── cli.py                  # CLI entry point
├── orchestrator/           # Project orchestration
│   ├── project.py         # Project management
│   └── state.py           # State machine
├── stages/                 # Pipeline stages
│   ├── stage_01_conversation.py
│   ├── stage_02_scoping.py
│   ├── stage_03_data_requirements.py
│   ├── stage_04_data_validation.py
│   ├── stage_05_environment.py
│   ├── stage_06_code_generation.py
│   └── stage_07_training.py
├── gemini_brain/          # Gemini API client
├── docker_runtime/        # Docker management
├── models/                # Data models
└── utils/                 # Utilities
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_project.py

# Run with coverage
pytest --cov=nnb --cov-report=html

# View coverage report
open htmlcov/index.html  # Mac
start htmlcov/index.html # Windows
```

## Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG  # Mac/Linux
$env:LOG_LEVEL="DEBUG"  # Windows PowerShell

# Run with verbose output
nnb start --verbose
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Create git tag
5. Build and publish

```bash
# Build
python -m build

# Publish to PyPI
python -m twine upload dist/*
```
