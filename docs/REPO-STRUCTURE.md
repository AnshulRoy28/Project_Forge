# Repository Structure

Clean, organized structure for the CLI Neural Network Builder project.

## Root Directory

```
cli-neural-network-builder/
├── README.md              # Main documentation
├── QUICKSTART.md          # Quick start guide
├── RUN-ME-FIRST.md        # Simplest setup guide
├── pyproject.toml         # Project configuration
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── .env.example          # Environment variable template
└── .gitignore            # Git ignore rules
```

## Source Code (`nnb/`)

```
nnb/
├── cli.py                 # CLI entry point
├── orchestrator/          # Project orchestration
│   ├── project.py        # Project management
│   └── state.py          # State machine
├── stages/                # Pipeline stages (1-7)
│   ├── stage_01_conversation.py
│   ├── stage_02_scoping.py
│   ├── stage_03_data_requirements.py
│   ├── stage_04_data_validation.py
│   ├── stage_05_environment.py
│   ├── stage_06_code_generation.py
│   └── stage_07_training.py
├── gemini_brain/         # Gemini API client
│   └── client.py
├── docker_runtime/       # Docker management
│   └── container.py
├── models/               # Data models (Pydantic)
│   └── project_spec.py
└── utils/                # Utilities
    ├── api_key_manager.py
    ├── id_generator.py
    ├── logging.py
    └── steering_loader.py
```

## Documentation (`docs/`)

```
docs/
├── API-KEY-SETUP.md      # API key configuration guide
├── SECURITY.md           # Security best practices
├── REPO-STRUCTURE.md     # This file
└── development/          # Development documentation
    ├── README.md         # Development guide
    ├── API-KEY-FEATURE-SUMMARY.md
    ├── DATASET-SOURCE-UPDATE.md
    ├── ENVIRONMENT-BUILD-IMPLEMENTATION.md
    ├── MOCK-RUN-IMPLEMENTATION.md
    ├── FIX-SUMMARY.md
    ├── STEERING-DOCS-SUMMARY.md
    ├── STEERING-REINFORCEMENT-SUMMARY.md
    └── TORCHVISION-DATASET-EXAMPLE.md
```

## Tests (`tests/`)

```
tests/
├── test_api_key_manager.py
├── test_project.py
└── test_state_machine.py
```

## Steering Documents (`.kiro/steering/`)

```
.kiro/steering/
├── CRITICAL-RULES.md     # Highest priority rules
├── architecture.md       # System architecture
├── coding-style.md       # Code standards
├── docker-patterns.md    # Docker best practices
├── error-handling.md     # Error handling patterns
├── gemini-prompts.md     # Prompt engineering
├── lessons-learned.md    # Project learnings
├── security.md           # Security guidelines
├── stage-workflow.md     # Stage details
└── testing.md            # Testing requirements
```

## Project Workspaces (`.nnb/`)

Each project gets its own workspace:

```
.nnb/
└── nnb-YYYYMMDD-HHMMSS-<id>/
    ├── state.json              # Current state
    ├── steering-doc.yaml       # Project spec
    ├── data-requirements.md    # Data requirements
    ├── Dockerfile              # Generated Dockerfile
    ├── workspace/              # Generated code
    │   ├── mock_train.py
    │   ├── train.py
    │   ├── model.py
    │   └── checkpoints/
    ├── data/                   # Dataset storage
    └── logs/                   # Execution logs
```

## Generated Files (Ignored by Git)

```
# Python artifacts
__pycache__/
*.pyc
*.egg-info/

# Virtual environment
venv/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Project workspaces
.nnb/

# IDE
.vscode/
.idea/
```

## File Organization Principles

### User-Facing Documentation
- **Root level**: Quick access to essential guides
- **docs/**: Detailed user documentation

### Development Documentation
- **docs/development/**: Implementation notes, summaries
- **Not in root**: Keeps root clean for users

### Code Organization
- **nnb/**: All source code
- **tests/**: All test code
- **Separation of concerns**: Each module has clear responsibility

### Configuration
- **Root level**: Project config (pyproject.toml, requirements.txt)
- **.kiro/**: Steering documents and AI configuration

## Adding New Files

### User Documentation
→ Add to `docs/` or root (if essential)

### Development Notes
→ Add to `docs/development/`

### Implementation Summaries
→ Add to `docs/development/` with descriptive name

### Code
→ Add to appropriate `nnb/` subdirectory

### Tests
→ Add to `tests/` with `test_` prefix

## Cleanup Guidelines

### Keep Root Clean
- Only essential user-facing files
- No temporary files
- No implementation notes

### Organize by Audience
- **Users**: Root + docs/
- **Developers**: docs/development/ + .kiro/steering/
- **Generated**: .nnb/ (gitignored)

### Regular Maintenance
- Move summaries to docs/development/
- Delete obsolete files
- Update documentation links
- Keep .gitignore current
