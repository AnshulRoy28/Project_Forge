# CLI Neural Network Builder (nnb)

Build and train neural networks through natural conversation with AI.

## Overview

The CLI Neural Network Builder is a tool that guides you through the entire machine learning pipeline - from project scoping to trained model - using conversational AI and Docker containers.

## Features

- 🗣️ **Conversational Interface** - Describe your project in plain language
- 🐳 **Isolated Environments** - Each project runs in its own Docker container
- 🤖 **AI-Powered** - Gemini guides you through every stage
- 📊 **Data Validation** - Automatic validation of your training data
- 🔄 **State Management** - Resume projects at any time
- ✅ **Mock Run Validation** - Test your code before training
- 📈 **Training Monitoring** - Track training progress in real-time
- 🚀 **Inference Ready** - Get deployment-ready inference code

## Installation

### Prerequisites

- Python 3.10 or higher
- Docker Desktop installed and running
- Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Install

```bash
# Clone the repository
git clone <repository-url>
cd cli-neural-network-builder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Configure API Key

The tool will prompt you to set up your API key on first use, or you can set it up manually:

```bash
# Interactive setup (recommended)
nnb config setup

# Check status
nnb config status

# Delete key (if needed)
nnb config delete-key
```

Your API key is stored securely in your system keyring (like passwords in your browser or password manager). It's never stored in plain text files.

## Quick Start

```bash
# Start a new project (will prompt for API key if not configured)
nnb start

# Resume an existing project
nnb resume <project-id>

# Check project status
nnb status

# Validate your data
nnb data validate --path ./my-data

# Build Docker environment
nnb env build

# Run mock training
nnb mock-run

# Start training
nnb train

# Test inference
nnb inference test
```

## Configuration Commands

```bash
# Set up API key
nnb config setup

# Check API key status
nnb config status

# Delete stored API key
nnb config delete-key
```

## The Pipeline

The tool guides you through 9 stages:

1. **Project Init** - Create project workspace
2. **Conversation** - Describe your project
3. **Scoping** - Answer clarifying questions
4. **Data Requirements** - Get data specifications
5. **Data Validation** - Validate your dataset
6. **Environment Setup** - Build Docker container
7. **Code Generation** - Generate training code + mock run
8. **Training** - Train your model
9. **Inference Setup** - Package for deployment

## Project Structure

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
```

## Documentation

- [Quick Start Guide](QUICKSTART.md) - Get started in 5 minutes
- [Run Me First](RUN-ME-FIRST.md) - Simplest setup guide
- [API Key Setup](docs/API-KEY-SETUP.md) - Detailed API key configuration
- [Security Guide](docs/SECURITY.md) - Security best practices

### Architecture & Development

- [Architecture](.kiro/steering/architecture.md) - System design and principles
- [Critical Rules](.kiro/steering/CRITICAL-RULES.md) - **Must-follow rules (highest priority)**
- [Stage Workflow](.kiro/steering/stage-workflow.md) - Detailed stage descriptions
- [Development Docs](docs/development/) - Implementation notes and guides

### Steering Documents

The project uses **steering documents** to ensure consistent, high-quality code generation. These documents are automatically included in every Gemini prompt:

- **CRITICAL-RULES.md** - Non-negotiable rules (highest priority)
- **architecture.md** - Three-layer architecture, state machine
- **coding-style.md** - Immutability, error handling, code quality
- **docker-patterns.md** - Container management patterns
- **security.md** - Security best practices
- **error-handling.md** - Error handling patterns
- **testing.md** - Testing requirements

## Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=nnb --cov-report=html

# Format code
black nnb tests

# Lint code
ruff check nnb tests

# Type check
mypy nnb
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please read our contributing guidelines first.
