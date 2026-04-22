# Quick Start Guide

## Prerequisites

Before you start, make sure you have:
- ✅ Python 3.10 or higher installed
- ✅ Docker Desktop installed and running
- ✅ A Gemini API key ([Get one free here](https://makersuite.google.com/app/apikey))

## Installation

### 1. Open a Terminal

**Windows**: PowerShell or Command Prompt  
**Mac/Linux**: Terminal

### 2. Navigate to the Project Directory

```bash
cd "D:\Programming\Project Forge"
```

### 3. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat

# Mac/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

```bash
# Install the package in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
# Check that the command is available
nnb --help
```

You should see:
```
Usage: nnb [OPTIONS] COMMAND [ARGS]...

  CLI Neural Network Builder - Build and train neural networks through
  conversation.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  attach    Attach to running training.
  config    Configuration management.
  data      Data validation commands.
  env       Docker environment commands.
  mock-run  Run mock training pass.
  resume    Resume an existing project.
  start     Start a new neural network project.
  status    Show project status and next action.
  train     Start training.
```

## First Time Setup

### Set Up Your Gemini API Key

The tool will prompt you automatically, but you can also set it up manually:

```bash
nnb config setup
```

Follow the prompts:
1. Get your API key from https://makersuite.google.com/app/apikey
2. Paste it when prompted (input is hidden for security)
3. The tool will test and store it securely

## Start Your First Project

### 1. Start a New Project

```bash
nnb start
```

### 2. Describe Your Project

When prompted, describe what you want to build. For example:

```
I want to build an image classifier to identify different types of flowers.
I have about 5000 images across 5 flower species.
I need it to be reasonably fast for real-time classification.
```

Press Enter twice when done.

### 3. Answer Scoping Questions

Gemini will ask clarifying questions like:
- How many classes do you have?
- What input size do you prefer?
- Do you want to use a pretrained model?

Answer each question.

### 4. Review and Confirm Specification

Review the generated specification and confirm it's correct.

### 5. Review Data Requirements

The tool will generate data requirements. Read them carefully.

### 6. Validate Your Data

```bash
nnb data validate --path /path/to/your/data
```

For example:
```bash
nnb data validate --path "C:\Users\YourName\Documents\flower-data"
```

### 7. Next Steps

The tool will guide you through:
- Building the Docker environment
- Generating training code
- Running a mock training pass
- Training your model
- Setting up inference

## Common Commands

### Check Project Status

```bash
nnb status
```

### Resume a Project

```bash
# List projects (they're in .nnb/ directory)
ls .nnb/

# Resume specific project
nnb resume nnb-20260423-143022-a1b2c3d4
```

### Check API Key Status

```bash
nnb config status
```

### Delete API Key

```bash
nnb config delete-key
```

## Example Session

Here's a complete example session:

```bash
# 1. Navigate to your workspace
cd "D:\Programming\Project Forge"

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Start a new project
nnb start

# Output:
# 🎯 Let's build your neural network!
# 
# Tell me about your project...
# (Type your description, press Enter twice when done)

# 4. Type your description
I want to classify images of cats and dogs.
I have 10,000 images total, split evenly.
I need good accuracy but it doesn't need to be real-time.

# (Press Enter twice)

# 5. Answer questions
# Q: How many classes do you have?
2

# Q: What input image size do you prefer?
224x224

# Q: Do you want to use a pretrained model?
yes

# 6. Confirm specification
# Does this look correct? [Y/n]: y

# 7. Review data requirements
# (Read the requirements)

# 8. Validate your data
nnb data validate --path "C:\Users\YourName\Documents\cat-dog-data"

# 9. Continue with next steps as prompted
```

## Troubleshooting

### "nnb: command not found"

**Solution**: Make sure you've installed the package and activated your virtual environment:
```bash
# Activate venv
.\venv\Scripts\Activate.ps1

# Reinstall if needed
pip install -e .
```

### "No module named 'nnb'"

**Solution**: Install the package:
```bash
pip install -e .
```

### "GEMINI_API_KEY not configured"

**Solution**: Set up your API key:
```bash
nnb config setup
```

### "Docker is not running"

**Solution**: Start Docker Desktop and wait for it to fully start.

### "Could not access keyring"

**Solution**: Use environment variable instead:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"

# Then run your command
nnb start
```

## Tips

### 1. Keep Your Terminal Open

Don't close the terminal while the tool is running. If you need to close it:
- The project state is saved
- You can resume later with `nnb resume <project-id>`

### 2. Check Status Anytime

```bash
nnb status
```

This shows:
- Current project state
- What stage you're on
- What to do next

### 3. Prepare Your Data First

Before starting:
- Organize your data in folders
- Make sure file formats are correct
- Have a rough idea of your dataset size

### 4. Use Descriptive Project Descriptions

The more detail you provide, the better Gemini can help:
- ✅ "Image classifier for 5 flower species, 5000 images, need real-time inference"
- ❌ "Classify flowers"

### 5. Review Generated Requirements

Always read the data requirements document carefully before validating your data.

## Next Steps

Once you're comfortable with the basics:

1. Read the [Architecture Guide](.kiro/steering/architecture.md)
2. Check the [Quick Reference](.kiro/QUICK-REFERENCE.md)
3. Explore the [Steering Documents](.kiro/steering/)

## Getting Help

### In the Tool

```bash
# General help
nnb --help

# Command-specific help
nnb config --help
nnb data --help
```

### Documentation

- [README.md](README.md) - Full documentation
- [API Key Setup](docs/API-KEY-SETUP.md) - Detailed API key guide
- [Project Status](PROJECT-STATUS.md) - Current implementation status

### Common Issues

Check [PROJECT-STATUS.md](PROJECT-STATUS.md) for known issues and workarounds.

---

**Ready to build your neural network? Run `nnb start` to begin! 🚀**
