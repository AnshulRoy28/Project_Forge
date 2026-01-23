# 🔥 Forge User Guide

Complete guide to fine-tuning language models with Forge.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Detailed Workflow](#detailed-workflow)
5. [Docker Setup](#docker-setup)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware
- **NVIDIA GPU** with 8GB+ VRAM (RTX 3060 or newer recommended)
- **16GB+ RAM**
- **50GB+ free disk space**

### Software
- **Windows 10/11** or **Linux**
- **Python 3.11+**
- **NVIDIA Driver 535+**
- **Docker Desktop** with NVIDIA Container Toolkit

### API Key
- **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/apikey)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourrepo/forge.git
cd Forge

# Install Forge CLI
pip install -e .

# Initialize (configures API key)
forge init

# Build Docker container for your GPU (auto-detects architecture)
forge docker build
```

---

## Quick Start

### Complete Workflow

```bash
# 1. Put your data in the data folder
mkdir data
# Copy your CSV/JSON/TXT files to ./data/

# 2. Prepare your data (AI-powered)
forge prepare ./data/mydata.csv

# 3. Generate training config
forge plan "I want to create a customer support chatbot"

# 4. Train via Docker container
forge train

# 5. Test your model
forge inference
```

---

## Detailed Workflow

### Step 1: Prepare Your Data

**Supported formats:**
- CSV with `text` or `content` column
- JSON/JSONL with `messages` or `text` field
- Plain text files

**Example CSV:**
```csv
text
"Hello! How can I help you today?"
"I'd like to track my order."
"Sure! Please provide your order number."
```

### Step 2: Preprocess Data

```bash
forge prepare ./data/mydata.csv
```

**What happens:**
1. Gemini analyzes your data structure
2. Generates a custom preprocessing script
3. Creates isolated sandbox environment
4. Auto-installs dependencies with your approval
5. Executes the script safely
6. Auto-fixes errors (up to 3 retries)
7. Outputs `processed_train.jsonl` and `processed_val.jsonl`

### Step 3: Generate Config

```bash
forge plan "I want to create a customer support chatbot"
```

This creates `forge.yaml` with settings optimized for your GPU.

### Step 4: Train via Docker

```bash
forge train
```

**Features:**
- Runs inside GPU-optimized Docker container
- Real-time progress streaming
- Automatic checkpointing
- Self-healing (auto-fixes common errors)


### Step 5: Test Your Model

```bash
forge inference
```

Options:
- `forge inference` - Run via Docker (default)
- `forge inference --ollama` - Use Ollama
- `forge inference --local` - Run on host directly

---

## Docker Setup

### Why Docker?

- **Zero dependency conflicts** - Pre-configured for your GPU
- **Guaranteed GPU support** - Works with sm_120 (Blackwell)
- **Native GPU speed** - No virtualization overhead
- **Reproducible** - Same environment everywhere

### Commands

```bash
# Detect your GPU
forge docker detect

# Build container (auto-detects GPU)
forge docker build

# Train via container (automatic)
forge train

# Interactive shell in container
forge docker shell
```

### Available Containers

| Container | GPUs | CUDA |
|-----------|------|------|
| `forge:blackwell` | RTX 5090/5080/5070 | 12.8 |
| `forge:ada` | RTX 4090/4080/4070 | 12.4 |
| `forge:ampere` | RTX 3090/3080/3070 | 12.1 |
| `forge:hopper` | H100/H200 | 12.4 |

---

## Troubleshooting

### Docker Not Running

```bash
# Check Docker is running
docker info

# If not running, start Docker Desktop
```

### Container Build Fails

```bash
# Rebuild without cache
forge docker build --no-cache
```

### Out of Memory

Edit `forge.yaml`:
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 8
  use_gradient_checkpointing: true
```

### Training Crashes

Forge has self-healing - it will automatically:
1. Detect the error
2. Adjust configuration
3. Retry training

If issues persist, check `checkpoints/` for the latest checkpoint and run:
```bash
forge train --resume
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `forge init` | Initialize environment and API key |
| `forge study <path>` | Analyze dataset with Gemini |
| `forge prepare <path>` | Auto-preprocess data |
| `forge plan "<goal>"` | Generate training config |
| `forge train` | Train via Docker |
| `forge inference` | Test trained model |
| `forge docker detect` | Detect GPU architecture |
| `forge docker build` | Build container |
| `forge docker shell` | Open container shell |

---

Happy training! 🔥
