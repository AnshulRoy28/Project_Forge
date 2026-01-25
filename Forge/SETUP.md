# 🔥 Forge Setup Guide

> **Docker-based CLI for fine-tuning Small Language Models with Gemini as your AI co-pilot**

Forge uses **Gemini** as an intelligent agent to handle data preprocessing, hyperparameter optimization, and training—all via **Docker containers** with GPU passthrough. Optimized for **RTX 50-series (Blackwell)**, 40-series, 30-series, and datacenter GPUs.

## ✨ Features

- 🧠 **AI-Powered** - Describe your goal in natural language, Gemini handles the rest
- ⚡ **Agentic Preprocessing** - Auto-generates and runs data preparation scripts
- 🔄 **Self-Healing Training** - Automatically fixes errors and retries
- 🐳 **Docker-First** - Training runs in GPU-optimized containers
- 🖥️ **Local Data** - Your data never leaves your machine
- 🎯 **Hardware-Aware** - Auto-detects GPU and selects optimal container
- ⚡ **Unsloth Powered** - 2x faster training, 70% less VRAM
- 🔄 **Container Persistence** - No model redownloading between training runs

## 🚀 Quick Start

### Prerequisites

**Hardware Requirements:**
- **NVIDIA GPU** with 8GB+ VRAM (RTX 3060 or newer recommended)
- **16GB+ RAM**
- **50GB+ free disk space**

**Software Requirements:**
- **Windows 10/11** or **Linux**
- **Python 3.11+**
- **NVIDIA Driver 535+**
- **Docker Desktop** with NVIDIA Container Toolkit

**API Key:**
- **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/apikey)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/AnshulRoy28/Project_Forge.git
cd Project_Forge

# 2. Verify you're in the right directory (should see pyproject.toml)
ls pyproject.toml

# 3. Install Forge CLI
pip install -e .

# 4. Initialize Forge (configure session credentials)
forge init

# 5. Build Docker container for your GPU (auto-detects architecture)
forge docker build

# 6. Generate training config
forge plan "Create a helpful chatbot assistant"

# 7. Prepare dataset (if you have data)
forge prepare ./data/your_data.csv

# 8. Start training
forge train
```

## 🐳 GPU Containers

Forge automatically detects your GPU and builds the optimal container:

| Container | GPU Support | CUDA Version |
|-----------|-------------|--------------|
| `forge:blackwell` | RTX 5090/5080/5070 | 12.8 |
| `forge:ada` | RTX 4090/4080/4070 | 12.4 |
| `forge:ampere` | RTX 3090/3080/3070 | 12.1 |
| `forge:hopper` | H100/H200 | 12.4 |

GPU is **auto-detected** - just run `forge docker build` and it picks the right one.

## 📋 Commands Reference

| Command | Description |
|---------|-------------|
| `forge init` | Setup environment and configure API key |
| `forge study <path>` | Analyze dataset quality with Gemini |
| `forge prepare <path>` | **Agentic preprocessing** - auto-generate & run scripts |
| `forge plan "<goal>"` | Generate hardware-optimized training config |
| `forge train` | Execute training via Docker (self-healing) |
| `forge inference` | Interactive chat with trained model |
| `forge docker build` | Build container for your GPU |
| `forge docker detect` | Show detected GPU architecture |

## 🔄 Container Persistence & Model Caching

Forge uses persistent Docker containers with HuggingFace cache mounting for **instant training restarts**:

### Container Management Commands
- `forge container status` - Check persistent container status
- `forge container list` - Show all Forge containers
- `forge container cleanup` - Remove persistent containers
- `forge container logs` - View container logs

### How Model Caching Works
1. **First Training Run**: Creates persistent container and mounts HuggingFace cache (~5 minutes)
2. **Subsequent Runs**: Reuses cached model files (loads in ~10-20 seconds instead of 2-5 minutes)
3. **Different Model**: Prompts for container replacement if model mismatch detected
4. **Failed Training**: Automatically cleans up failed containers

### Benefits
✅ **No model redownloading** - Files persist in `~/.cache/huggingface`  
✅ **Works across container restarts** - Cache is on host filesystem  
✅ **Simple and reliable** - Uses standard HuggingFace caching  
✅ **Shared across projects** - All forge projects share the same cache  
✅ **Automatic cleanup** - Failed containers don't accumulate  

## 🎯 Hardware-Aware Optimization

Forge automatically detects your GPU and optimizes training:

```
📝 Planning training for: "customer service chatbot"

  GPU: NVIDIA GeForce RTX 5080
  VRAM: 16.0 GB
  Compute: 12.0 (blackwell)
  BF16: ✓ Supported
  FP8: ✓ Supported

🎯 Hardware Optimization:
  Architecture: blackwell
  Precision: BF16
  Recommended Model: Gemma 2B or 9B (4-bit)
```

## ⚡ Agentic Preprocessing

```bash
forge prepare ./data/english_support.csv
```

**What happens:**
1. 📊 Gemini analyzes your dataset structure
2. 🧠 Generates a custom preprocessing script
3. 📦 Creates isolated sandbox venv
4. 📥 Auto-installs dependencies with your permission
5. ▶️ Executes the script safely
6. 🔄 **Self-heals** if errors occur (up to 3 retries)
7. 🧹 Cleans up sandbox, keeps processed data

**Output:** `./data/processed_train.jsonl` and `./data/processed_val.jsonl`

## 🔄 Self-Healing Training

Training errors are automatically diagnosed and fixed:

```
Training fails → Gemini diagnoses → Config adjusted → Retry
```

Auto-handled errors:
- **BFloat16 issues** - Switches to FP16
- **Out of memory** - Reduces batch size, smaller model
- **Model not found** - Tries alternative models
- **Gated models** - Prompts for HuggingFace login

## 🔒 Security Model

**🔒 Session-Based Security**: Forge uses session-based credential storage for maximum security:

- **No Persistent Storage**: API keys are stored only in memory for the current session
- **Automatic Cleanup**: Credentials are automatically cleared when the terminal closes
- **Fresh Start**: Each new terminal session requires re-entering credentials
- **Zero Risk**: No sensitive data persists on disk or across sessions

### Security Benefits
✅ **No Keyring Dependencies**: Removed system keyring storage  
✅ **Session Isolation**: Each terminal session is independent  
✅ **Automatic Cleanup**: Credentials cleared on session end  
✅ **Zero Persistence**: No sensitive data stored on disk  
✅ **Fresh Authentication**: Always requires explicit credential entry  
✅ **Container Isolation**: Models cached in isolated Docker containers  

## 📁 Project Structure

```
your-project/
├── data/
│   ├── raw_data.csv
│   ├── processed_train.jsonl
│   └── processed_val.jsonl
├── output/             # Trained models
├── checkpoints/        # Training checkpoints
└── forge.yaml          # Training config
```

## 🛠️ Troubleshooting

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

### "Session credentials required"
- Run `forge init` to configure API keys for the current session
- Or run `forge login` to update existing session credentials

### "neither 'setup.py' nor 'pyproject.toml' found"
- Make sure you're in the Forge directory: `cd Project_Forge`
- Verify pyproject.toml exists: `ls pyproject.toml`
- The file should be at the root level of the cloned repository

## 🛠️ Technical Stack

- **Training**: Unsloth + TRL + Transformers
- **Quantization**: BitsAndBytes (4-bit/8-bit)
- **AI Backend**: Google Gemini
- **Containers**: Docker with NVIDIA GPU passthrough
- **CLI**: Typer + Rich
- **Caching**: HuggingFace Hub cache with Docker volume mounting

## 📄 License

MIT

---

Happy training! 🔥