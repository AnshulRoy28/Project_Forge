# 🔥 Forge

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

## 🚀 Quick Start

```bash
# 1. Install Forge CLI
cd Forge
pip install -e .
forge init

# 2. Build Docker container for your GPU
forge docker build

# 3. Prepare your data
forge prepare ./data/mydata.csv

# 4. Generate training config
forge plan "Make a coding assistant" --data ./data/processed_train.jsonl

# 5. Train via Docker
forge train

# 6. Test your model
forge inference
```

## 🐳 GPU Containers

| Container | GPU Support |
|-----------|-------------|
| `forge:blackwell` | RTX 5090/5080/5070 |
| `forge:ada` | RTX 4090/4080/4070 |
| `forge:ampere` | RTX 3090/3080/3070 |
| `forge:hopper` | H100/H200 |

GPU is **auto-detected** - just run `forge docker build` and it picks the right one.

## 📋 Commands

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

## 🔒 Privacy & Security

- **Data stays local** - Only metadata sent to Gemini
- **Security Sentinel** - Reviews generated scripts before execution
- **Sandboxed execution** - Scripts run in isolated venv
- **Secure credentials** - API keys stored in system keyring

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

## 🛠️ Technical Stack

- **Training**: Unsloth + TRL + Transformers
- **Quantization**: BitsAndBytes (4-bit/8-bit)
- **AI Backend**: Google Gemini
- **Containers**: Docker with NVIDIA GPU passthrough
- **CLI**: Typer + Rich

## 📄 License

MIT
