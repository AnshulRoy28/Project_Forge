# 🔥 Forge

> **Docker-based CLI for fine-tuning Small Language Models with Gemini as your AI co-pilot**

Forge uses **Gemini** as an intelligent agent to handle data preprocessing, hyperparameter optimization, and training—all via **Docker containers** with GPU passthrough. Optimized for **RTX 50-series (Blackwell)**, 40-series, 30-series, and datacenter GPUs.

## ✨ Key Features

- 🧠 **AI-Powered** - Describe your goal in natural language, Gemini handles the rest
- ⚡ **Agentic Preprocessing** - Auto-generates and runs data preparation scripts
- 🔄 **Self-Healing Training** - Automatically fixes errors and retries
- 🐳 **Docker-First** - Training runs in GPU-optimized containers
- 🔄 **Container Persistence** - No model redownloading between training runs
- 🖥️ **Local Data** - Your data never leaves your machine
- 🎯 **Hardware-Aware** - Auto-detects GPU and selects optimal container

## 🚀 Quick Start

```bash
# 1. Install Forge CLI
git clone https://github.com/AnshulRoy28/Project_Forge.git
cd Project_Forge
pip install -e .

# 2. Initialize and build container
forge init
forge docker build

# 3. Train your model
forge plan "Make a coding assistant"
forge prepare ./data/mydata.csv  # if you have data
forge train

# 4. Test your model
forge inference
```

## 📋 Commands

| Command | Description |
|---------|-------------|
| `forge init` | Setup environment and configure API key |
| `forge plan "<goal>"` | Generate hardware-optimized training config |
| `forge prepare <path>` | **Agentic preprocessing** - auto-generate & run scripts |
| `forge train` | Execute training via Docker (self-healing) |
| `forge inference` | Interactive chat with trained model |
| `forge docker build` | Build container for your GPU |

## 🐳 GPU Support

| Container | GPU Support |
|-----------|-------------|
| `forge:blackwell` | RTX 5090/5080/5070 |
| `forge:ada` | RTX 4090/4080/4070 |
| `forge:ampere` | RTX 3090/3080/3070 |
| `forge:hopper` | H100/H200 |

GPU is **auto-detected** - just run `forge docker build` and it picks the right one.

## 📖 Documentation

For detailed setup instructions, troubleshooting, and advanced usage, see **[SETUP.md](SETUP.md)**.

## 📄 License

MIT
