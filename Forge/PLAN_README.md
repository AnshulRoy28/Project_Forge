# 🔥 Forge - AI-Powered Local Fine-Tuning

<p align="center">
  <strong>Fine-tune language models on your local GPU using natural language.</strong>
  <br>
  <em>Powered by Google Gemini • GPU-Optimized Docker Containers • Self-Healing Training</em>
</p>

---

## 🎯 What is Forge?

Forge is a **local-first CLI tool** that lets you fine-tune Small Language Models (SLMs) without writing code. Just describe what you want in plain English—Gemini handles the rest.

```bash
forge plan "Create a customer support chatbot that's friendly and helpful"
forge train
```

That's it. No ML expertise required.

---

## ✨ Key Features

### 🧠 Gemini Integration

Forge uses **Google Gemini** as an intelligent co-pilot throughout the entire workflow:

| Stage | Gemini's Role |
|-------|---------------|
| **Data Analysis** | Analyzes your dataset, identifies issues, suggests improvements |
| **Preprocessing** | Generates custom Python scripts to clean and format your data |
| **Configuration** | Creates optimized training configs based on your goal + hardware |
| **Error Diagnosis** | Diagnoses training failures and suggests fixes |
| **Self-Healing** | Automatically repairs and retries failed operations |

**Example: Agentic Preprocessing**
```bash
forge prepare ./data/customer_support.csv
```

Gemini will:
1. Analyze your CSV structure
2. Generate a preprocessing script
3. Create an isolated sandbox
4. Execute safely with auto-retry
5. Output training-ready JSONL

### 🐳 GPU-Optimized Docker Containers

**The #1 pain point in ML: dependency hell.** Forge solves this with pre-built Docker containers for every major GPU architecture:

| Container | GPUs | CUDA | Status |
|-----------|------|------|--------|
| `forge:blackwell` | RTX 5090/5080/5070 | 12.8 nightly | ✅ Tested |
| `forge:ada` | RTX 4090/4080/4070 | 12.4 | ✅ Ready |
| `forge:ampere` | RTX 3090/3080/3070 | 12.1 | ✅ Ready |
| `forge:hopper` | H100/H200 | 12.4 | ✅ Ready |

**Zero GPU overhead** — NVIDIA Container Toolkit passes your GPU directly to the container. Native CUDA performance.

```bash
# Auto-detect GPU and run
forge docker build    # One-time build
forge docker run train  # Training with perfect deps
```

### 🔄 Self-Healing Training

Training crashes? Forge diagnoses and fixes:

```
Error detected → Gemini analyzes → Config adjusted → Retry
```

**Auto-handled issues:**
- Out of memory → Reduces batch size
- BFloat16 errors → Switches precision
- Missing dependencies → Installs them
- Gated models → Prompts for auth

### 🎯 Hardware-Aware Optimization

Forge detects your GPU architecture and configures training optimally:

```
🔍 Detecting GPU...
  GPU: NVIDIA GeForce RTX 5080
  VRAM: 16.0 GB
  Compute: 12.0 (Blackwell)
  BF16: ✓ Supported
  FP8: ✓ Supported

🎯 Recommended Settings:
  Precision: BF16 (native to Blackwell)
  Optimizer: adamw_8bit (saves 2GB VRAM)
  Batch Size: 8
```

---

## 🚀 Quick Demo

```bash
# 1. Clone and enter
git clone https://github.com/yourrepo/forge.git && cd forge

# 2. Set your Gemini API key
set GEMINI_API_KEY=your_key_here   # Windows
export GEMINI_API_KEY=your_key_here  # Linux/Mac

# 3. Build your GPU-specific container
forge docker build

# 4. Analyze and preprocess your data
forge docker run study ./data/my_data.csv
forge docker run prepare ./data/my_data.csv

# 5. Generate training config with natural language
forge docker run plan "Make a helpful coding assistant"

# 6. Train!
forge docker run train

# 7. Test your model
forge docker run test
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User CLI                              │
│   forge plan | forge prepare | forge train | forge test      │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                     Gemini Brain 🧠                          │
│  • Data analysis    • Script generation                      │
│  • Config planning  • Error diagnosis                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│              GPU-Optimized Docker Containers 🐳              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Blackwell │ │   Ada    │ │  Ampere  │ │  Hopper  │       │
│  │RTX 50xx  │ │RTX 40xx  │ │RTX 30xx  │ │  H100    │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Training Engine ⚡                        │
│  • Unsloth (2x speed, 70% less VRAM)                        │
│  • BF16/TF32 precision                                       │
│  • Gradient checkpointing                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Commands Overview

| Command | Description |
|---------|-------------|
| `forge init` | Setup environment, configure API key |
| `forge study <data>` | Analyze dataset with Gemini |
| `forge prepare <data>` | Auto-preprocess (agentic) |
| `forge plan "<goal>"` | Generate training config |
| `forge train` | Start self-healing training |
| `forge test` | Interactive model testing |
| `forge docker detect` | Show GPU architecture |
| `forge docker build` | Build container for your GPU |
| `forge docker run <cmd>` | Run command in container |

---

## 🔒 Privacy & Security

- ✅ **Data stays local** — Only metadata sent to Gemini
- ✅ **Sandboxed execution** — Scripts run in isolated venvs
- ✅ **Security Sentinel** — Reviews generated code before execution
- ✅ **Secure credentials** — API keys in system keyring

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| AI Brain | Google Gemini |
| Training | Unsloth, TRL, Transformers |
| Quantization | BitsAndBytes |
| Containers | Docker + NVIDIA Container Toolkit |
| CLI | Typer + Rich |

---

## 📈 Roadmap

- [x] Core CLI commands
- [x] Gemini integration for all stages
- [x] Docker containers for all GPU architectures
- [x] Self-healing training
- [x] Hardware-aware config generation
- [ ] Web UI for monitoring
- [ ] Multi-GPU training
- [ ] Cloud deployment support
- [ ] Model marketplace integration

---

## 🏆 Built for Google Gemini Hackathon

Forge demonstrates how **Gemini can orchestrate complex ML workflows** — turning natural language into trained models with minimal friction.

---

<p align="center">
  <strong>🔥 Forge — Because fine-tuning should be as easy as asking.</strong>
</p>
