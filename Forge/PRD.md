# Forge PRD
## AI-Powered LLM Fine-Tuning Tool

---

## Executive Summary

**Forge** is a CLI tool that enables developers to fine-tune language models using natural language commands. Powered by Gemini, it automates the complex process of configuring, training, and deploying custom LLMs with zero ML expertise required. All training runs inside Docker containers with GPU passthrough.

---

## Problem Statement

Fine-tuning LLMs today requires:
- Deep ML knowledge (LoRA, quantization, optimizers)
- GPU configuration expertise (CUDA, drivers, memory management)
- Hours of trial-and-error for hyperparameter tuning
- Complex environment setup (dependencies, Docker, NVIDIA toolkit)

**Result**: Only ML engineers can fine-tune models, leaving 90% of developers unable to customize LLMs for their use cases.

---

## Solution: The Agentic Pipeline

Forge is an **AI-powered agent** that handles the entire fine-tuning workflow autonomously:

### Step 1: Initialize
```bash
forge init
```
- Sets up project directory
- Configures Gemini API connection
- Gemini becomes the "brain" for all subsequent decisions

### Step 2: Data Understanding (Agentic)
```bash
forge prepare data.csv "Make a customer support bot"
```
- Gemini **samples a few rows** from your data to understand structure
- Analyzes columns, formats, text patterns
- Identifies what preprocessing is needed based on your intent

### Step 3: Automated Data Processing
- Gemini **generates a data processing pipeline** (Python script)
- Runs pipeline in **isolated sandbox environment**
- Handles: text cleaning, format conversion, train/val splits
- Saves processed data to `/data/processed_train.jsonl`

### Step 4: Model Selection
- Gemini **chooses the optimal model** for your use case:
  - Customer support → Qwen (multilingual, conversational)
  - Code assistant → Llama or Phi (reasoning)
  - Small deployment → SmolLM or TinyLlama
- Considers VRAM constraints from hardware probe

### Step 5: Config Generation
```bash
forge plan "Make a customer support bot"
```
- Gemini creates `forge.yaml` with optimal settings:
  - LoRA rank, learning rate, batch size
  - Based on model, data size, and hardware

### Step 6: Docker Training
```bash
forge train
```
- Training automatically runs in **architecture-specific container**
- Auto-detects GPU and selects optimal Docker image
- PyTorch optimized for your GPU (Blackwell, Ada, Ampere)
- Self-healing: auto-retries on OOM, adjusts batch size
- Real-time output streaming

### Step 7: Export to Ollama
- Model automatically exported to GGUF
- Registered with local Ollama
- Ready to use: `ollama run my-model`

---

## CLI Commands

### Core Commands

```bash
# 1. Setup (one-time)
forge init
forge docker build

# 2. Train (does everything automatically via Docker)
forge prepare data.csv
forge plan "Make a customer support bot"
forge train

# 3. Test your model
forge inference
```

### Optional Flags

Customize any step:

```bash
# Force a specific model
forge ignite data.csv "goal" --model qwen

# Adjust training
forge ignite data.csv "goal" --epochs 5 --batch-size 16

# Skip export (training only)
forge ignite data.csv "goal" --no-export

# Dry run (show plan without executing)
forge ignite data.csv "goal" --dry-run
```

### Available Models

| Flag | Model | Best For |
|------|-------|----------|
| `--model smollm` | SmolLM 1.7B | Testing, edge deployment |
| `--model tinyllama` | TinyLlama 1.1B | Lightweight apps |
| `--model llama` | Llama 3.2 3B | General purpose |
| `--model qwen` | Qwen 2.5 3B | Multilingual, chat |
| `--model phi` | Phi-3.8B | Compact reasoning |
| `--model gemma` | Gemma 2B/9B | Quality + efficiency |
| `--model mistral` | Mistral 7B | Strong general |

---

### Individual Commands (Power Users)

| Command | Description |
|---------|-------------|
| `forge study <file>` | Analyze data quality |
| `forge prepare <file>` | Preprocess data |
| `forge plan "<goal>"` | Generate config only |
| `forge train` | Run training only |
| `forge export` | Export model only |
| `forge status` | Show project health |

---

## Supported Models

| Family | Models | Use Case |
|--------|--------|----------|
| SmolLM | 135M, 360M, 1.7B | Pipeline testing, edge devices |
| TinyLlama | 1.1B | Lightweight deployment |
| Llama | 1B, 3B, 8B | General purpose |
| Qwen | 1.5B, 3B, 7B | Multilingual, reasoning |
| Phi | 3.8B | Compact powerhouse |
| Gemma | 2B, 9B | Quality + efficiency |
| Mistral | 7B | Strong general model |

---

## GPU Architectures

| Architecture | GPUs | Docker Image |
|--------------|------|--------------|
| Blackwell | RTX 50xx | `forge:blackwell` |
| Ada Lovelace | RTX 40xx | `forge:ada` |
| Ampere | RTX 30xx | `forge:ampere` |
| Hopper | H100/H200 | `forge:hopper` |

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LOCAL MACHINE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  Forge CLI (Python)                     │ │
│  │  init · login · status · study · prepare · plan         │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 Gemini API (Brain)                      │ │
│  │  Dataset analysis · Config generation · Error diagnosis │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
├────────────────────────────┼────────────────────────────────┤
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Docker Container (GPU)                     │ │
│  │  train · test · export                                  │ │
│  │  PyTorch + CUDA + Unsloth + Transformers               │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                      Ollama                             │ │
│  │               Local model serving                       │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Known Issues & Fixes

| Issue | Cause | Solution |
|-------|-------|----------|
| Hardware detection fails | Runs locally without PyTorch | Use `forge docker probe` to detect inside container |
| SmolLM produces NaN | Gradient overflow in tiny models | Use LR=1e-5, BF16, rank=4 for <500M models |
| Local test fails | PyTorch doesn't support sm_120 | Use Docker or Ollama for testing |
| Batch size too small | CPU fallback during planning | Probe container returns accurate VRAM |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Time to first training | < 10 minutes |
| Commands to train | ≤ 5 (or 1 with `forge ignite`) |
| Training success rate | > 95% (with self-healing) |
| Model export success | > 99% |

---

## Roadmap

| Phase | Feature | Priority |
|-------|---------|----------|
| v0.9 | Probe container | Critical |
| v0.9 | SmolLM stability fix | Critical |
| v1.0 | Auto-eval command | High |
| v1.0 | Synthetic expansion | High |
| v1.0 | Ignite one-click | High |
| v1.1 | Web UI dashboard | Medium |
| v1.1 | Cloud GPU (RunPod) | Medium |

---

## Competitive Advantage

| vs. | Forge Advantage |
|-----|-----------------|
| Raw HuggingFace | 10x simpler, no ML knowledge needed |
| Axolotl | Pre-configured, Gemini-guided |
| OpenAI Fine-tuning | Local, private, no data upload |
| LlamaFactory | Docker-based, zero environment setup |
