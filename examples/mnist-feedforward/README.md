# MNIST Feedforward Neural Network — Example Project

This is a pre-generated example project for the Neural Network Builder (`nnb`).
It contains a complete, working MNIST digit classification pipeline using a simple
feedforward neural network in PyTorch.

## What's Inside

```
examples/mnist-feedforward/
├── state.json              # Project state (TRAINING_COMPLETE)
├── steering-doc.yaml       # Project specification
├── imp_points.md           # Key architectural decisions
├── data-requirements.md    # Data requirements document
├── conversation.txt        # Original user description
├── Dockerfile              # Docker environment for this project
└── workspace/
    ├── model.py            # Feedforward NN (784→48→10)
    ├── dataset.py          # MNIST data loading & preprocessing
    ├── train.py            # Complete training loop
    ├── config.yaml         # All hyperparameters
    ├── requirements.txt    # Python dependencies
    └── mock_runner.py      # Mock validation script
```

## Using This Example

You can use this example to skip the AI generation stages and jump straight to
building the Docker environment, mock-running, and training.

### Option 1: Copy into a new project

```bash
# Start a new project and go through stages 1-5 (conversation → env build)
nnb start
# ... answer questions, validate data, build environment ...

# Then instead of generating code with Gemini, copy from this example:
nnb generate --from examples/mnist-feedforward
```

### Option 2: Quick test

```bash
# From the project root, start a project and skip to code generation:
nnb start
# Go through the pipeline, then at the generate step:
nnb generate --from examples/mnist-feedforward

# Validate the code
nnb mock-run

# Train!
nnb train
```

## Model Details

- **Architecture**: Single-layer feedforward network (784 → 48 → 10)
- **Dataset**: MNIST (auto-downloaded via torchvision)
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 10
- **Batch Size**: 64
- **Loss**: CrossEntropyLoss
