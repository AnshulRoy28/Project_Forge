# Auto-generated Dockerfile for nnb-20260423-091628-af157ba1
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch>=2.1.0 torchvision>=0.16.0 torchaudio>=2.1.0 numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0 matplotlib>=3.7.0 tqdm>=4.65.0 tensorboard>=2.14.0

# Create data directory
RUN mkdir -p /data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/data/.torch

# Default command
CMD ["/bin/bash"]
