# Forge Docker

Pre-configured Docker containers for each GPU architecture.

## Quick Start

### Linux/Mac
```bash
# Auto-detect GPU and run
./docker/forge-docker.sh plan "Make a coding assistant" --data /data/train.jsonl
./docker/forge-docker.sh train
```

### Windows
```powershell
# Auto-detect GPU and run
.\docker\forge-docker.bat plan "Make a coding assistant" --data /data/train.jsonl
.\docker\forge-docker.bat train
```

## Available Images

| Image | GPU | CUDA | PyTorch |
|-------|-----|------|---------|
| `forge:blackwell` | RTX 5090/5080/5070 | 12.8 | nightly |
| `forge:ada` | RTX 4090/4080/4070 | 12.4 | 2.6 |
| `forge:ampere` | RTX 3090/3080/3070 | 12.1 | 2.5 |
| `forge:hopper` | H100/H200 | 12.4 | 2.6 |

## Manual Build

```bash
# Build specific architecture
docker compose -f docker/docker-compose.yml --profile blackwell build
docker compose -f docker/docker-compose.yml --profile ada build
docker compose -f docker/docker-compose.yml --profile ampere build
docker compose -f docker/docker-compose.yml --profile hopper build
```

## Manual Run

```bash
# Run with specific profile
docker compose -f docker/docker-compose.yml --profile ada run forge-ada train

# Or directly
docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/output:/output \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  forge:ada train
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Your Gemini API key |
| `NVIDIA_VISIBLE_DEVICES` | GPU selection (default: all) |

## Data Volumes

| Container Path | Description |
|---------------|-------------|
| `/data` | Training data (input) |
| `/output` | Trained models (output) |
| `/checkpoints` | Training checkpoints |
| `/app/forge.yaml` | Configuration file |

## Performance

Docker with NVIDIA Container Toolkit has **zero GPU overhead**:
- GPU is passed directly to container
- CUDA runs at native speed
- No virtualization penalty

The only overhead is the container startup (~1-2 seconds).
