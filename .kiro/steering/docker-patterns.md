---
inclusion: auto
description: Docker patterns and best practices for container management in the neural network builder.
---

# Docker Patterns

## Container Lifecycle

### One Container Per Project

```python
# WRONG: New container for every command
def run_command(cmd):
    docker.run(f"nnb-{project_id}", cmd)  # Creates new container each time

# CORRECT: Reuse project container
def run_command(cmd):
    container = docker.get_container(f"nnb-{project_id}")
    container.exec(cmd)
```

### Container States

| State | Description | Commands Available |
|-------|-------------|-------------------|
| Not Built | No container exists | `nnb env build` |
| Built | Container exists but stopped | `nnb env shell`, `nnb mock-run` |
| Running | Container is executing | `nnb attach`, `nnb env shell` |
| Stopped | Container stopped after execution | `nnb env start` |

## Volume Mount Patterns

### Read-Only Data Mount

```dockerfile
# Data should NEVER be modified by training
VOLUME /data:ro
```

```python
# Mount user's data directory as read-only
docker.run(
    volumes={
        str(data_path): {"bind": "/data", "mode": "ro"},
        str(workspace_path): {"bind": "/workspace", "mode": "rw"}
    }
)
```

### Read-Write Workspace Mount

```python
# All generated files appear on host immediately
workspace_path = project_dir / ".nnb" / "workspace"
workspace_path.mkdir(parents=True, exist_ok=True)

docker.run(
    volumes={
        str(workspace_path): {"bind": "/workspace", "mode": "rw"}
    }
)
```

## Dockerfile Generation

### Base Image Selection

```python
def select_base_image(framework, cuda_required):
    """Select appropriate base image based on project requirements."""
    if framework == "pytorch":
        if cuda_required:
            return "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
        else:
            return "pytorch/pytorch:2.1.0-runtime"
    elif framework == "tensorflow":
        if cuda_required:
            return "tensorflow/tensorflow:2.14.0-gpu"
        else:
            return "tensorflow/tensorflow:2.14.0"
    elif framework == "jax":
        if cuda_required:
            return "nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04"
        else:
            return "python:3.10-slim"
```

### Minimal Dependencies

```dockerfile
# WRONG: Install everything
RUN pip install torch torchvision torchaudio transformers datasets \
    pandas numpy scipy scikit-learn matplotlib seaborn plotly \
    jupyter notebook ipython

# CORRECT: Install only what's needed
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    pillow==10.1.0
```

### Pinned Versions

```dockerfile
# WRONG: Unpinned versions
RUN pip install torch torchvision

# CORRECT: Pinned versions for reproducibility
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0
```

## Health Checks

### Container Health Check

```python
def check_container_health(container):
    """Run comprehensive health check after container build."""
    checks = [
        ("Python version", "python --version"),
        ("GPU availability", "python -c 'import torch; print(torch.cuda.is_available())'"),
        ("CUDA version", "python -c 'import torch; print(torch.version.cuda)'"),
        ("Framework import", "python -c 'import torch; import torchvision'"),
        ("Data mount", "ls /data"),
        ("Workspace mount", "ls /workspace"),
    ]
    
    results = []
    for name, cmd in checks:
        try:
            output = container.exec_run(cmd)
            results.append((name, "✓", output.output.decode()))
        except Exception as e:
            results.append((name, "✗", str(e)))
    
    return results
```

## Detached Execution

### Training as Background Process

```python
def start_training(container, config):
    """Start training as detached process."""
    # Start training in background
    exec_id = container.exec_run(
        "python /workspace/train.py --config /workspace/config.yaml",
        detach=True,
        stream=True
    )
    
    # Save exec_id for later attachment
    state.save("training_exec_id", exec_id)
    
    return exec_id

def attach_to_training(container, exec_id):
    """Reattach to running training process."""
    # Stream logs from detached process
    for line in container.exec_run(exec_id, stream=True):
        print(line.decode(), end="")
```

## Resource Limits

### GPU Allocation

```python
def create_container_with_gpu(image, gpus="all"):
    """Create container with GPU access."""
    return docker.containers.run(
        image,
        device_requests=[
            docker.types.DeviceRequest(
                count=-1 if gpus == "all" else int(gpus),
                capabilities=[["gpu"]]
            )
        ],
        detach=True
    )
```

### Memory Limits

```python
def create_container_with_limits(image, memory_gb=8):
    """Create container with memory limits."""
    return docker.containers.run(
        image,
        mem_limit=f"{memory_gb}g",
        memswap_limit=f"{memory_gb}g",
        detach=True
    )
```

## Container Cleanup

### Automatic Cleanup

```python
def cleanup_container(project_id):
    """Clean up container and volumes."""
    container_name = f"nnb-{project_id}"
    
    try:
        container = docker.containers.get(container_name)
        container.stop(timeout=10)
        container.remove(v=True)  # Remove volumes
        logger.info(f"Cleaned up container {container_name}")
    except docker.errors.NotFound:
        logger.warning(f"Container {container_name} not found")
```

### Preserve Artifacts

```python
def cleanup_but_preserve_artifacts(project_id):
    """Clean up container but keep workspace artifacts."""
    # Workspace is on host via volume mount, so it persists
    # Only remove the container itself
    container_name = f"nnb-{project_id}"
    
    try:
        container = docker.containers.get(container_name)
        container.stop(timeout=10)
        container.remove(v=False)  # Don't remove volumes
        logger.info(f"Container removed, artifacts preserved in .nnb/workspace/")
    except docker.errors.NotFound:
        pass
```

## Error Handling

### Container Build Failures

```python
def build_container_with_retry(dockerfile_path, tag, max_retries=3):
    """Build container with retry logic."""
    for attempt in range(max_retries):
        try:
            image, logs = docker.images.build(
                path=str(dockerfile_path.parent),
                dockerfile=str(dockerfile_path.name),
                tag=tag,
                rm=True
            )
            
            # Log build output
            for line in logs:
                if "stream" in line:
                    logger.debug(line["stream"].strip())
            
            return image
        except docker.errors.BuildError as e:
            logger.error(f"Build attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### Container Execution Failures

```python
def exec_with_error_handling(container, cmd):
    """Execute command with comprehensive error handling."""
    try:
        result = container.exec_run(cmd)
        
        if result.exit_code != 0:
            error_msg = result.output.decode()
            logger.error(f"Command failed: {cmd}\n{error_msg}")
            raise ContainerExecutionError(error_msg)
        
        return result.output.decode()
    except docker.errors.APIError as e:
        logger.error(f"Docker API error: {e}")
        raise
```

## Best Practices

### DO:
- ✅ Use one container per project
- ✅ Mount data as read-only
- ✅ Pin all dependency versions
- ✅ Run health checks after build
- ✅ Use detached execution for long-running tasks
- ✅ Stream logs for user feedback
- ✅ Preserve artifacts on host via volume mounts

### DON'T:
- ❌ Create new containers for every command
- ❌ Allow training code to modify source data
- ❌ Use unpinned dependency versions
- ❌ Skip health checks
- ❌ Block CLI while training runs
- ❌ Store artifacts only in container
- ❌ Hardcode resource limits
