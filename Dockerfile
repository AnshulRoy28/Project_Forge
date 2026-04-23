# ===========================================================================
# Project Forge — Full CLI containerized for demo/reproducibility
# ===========================================================================
# 
# Runs the entire `nnb` CLI inside a container.
# Uses Docker socket mounting to create sibling training containers on the host.
#
# BUILD:   docker build -t project-forge .
# RUN:     docker run -it --rm -v /var/run/docker.sock:/var/run/docker.sock -e GEMINI_API_KEY=<key> project-forge
#
# On Windows (Docker Desktop):
#   docker run -it --rm -v //var/run/docker.sock:/var/run/docker.sock -e GEMINI_API_KEY=<key> project-forge
# ===========================================================================

FROM python:3.11-slim

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Install Python deps first (cache layer) ---
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy source ---
COPY nnb/ nnb/
COPY tests/ tests/
COPY .kiro/ .kiro/
COPY README.md QUICKSTART.md ./

# --- Install nnb CLI ---
RUN pip install --no-cache-dir -e .

# --- Create project directory ---
RUN mkdir -p /app/.nnb

# --- Default env ---
ENV PYTHONUNBUFFERED=1
ENV NNB_HOME=/app

# --- Entrypoint ---
# Drop into bash with nnb available on PATH
CMD ["/bin/bash", "-c", "echo '🔥 Project Forge — nnb CLI ready' && echo '   Run: nnb start' && echo '' && /bin/bash"]
