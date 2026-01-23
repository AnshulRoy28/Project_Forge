#!/bin/bash
# =============================================================================
# Forge Docker Auto-Detect Script
# Detects GPU architecture and runs the appropriate container
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🔥 Forge Docker Runner${NC}"
echo ""

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi not found. Is NVIDIA driver installed?${NC}"
    exit 1
fi

# Get GPU compute capability
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

echo -e "  GPU: ${CYAN}${GPU_NAME}${NC}"
echo -e "  Compute: ${CYAN}${COMPUTE_CAP}${NC}"

# Determine architecture from compute capability
MAJOR=$(echo $COMPUTE_CAP | cut -d'.' -f1)
MINOR=$(echo $COMPUTE_CAP | cut -d'.' -f2)

if [ "$MAJOR" -ge 12 ]; then
    ARCH="blackwell"
    PROFILE="blackwell"
    echo -e "  Architecture: ${GREEN}Blackwell (RTX 50-series)${NC}"
elif [ "$MAJOR" -eq 9 ]; then
    ARCH="hopper"
    PROFILE="hopper"
    echo -e "  Architecture: ${GREEN}Hopper (H100/H200)${NC}"
elif [ "$MAJOR" -eq 8 ] && [ "$MINOR" -ge 9 ]; then
    ARCH="ada"
    PROFILE="ada"
    echo -e "  Architecture: ${GREEN}Ada Lovelace (RTX 40-series)${NC}"
elif [ "$MAJOR" -eq 8 ]; then
    ARCH="ampere"
    PROFILE="ampere"
    echo -e "  Architecture: ${GREEN}Ampere (RTX 30-series)${NC}"
else
    ARCH="ampere"
    PROFILE="ampere"
    echo -e "  Architecture: ${YELLOW}Legacy (using Ampere image)${NC}"
fi

echo ""

# Check if image exists, build if not
IMAGE="forge:${ARCH}"
if ! docker image inspect $IMAGE &> /dev/null; then
    echo -e "${YELLOW}Building ${IMAGE}...${NC}"
    docker compose -f docker/docker-compose.yml --profile $PROFILE build
fi

# Run the command
echo -e "${GREEN}Running Forge...${NC}"
echo ""

docker compose -f docker/docker-compose.yml --profile $PROFILE run --rm forge-${ARCH} "$@"
