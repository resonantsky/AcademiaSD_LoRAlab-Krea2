#!/usr/bin/env bash

# Triton and SDNQ Installer for Linux

# This script lives in scripts/shell/, so the root directory is two levels up.
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$BASE_DIR/../.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[ERROR] Virtual environment not found at $VENV_PYTHON"
    echo "Run ./install_LoRAlab-Krea2.sh first."
    exit 1
fi

echo "======================================================="
echo "   Triton & SDNQ Installation (Linux)"
echo "======================================================="

# Redirect Triton and PyTorch Inductor cache directories to .cache inside the project root
export TRITON_CACHE_DIR="$PROJECT_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/.cache/torchinductor"

# Ensure cache directories exist
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Install pinned Triton version on Linux
echo "[INFO] Installing native Linux Triton 3.7.1..."
"$VENV_PYTHON" -m pip install -U triton==3.7.1

# Install SDNQ from PyPI
echo "[INFO] Installing SDNQ..."
"$VENV_PYTHON" -m pip install -U sdnq

echo "======================================================="
echo "   Process completed"
echo "======================================================="