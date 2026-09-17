#!/usr/bin/env bash

# Krea-2 Trainer Venv Installer for Linux (AMD GPU)

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$BASE_DIR/venv"

echo "========================================================"
echo "   KREA-2 LORA TRAINER INSTALLER (LINUX)"
echo "   Python 3.12 + PyTorch ROCm Environment"
echo "========================================================"
echo ""

# 1. Bootstrap 'uv' for standalone Python runtime provisioning
echo "[INFO] Ensuring 'uv' is available..."
if command -v uv &>/dev/null; then
    UV_BIN="uv"
else
    UV_DIR="$BASE_DIR/.uv_bin"
    mkdir -p "$UV_DIR"
    if [ ! -f "$UV_DIR/uv" ]; then
        echo "[INFO] Downloading standalone 'uv' binary..."
        curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_DIR" sh &>/dev/null
    fi
    UV_BIN="$UV_DIR/uv"
fi

# 2. Create virtual environment with auto-fetched standalone Python 3.12 & pip seed
if [ -d "$VENV_DIR" ]; then
    echo "[INFO] Removing previous virtual environment..."
    rm -rf "$VENV_DIR"
fi

echo "[INFO] Fetching standalone Python 3.12 runtime and building venv..."
"$UV_BIN" venv "$VENV_DIR" --python 3.12 --seed

VENV_PYTHON="$VENV_DIR/bin/python"
echo "[OK] Environment created with: $("$VENV_PYTHON" --version)"

# 3. Check AMD GPU
echo "[INFO] Checking AMD ROCm drivers..."
if command -v rocm-smi &>/dev/null; then
    echo "[OK] rocm-smi detected:"
    rocm-smi --showdriverversion --showproductname
else
    echo "[WARNING] rocm-smi not found. Make sure you have AMD ROCm drivers installed."
fi

# 4. Upgrade core build tools using uv
echo "[INFO] Upgrading pip, setuptools, and wheel..."
"$UV_BIN" pip install --python "$VENV_PYTHON" --upgrade pip setuptools wheel

# 5. Detect architecture and install PyTorch with ROCm support
echo "[INFO] Detecting AMD GPU architecture..."

GFX_ARCH=""
if command -v rocminfo &>/dev/null; then
    GFX_ARCH=$(rocminfo | grep -E "Name:\s+gfx" | head -n 1 | awk '{print $2}')
elif command -v rocm-smi &>/dev/null; then
    GFX_ARCH=$(rocm-smi --showid | grep -oE "gfx[0-9a-z]+" | head -n 1)
fi

if [ -z "$GFX_ARCH" ]; then
    echo "[WARNING] Could not auto-detect GFX architecture."
    echo "Defaulting to gfx1030 (RDNA2 / RX 6800)."
    GFX_ARCH="gfx1030"
fi

echo "[INFO] Using architecture target: $GFX_ARCH"
echo "[INFO] Installing PyTorch with ROCm support..."

INDEX_URL="https://rocm.nightlies.amd.com/whl-multi-arch/"

"$UV_BIN" pip install --python "$VENV_PYTHON" \
    rocm \
    rocm-sdk-core \
    rocm-sdk-devel \
    rocm-sdk-libraries \
    "rocm-sdk-device-$GFX_ARCH" \
    "amd-torch-device-$GFX_ARCH" \
    "amd-torchvision-device-$GFX_ARCH" \
    torch torchvision torchaudio \
    --index-url "$INDEX_URL"

if [ "$GFX_ARCH" = "gfx1030" ]; then
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    export PYTORCH_ROCM_ARCH=gfx1030
fi

# 6. Install Krea-2 dependencies, Triton 3.7.1, and SDNQ
echo "[INFO] Installing Diffusers, Transformers, PEFT, Accelerate, Safetensors, and Hugging Face Hub..."
"$UV_BIN" pip install --python "$VENV_PYTHON" diffusers transformers peft accelerate safetensors huggingface_hub

echo "[INFO] Installing workspace utilities..."
"$UV_BIN" pip install --python "$VENV_PYTHON" sentencepiece protobuf psutil

echo "[INFO] Pinning Triton to 3.7.1 and installing SDNQ..."
"$UV_BIN" pip install --python "$VENV_PYTHON" triton==3.7.1 sdnq

# Dataset curation dependencies
echo "[INFO] Installing InsightFace for dataset curation..."
"$UV_BIN" pip install --python "$VENV_PYTHON" insightface onnxruntime opencv-python || \
    echo "[WARNING] InsightFace could not be installed; curation will not be available."

# 7. Final verification
echo ""
echo "========================================================"
echo "   FINAL INSTALLATION VERIFICATION"
echo "========================================================"
"$VENV_PYTHON" -c "
import sys, torch, triton
print('Python Executable:', sys.executable)
print('Python Version:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('Triton:', triton.__version__)
print('ROCM available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

echo ""
echo "[OK] Installation complete. Run ./run_LoRAlab-Krea2.sh to launch the trainer."