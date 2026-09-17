#!/usr/bin/env bash

# Krea-2 Trainer Venv Installer for Linux (AMD GPU)

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE=""

echo "========================================================"
echo "   KREA-2 LORA TRAINER INSTALLER (LINUX)"
echo "   Python + PyTorch ROCm Environment"
echo "========================================================"
echo ""

# 1. Check Python 3 on the system
if command -v python3.13 &>/dev/null; then
    PYTHON_EXE="python3.13"
elif command -v python3.12 &>/dev/null; then
    PYTHON_EXE="python3.12"
elif [ -x "/usr/bin/python3" ]; then
    PYTHON_EXE="/usr/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON_EXE="python3"
fi

if [ -n "$PYTHON_EXE" ]; then
    echo "[OK] Python detected: $($PYTHON_EXE --version)"
else
    echo "[ERROR] Python 3 is not installed on the system."
    echo "Please install it using your package manager (e.g., sudo apt install python3 python3-venv python3-pip)"
    exit 1
fi

# 2. Check venv module
$PYTHON_EXE -m venv --help &>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] Python venv module is not available."
    echo "Install Python first!"
    exit 1
fi

# 3. Check AMD GPU
echo "[INFO] Checking AMD ROCm drivers..."
if command -v rocm-smi &>/dev/null; then
    echo "[OK] rocm-smi detected:"
    rocm-smi --showdriverversion --showproductname
else
    echo "[WARNING] rocm-smi not found. Make sure you have AMD ROCm drivers installed."
fi

# 4. Create clean venv
VENV_DIR="$BASE_DIR/venv"
if [ -d "$VENV_DIR" ]; then
    echo "[INFO] Removing previous virtual environment..."
    rm -rf "$VENV_DIR"
fi

echo "[INFO] Creating new virtual environment..."
$PYTHON_EXE -m venv "$VENV_DIR"
if [ $? -ne 0 ]; then
    echo "[ERROR] Could not create the virtual environment."
    exit 1
fi

VENV_PYTHON="$VENV_DIR/bin/python"

# 5. Upgrade pip tools
echo "[INFO] Upgrading pip, setuptools, and wheel..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

# 6. Install PyTorch with ROCm support

echo "[INFO] Detecting AMD GPU architecture..."

# Auto-detect GFX architecture ID (e.g., gfx1100, gfx1030)
GFX_ARCH=""
if command -v rocminfo &>/dev/null; then
    GFX_ARCH=$(rocminfo | grep -E "Name:\s+gfx" | head -n 1 | awk '{print $2}')
elif command -v rocm-smi &>/dev/null; then
    GFX_ARCH=$(rocm-smi --showid | grep -oE "gfx[0-9a-z]+" | head -n 1)
fi

# Fallback or manual entry if auto-detection fails
if [ -z "$GFX_ARCH" ]; then
    echo "[WARNING] Could not auto-detect GFX architecture."
    echo "Defaulting to gfx1030 (RDNA2 / RX 6000 series). Adjust GFX_ARCH if needed."
    GFX_ARCH="gfx1030"
fi

echo "[INFO] Using architecture target: $GFX_ARCH"
echo "[INFO] Installing PyTorch with ROCm support..."

INDEX_URL="https://rocm.nightlies.amd.com/whl-multi-arch/"

"$VENV_PYTHON" -m pip install \
    "amd-torch-device-$GFX_ARCH" \
    "amd-torchvision-device-$GFX_ARCH" \
    torch torchvision torchaudio \
    --index-url "$INDEX_URL"

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030

# 7. Install Krea-2 dependencies
echo "[INFO] Installing Diffusers, Transformers, PEFT, Accelerate, Safetensors, and Hugging Face Hub..."
"$VENV_PYTHON" -m pip install diffusers transformers peft accelerate safetensors huggingface_hub

echo "[INFO] Installing BitsAndBytes and utilities..."
"$VENV_PYTHON" -m pip install bitsandbytes sentencepiece protobuf psutil

# Dataset curation (0_curate_dataset.py): facial identity scoring
# with ArcFace. CPU-only; the model (~300 MB) is downloaded on first use.
echo "[INFO] Installing InsightFace for dataset curation..."
"$VENV_PYTHON" -m pip install insightface onnxruntime opencv-python || \
    echo "[WARNING] InsightFace could not be installed; curation will not be available. The rest of the trainer works regardless."

# 8. Final verification
echo ""
echo "========================================================"
echo "   FINAL INSTALLATION VERIFICATION"
echo "========================================================"
"$VENV_PYTHON" -c "import torch; print('PyTorch:', torch.__version__); print('ROCM available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo ""
echo "[OK] Installation complete. Run ./run_LoRAlab-Krea2.sh to launch the trainer."