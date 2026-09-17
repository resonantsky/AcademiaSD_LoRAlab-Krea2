#!/usr/bin/env bash

# AcademiaSD - Krea-2 LoRA Trainer runner script for Linux

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE=""

# Auto-wrap in tmux: the server survives SSH session drops and running
# this script again reattaches to the existing session (-A) instead of duplicating.
# Can be disabled with NO_TMUX=1.
if [ -z "$TMUX" ] && [ -z "$NO_TMUX" ] && command -v tmux &>/dev/null; then
    echo ""
    echo "[INFO] Server running in tmux session 'loralab'."
    echo "       Detach without stopping: Ctrl+B then D"
    echo "       Reattach to session:    ./run_LoRAlab-Krea2.sh  (or: tmux attach -t loralab)"
    echo "       Run without tmux:       NO_TMUX=1 ./run_LoRAlab-Krea2.sh"
    echo ""
    exec tmux new-session -A -s loralab "NO_TMUX=1 '$BASE_DIR/run_LoRAlab-Krea2.sh'"
fi

echo ""
echo "================================================================"
echo "       ACADEMIASD - KREA-2 LORA TRAINER (Linux)"
echo "================================================================"
echo ""
echo "Trainer directory:"
echo "$BASE_DIR"
echo ""

# Find existing virtual environment
if [ -f "$BASE_DIR/.venv/bin/python" ]; then
    PYTHON_EXE="$BASE_DIR/.venv/bin/python"
elif [ -f "$BASE_DIR/venv/bin/python" ]; then
    PYTHON_EXE="$BASE_DIR/venv/bin/python"
elif [ -f "$BASE_DIR/env/bin/python" ]; then
    PYTHON_EXE="$BASE_DIR/env/bin/python"
elif [ -f "$BASE_DIR/../venv/bin/python" ]; then
    PYTHON_EXE="$BASE_DIR/../venv/bin/python"
fi

if [ -z "$PYTHON_EXE" ]; then
    echo "[ERROR] Virtual environment not found."
    echo "Run ./install_LoRAlab-Krea2.sh first to create it."
    exit 1
fi

echo "Python environment found:"
echo "$PYTHON_EXE"
echo ""

if [ ! -f "$BASE_DIR/scripts/python/server.py" ]; then
    echo "[ERROR] scripts/python/server.py does not exist"
    exit 1
fi

if [ ! -f "$BASE_DIR/web/trainer_ui.html" ]; then
    echo "[ERROR] web/trainer_ui.html does not exist"
    exit 1
fi

echo "Checking Python..."
"$PYTHON_EXE" --version || { echo "[ERROR] Cannot execute Python."; exit 1; }

echo ""
echo "================================================================"
echo "Starting web server..."
echo "================================================================"
echo ""
echo "Open in browser:"
echo ""
echo "    http://127.0.0.1:5000"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Reduces VRAM fragmentation, critical on 12 GB GPUs at 768x768 or higher.
# server.py launches scripts via subprocess.Popen without env=, so they inherit it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON_EXE" "$BASE_DIR/scripts/python/server.py"