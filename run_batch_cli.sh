#!/usr/bin/env bash

# AcademiaSD - Krea-2 LoRA Trainer CLI / Batch Runner for Linux
# Runs pre-cache and/or training without needing the Flask web interface.

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON=""

# Auto-wrap in tmux: the batch survives SSH session drops; running
# the script again reattaches (-A). Can be disabled with NO_TMUX=1.
if [ -z "$TMUX" ] && [ -z "$NO_TMUX" ] && command -v tmux &>/dev/null; then
    echo ""
    echo "[INFO] Batch in tmux session 'loralab-batch'."
    echo "       Detach without stopping: Ctrl+B then D"
    echo "       Reattach to session:    tmux attach -t loralab-batch"
    echo ""
    exec tmux new-session -A -s loralab-batch "NO_TMUX=1 '$BASE_DIR/run_batch_cli.sh' $*"
fi

# Find virtual environment
if [ -f "$BASE_DIR/.venv/bin/python" ]; then
    VENV_PYTHON="$BASE_DIR/.venv/bin/python"
elif [ -f "$BASE_DIR/venv/bin/python" ]; then
    VENV_PYTHON="$BASE_DIR/venv/bin/python"
elif [ -f "$BASE_DIR/env/bin/python" ]; then
    VENV_PYTHON="$BASE_DIR/env/bin/python"
fi

if [ -z "$VENV_PYTHON" ]; then
    echo "[ERROR] Virtual environment not found."
    echo "Run ./install_LoRAlab-Krea2.sh first."
    exit 1
fi

MODE="${1:-all}"  # Options: precache, train, all

# Choose trainer based on 'progressive' in train_settings.json, just as the
# UI's Train button does. Without this, a progressive project launched the direct
# trainer against the cache parent directory (which only contains 512/, 768/,
# 1024/ subdirs), which printed "empty cache" and exited with code 0: a silent
# no-op that the script reported as success.
TRAIN_SCRIPT="$BASE_DIR/scripts/python/2_train_lora_krea2.py"
PROGRESSIVE="$("$VENV_PYTHON" -c "
import json, sys
try:
    with open('$BASE_DIR/train_settings.json', encoding='utf-8') as f:
        print(str(json.load(f).get('progressive', 'off')).strip().lower())
except Exception:
    print('off')
" 2>/dev/null)"

if [ -n "$PROGRESSIVE" ] && [ "$PROGRESSIVE" != "off" ] && [ "$PROGRESSIVE" != "none" ]; then
    TRAIN_SCRIPT="$BASE_DIR/scripts/python/run_progressive.py"
fi

echo "================================================================"
echo "   KREA-2 LORA TRAINER - ALTERNATE BATCH PROCESS (CLI LINUX)"
echo "================================================================"
echo "Selected mode: $MODE"
echo "Python: $VENV_PYTHON"
echo "Trainer: $(basename "$TRAIN_SCRIPT") (progressive=$PROGRESSIVE)"
echo "================================================================"
echo ""

STATUS=0

case "$MODE" in
    precache)
        echo "[1/1] Running Pre-Cache..."
        "$VENV_PYTHON" "$BASE_DIR/scripts/python/1_pre_cache_krea2.py"
        STATUS=$?
        ;;
    train)
        echo "[1/1] Running LoRA Training..."
        "$VENV_PYTHON" "$TRAIN_SCRIPT"
        STATUS=$?
        ;;
    all)
        echo "[1/2] Step 1: Running Pre-Cache..."
        "$VENV_PYTHON" "$BASE_DIR/scripts/python/1_pre_cache_krea2.py"

        if [ $? -eq 0 ]; then
            echo ""
            echo "[2/2] Step 2: Running LoRA Training..."
            "$VENV_PYTHON" "$TRAIN_SCRIPT"
            STATUS=$?
        else
            echo "[ERROR] Pre-Cache failed. Aborting training."
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 [precache|train|all]"
        echo "  precache : Runs only 1_pre_cache_krea2.py"
        echo "  train    : Runs only 2_train_lora_krea2.py (or run_progressive.py)"
        echo "  all      : Runs both sequentially (default)"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
if [ $STATUS -eq 0 ]; then
    echo "Batch process completed."
else
    echo "[ERROR] Batch process exited with code $STATUS."
fi
echo "================================================================"
exit $STATUS