# -*- coding: utf-8 -*-
"""

2_train_lora_krea2.py — Entrenamiento LoRA para Krea 2 (RAW) con NF4 / LoRA Training for Krea 2

Reads configuration from train_settings.json if present.
"""
import os
import gc
import re
import csv
import math
import time
import random
import json
import shutil
import signal
import sys
import zlib
import collections
from collections import defaultdict

# Must be set before the CUDA allocator is initialized. Reduces VRAM
# fragmentation, which is the typical failure mode on 12 GB GPUs when
# training at 768x768 or higher.

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import bitsandbytes as bnb
from safetensors.torch import save_file, load
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors import safe_open
from bitsandbytes.functional import QuantState

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Project root (this script lives in scripts/python/). All paths are
# anchored here instead of the working directory, so it works when
# invoked from anywhere.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def from_root(path):
    """Resuelve una ruta relativa contra la raíz del proyecto (absolutas intactas)."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))


# Container folders: everything generated is grouped here instead of at the root.

CACHE_ROOT  = os.path.join(PROJECT_ROOT, "cached_data_local")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output_local")

# ── DEFAULTS ────────────────────────────────────────────────────────────────
# Everything added after the original block is additive and opt-in: with an
# empty sidecar and no preset, these values reproduce the historical behavior.

DEFAULTS = {
    "model_id": "Krea-2-NF4",
    "cache_dir": f"{CACHE_ROOT}/default",
    "output_dir": f"{OUTPUT_ROOT}/default",
    "total_steps": 1200,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 1e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 100,
    "lora_rank": 16,
    "lora_alpha": 32,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "save_every": 25,
    "seed": 42,
    "timestep_sampling": "krea2_shift",
    "preview_every": 0,
    "preview_steps": 28,
    "preview_cfg": 3.5,
    "preview_caption_mode": "first",
    "project_name": "",
    "trigger_word": "",
    "lora_target": "all",
    "compact_text": True,
    "init_lora_from": "",
    "gradient_checkpointing": True,

    # ── A1/A10: precision ────────────────────────────────────────────────────
    "lora_dtype": "bf16",              # "bf16" | "fp32" (fp32 = +235 MB VRAM)
    "high_precision_targets": False,   # generate noise and target in fp32

    # ── A2: dataset sampling ─────────────────────────────────────────────────
    "sampler": "legacy",               # "epoch" | "legacy"

    # ── A3/A8: guardrails ────────────────────────────────────────────────────
    "nan_guard": True,
    "nan_abort_after": 20,
    "max_loss": 0.0,                   # 0 = disabled
    "oom_guard": True,
    "oom_abort_after": 3,

    # ── A4: checkpoints ──────────────────────────────────────────────────────
    "resume_on_corrupt": "abort",      # "abort" | "restart"

    # ── A5/A7: schedule and optimizer ────────────────────────────────────────
    "warmup_units": "updates",         # "updates" | "micro_steps" | "ratio"
    "optimizer": "adamw8bit_paged",    # | "adamw8bit" | "adamw"
    "optimizer_eps": 1e-8,
    "optimizer_betas": [0.9, 0.999],

    # ── B1: timesteps ────────────────────────────────────────────────────────
    "timestep_weighting": "none",      # "none" | "bell" | "half_bell"
    "logit_normal_mu": 0.0,
    "logit_normal_sigma": 1.0,
    "sigma_min": 0.0,
    "sigma_max": 1.0,
    "content_or_style": "balanced",    # "balanced" | "content" | "style"
    "noise_offset": 0.0,               # see B3: discouraged in rectified flow

    # ── B2: EMA ──────────────────────────────────────────────────────────────
    "use_ema": False,
    "ema_decay": 0.99,
    "ema_device": "cpu",               # "cpu" | "cuda"

    # ── B4: LR scheduler ─────────────────────────────────────────────────────
    "lr_scheduler": "cosine",          # | "constant" | "linear" | "cosine_with_restarts" | "step"
    "lr_num_cycles": 3,
    "lr_step_gamma": 0.5,
    "lr_step_count": 4,

    # ── C1: validation ───────────────────────────────────────────────────────
    "val_split": 0.0,
    "val_cache_dir": "",
    "validate_every": 0,
    "validation_sigmas": [0.25, 0.5, 0.75, 1.0],
    "val_seed": 1234,

    # ── C2/C3/C4: observability ─────────────────────────────────────────────
    "loss_window": 100,
    "loss_display": "cumulative",      # "window" | "cumulative"
    "csv_log": True,
    "max_checkpoints_to_keep": 0,      # 0 = keep all
    "export_metadata": True,
    "export_alpha_tensors": False,

    # ── C5: previews ─────────────────────────────────────────────────────────
    "preview_source": "caption",       # "caption" | "prompts"
    "preview_walk_seed": False,

    # ── D3: caption dropout ──────────────────────────────────────────────────
    "caption_dropout_rate": 0.0,

    # ── Dataset curation ─────────────────────────────────────────────────────
    # 0_curate_dataset.py splits the dataset into two groups by facial identity;
    # here each group trains with its own weight. Without curation_report.json in the
    # dataset folder this is an exact no-op, so an uncurated run behaves
    # identically to how it did before the option existed.
    "dataset_path": "./dataset",
    "curation_weights": True,

    # ── Progressive / Multi-phase ─────────────────────────────────────────────
    "run_id": "",
    "phase_index": 0,
    "phase_count": 1,
    "phase_label": "",
    "global_step_offset": 0,
}

# Recommended value bundles. They change the defaults, never overwriting an
# explicit key from the user. They exist so that enabling a sensible set costs a
# single keystroke instead of fifteen, keeping everything else opt-in.

PRESETS = {
    "stable_v2": {
        "sampler": "epoch",
        "lora_dtype": "fp32",
        "optimizer_eps": 1e-6,
        "optimizer_betas": [0.9, 0.99],
        "high_precision_targets": True,
        "loss_display": "window",
        "max_checkpoints_to_keep": 5,
        "nan_guard": True,
        "oom_guard": True,
    },
}

# ── LOAD CONFIG ─────────────────────────────────────────────────────────────
# The progressive resolution orchestrator passes a per-phase file via this
# env-var so as not to overwrite the user's train_settings.json.

CONFIG_PATH = os.environ.get("TRAIN_SETTINGS_PATH",
                             os.path.join(PROJECT_ROOT, "train_settings.json"))

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"[OK] Configuration loaded from {CONFIG_PATH} / Configuración cargada desde {CONFIG_PATH}")
else:
    cfg = {}
    print(f"[!] {CONFIG_PATH} not found, using default values / No se encontró {CONFIG_PATH}, usando valores por defecto.")

# Advanced sidecar. The web UI rewrites train_settings.json with a fixed-key
# object, so advanced options live here, out of its reach.
# Precedence: train_settings.json > train_advanced.json > preset > DEFAULTS.

ADVANCED_PATH = os.environ.get("TRAIN_ADVANCED_PATH",
                               os.path.join(PROJECT_ROOT, "train_advanced.json"))
adv = {}
if os.path.exists(ADVANCED_PATH):
    try:
        with open(ADVANCED_PATH, "r", encoding="utf-8") as f:
            adv = json.load(f)
        print(f"[OK] Advanced settings merged from {ADVANCED_PATH} ({len(adv)} keys)")
    except Exception as exc:
        print(f"[!] Could not read {ADVANCED_PATH}: {exc} — ignoring / ignorando.")
        adv = {}

PRESET_NAME = str(cfg.get("preset", adv.get("preset", ""))).strip()
if PRESET_NAME and PRESET_NAME not in PRESETS:
    print(f"[!] Unknown preset '{PRESET_NAME}'. Valid: {', '.join(PRESETS)} — ignoring / ignorando.")
    PRESET_NAME = ""
preset_vals = PRESETS.get(PRESET_NAME, {})

_CFG_SOURCE = {}


def _cfg(key, default=None):
    """Resuelve una clave y anota su procedencia para el bloque EFFECTIVE CONFIG."""
    if key in cfg:
        src, val = "json", cfg[key]
    elif key in adv:
        src, val = "advanced", adv[key]
    elif key in preset_vals:
        src, val = "preset", preset_vals[key]
    elif key in DEFAULTS:
        src, val = "default", DEFAULTS[key]
    else:
        src, val = "default", default
    _CFG_SOURCE[key] = src
    return val


MODEL_ID          = _cfg("model_id")

# The local model lives in the root of the project. It is anchored only if the folder
# exists there, to avoid breaking the case of a Hugging Face repo-id; without this,
# running from another directory would trigger a complete model download.

if not os.path.isabs(MODEL_ID) and os.path.isdir(from_root(MODEL_ID)):
    MODEL_ID = from_root(MODEL_ID)
TOTAL_STEPS       = _cfg("total_steps")
BATCH_SIZE        = _cfg("batch_size")
GRAD_ACCUM_STEPS  = _cfg("grad_accum_steps")
LR                = _cfg("lr")
MIN_LR_RATIO      = _cfg("min_lr_ratio")
WARMUP_STEPS      = _cfg("warmup_steps")
LORA_RANK         = _cfg("lora_rank")
LORA_ALPHA        = _cfg("lora_alpha")
WEIGHT_DECAY      = _cfg("weight_decay")
MAX_GRAD_NORM     = _cfg("max_grad_norm")
SAVE_EVERY        = _cfg("save_every")
SEED              = _cfg("seed")
TIMESTEP_SAMPLING = _cfg("timestep_sampling")
PREVIEW_EVERY     = _cfg("preview_every")
PREVIEW_STEPS     = _cfg("preview_steps")
PREVIEW_CFG       = _cfg("preview_cfg")
PREVIEW_CAPTION_MODE = _cfg("preview_caption_mode")
TRIGGER_WORD      = _cfg("trigger_word")
PROJECT_NAME      = _cfg("project_name").strip()
LORA_TARGET       = str(_cfg("lora_target")).strip().lower()
COMPACT_TEXT      = bool(_cfg("compact_text"))
INIT_LORA_FROM    = str(_cfg("init_lora_from")).strip()
if INIT_LORA_FROM:
    INIT_LORA_FROM = from_root(INIT_LORA_FROM)
GRAD_CHECKPOINTING = bool(_cfg("gradient_checkpointing"))

# Identifies the progressive pipeline run to which a checkpoint belongs.
# Empty = standalone trainer, no verification (classic behavior).

RUN_ID            = str(_cfg("run_id")).strip()
# Phase context: only affects the progress line.
PHASE_INDEX       = int(_cfg("phase_index"))
PHASE_COUNT       = int(_cfg("phase_count"))
PHASE_LABEL       = str(_cfg("phase_label")).strip()
GLOBAL_STEP_OFFSET = int(_cfg("global_step_offset"))
GLOBAL_TOTAL_STEPS = int(cfg.get("global_total_steps", TOTAL_STEPS))
MULTIPHASE        = PHASE_COUNT > 1

# ── Added keys (Phases A–D). All with default = historical behavior ──

LORA_DTYPE_NAME   = str(_cfg("lora_dtype")).strip().lower()
HIGH_PREC_TARGETS = bool(_cfg("high_precision_targets"))
SAMPLER_MODE      = str(_cfg("sampler")).strip().lower()
NAN_GUARD         = bool(_cfg("nan_guard"))
NAN_ABORT_AFTER   = int(_cfg("nan_abort_after"))
MAX_LOSS          = float(_cfg("max_loss"))
OOM_GUARD         = bool(_cfg("oom_guard"))
OOM_ABORT_AFTER   = int(_cfg("oom_abort_after"))
RESUME_ON_CORRUPT = str(_cfg("resume_on_corrupt")).strip().lower()
WARMUP_UNITS      = str(_cfg("warmup_units")).strip().lower()
OPTIMIZER_NAME    = str(_cfg("optimizer")).strip().lower()
OPTIMIZER_EPS     = float(_cfg("optimizer_eps"))
OPTIMIZER_BETAS   = tuple(_cfg("optimizer_betas"))
TIMESTEP_WEIGHTING = str(_cfg("timestep_weighting")).strip().lower()
LOGIT_NORMAL_MU   = float(_cfg("logit_normal_mu"))
LOGIT_NORMAL_SIGMA = float(_cfg("logit_normal_sigma"))
SIGMA_MIN         = float(_cfg("sigma_min"))
SIGMA_MAX         = float(_cfg("sigma_max"))
CONTENT_OR_STYLE  = str(_cfg("content_or_style")).strip().lower()
NOISE_OFFSET      = float(_cfg("noise_offset"))
USE_EMA           = bool(_cfg("use_ema"))
EMA_DECAY         = float(_cfg("ema_decay"))
EMA_DEVICE        = str(_cfg("ema_device")).strip().lower()
LR_SCHEDULER      = str(_cfg("lr_scheduler")).strip().lower()
LR_NUM_CYCLES     = int(_cfg("lr_num_cycles"))
LR_STEP_GAMMA     = float(_cfg("lr_step_gamma"))
LR_STEP_COUNT     = int(_cfg("lr_step_count"))
VAL_SPLIT         = float(_cfg("val_split"))
VAL_CACHE_DIR     = str(_cfg("val_cache_dir")).strip()
VALIDATE_EVERY    = int(_cfg("validate_every"))
VALIDATION_SIGMAS = list(_cfg("validation_sigmas"))
VAL_SEED          = int(_cfg("val_seed"))
LOSS_WINDOW       = int(_cfg("loss_window"))
LOSS_DISPLAY      = str(_cfg("loss_display")).strip().lower()
CSV_LOG           = bool(_cfg("csv_log"))
MAX_CKPT_KEEP     = int(_cfg("max_checkpoints_to_keep"))
EXPORT_METADATA   = bool(_cfg("export_metadata"))
EXPORT_ALPHA_TENSORS = bool(_cfg("export_alpha_tensors"))
PREVIEW_SOURCE    = str(_cfg("preview_source")).strip().lower()
PREVIEW_WALK_SEED = bool(_cfg("preview_walk_seed"))
CAPTION_DROPOUT   = float(_cfg("caption_dropout_rate"))
DATASET_PATH      = from_root(str(_cfg("dataset_path")).strip())
CURATION_WEIGHTS  = bool(_cfg("curation_weights"))
if VAL_CACHE_DIR:
    VAL_CACHE_DIR = from_root(VAL_CACHE_DIR)


def _validate_choice(name, value, allowed, fallback):
    if value not in allowed:
        print(f"[!] Invalid {name} '{value}'. Using '{fallback}' / valor inválido. Usando '{fallback}'.")
        return fallback
    return value


LORA_TARGET       = _validate_choice("lora_target", LORA_TARGET, ("all", "attn", "attn+ff"), "all")
LORA_DTYPE_NAME   = _validate_choice("lora_dtype", LORA_DTYPE_NAME, ("bf16", "fp32"), "bf16")
SAMPLER_MODE      = _validate_choice("sampler", SAMPLER_MODE, ("epoch", "legacy"), "legacy")
RESUME_ON_CORRUPT = _validate_choice("resume_on_corrupt", RESUME_ON_CORRUPT, ("abort", "restart"), "abort")
WARMUP_UNITS      = _validate_choice("warmup_units", WARMUP_UNITS, ("updates", "micro_steps", "ratio"), "updates")
OPTIMIZER_NAME    = _validate_choice("optimizer", OPTIMIZER_NAME,
                                     ("adamw8bit_paged", "adamw8bit", "adamw"), "adamw8bit_paged")
TIMESTEP_WEIGHTING = _validate_choice("timestep_weighting", TIMESTEP_WEIGHTING,
                                      ("none", "bell", "half_bell"), "none")
CONTENT_OR_STYLE  = _validate_choice("content_or_style", CONTENT_OR_STYLE,
                                     ("balanced", "content", "style"), "balanced")
EMA_DEVICE        = _validate_choice("ema_device", EMA_DEVICE, ("cpu", "cuda"), "cpu")
LR_SCHEDULER      = _validate_choice("lr_scheduler", LR_SCHEDULER,
                                     ("cosine", "constant", "linear", "cosine_with_restarts", "step"), "cosine")
LOSS_DISPLAY      = _validate_choice("loss_display", LOSS_DISPLAY, ("window", "cumulative"), "cumulative")
PREVIEW_SOURCE    = _validate_choice("preview_source", PREVIEW_SOURCE, ("caption", "prompts"), "caption")

LORA_DTYPE  = torch.float32 if LORA_DTYPE_NAME == "fp32" else torch.bfloat16

# A1b: the model's compute dtype is a known constant, not something to be
# inferred from `next(model.parameters())` — that order depends on the PEFT wrap
# and quantization, and one day it would return a Params4bit (uint8).

MODEL_DTYPE = torch.bfloat16

# The text compaction removes the padding tokens from each caption, 
# allowing to dispense with the attention mask. With the mask + GQA, 
# PyTorch cannot use neither flash nor mem-efficient and falls back to the `math` backend, 
# which materializes the full matrix [B, heads, S, S] (~568 MB at 768x768). 
# Without the mask, it uses flash.
# It is only applicable with batch 1: when compacting, 
# each sample ends up with a different text length and torch.cat would stop working.

if COMPACT_TEXT and BATCH_SIZE > 1:
    print("[!] compact_text requires batch_size 1; disabling / compact_text requiere batch_size 1; desactivado.")
    COMPACT_TEXT = False

# Automatic folder formatting based on project name.
# Without project_name, explicit cache_dir/output_dir are respected: this is how
# run_progressive.py points each phase to its resolution subdir and to phaseN_*.

if PROJECT_NAME:
    CACHE_DIR  = os.path.join(CACHE_ROOT,  PROJECT_NAME)
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, PROJECT_NAME)
else:
    CACHE_DIR  = from_root(_cfg("cache_dir"))
    OUTPUT_DIR = from_root(_cfg("output_dir"))

# ── A5: unidades del warmup ─────────────────────────────────────────────────
# The schedule is measured in optimizer updates, but `total_steps` is in
# micro-steps. Mixing both units without saying so is the trap of this config:
# with the defaults (100 / 1200 / GA=4) the warmup consumes a third of the run.

TOTAL_UPDATES = max(1, TOTAL_STEPS / max(1, GRAD_ACCUM_STEPS))
if WARMUP_UNITS == "micro_steps":
    WARMUP_UPDATES = WARMUP_STEPS / max(1, GRAD_ACCUM_STEPS)
elif WARMUP_UNITS == "ratio":
    WARMUP_UPDATES = WARMUP_STEPS * TOTAL_UPDATES
else:
    WARMUP_UPDATES = float(WARMUP_STEPS)
if WARMUP_UPDATES >= TOTAL_UPDATES:
    print(f"[!] Warmup ({WARMUP_UPDATES:.0f} updates) >= total updates ({TOTAL_UPDATES:.0f}); "
          f"clamping to 10% / recortando al 10%.")
    WARMUP_UPDATES = 0.1 * TOTAL_UPDATES


def _print_effective_config():
    """Dumps each resolved value with its provenance.

    With four-level precedence (json > advanced > preset > default) and everything
    being opt-in, this is the only thing that makes "why what I set wasn't applied"
    debuggable. It goes to stdout, so the web UI shows it unchanged.
    """
    rows = [
        ("model_id",             MODEL_ID),
        ("project_name",         PROJECT_NAME or "(default)"),
        ("trigger_word",         TRIGGER_WORD),
        ("cache_dir",            CACHE_DIR),
        ("output_dir",           OUTPUT_DIR),
        ("total_steps",          f"{TOTAL_STEPS} micro-steps = {TOTAL_UPDATES:.0f} updates"),
        ("batch_size",           BATCH_SIZE),
        ("grad_accum_steps",     GRAD_ACCUM_STEPS),
        ("lr",                   LR),
        ("lr_scheduler",         f"{LR_SCHEDULER} (min_lr_ratio {MIN_LR_RATIO})"),
        ("warmup_steps",         f"{WARMUP_STEPS} {WARMUP_UNITS} = {WARMUP_UPDATES:.0f} updates "
                                 f"({100.0 * WARMUP_UPDATES / TOTAL_UPDATES:.1f}% of run)"),
        ("lora_rank",            f"{LORA_RANK} / alpha {LORA_ALPHA} (scale {LORA_ALPHA / max(1, LORA_RANK):.2f})"),
        ("lora_target",          LORA_TARGET),
        ("lora_dtype",           LORA_DTYPE_NAME),
        ("optimizer",            f"{OPTIMIZER_NAME} betas={list(OPTIMIZER_BETAS)} eps={OPTIMIZER_EPS} "
                                 f"wd={WEIGHT_DECAY}"),
        ("max_grad_norm",        MAX_GRAD_NORM),
        ("sampler",              SAMPLER_MODE),
        ("timestep_sampling",    TIMESTEP_SAMPLING),
        ("timestep_weighting",   TIMESTEP_WEIGHTING),
        ("content_or_style",     CONTENT_OR_STYLE),
        ("sigma_min",            f"{SIGMA_MIN} .. {SIGMA_MAX}"),
        ("noise_offset",         NOISE_OFFSET),
        ("high_precision_targets", HIGH_PREC_TARGETS),
        ("caption_dropout_rate", CAPTION_DROPOUT),
        ("curation_weights",     CURATION_WEIGHTS),
        ("use_ema",              f"{USE_EMA} (decay {EMA_DECAY}, {EMA_DEVICE})" if USE_EMA else False),
        ("compact_text",         COMPACT_TEXT),
        ("gradient_checkpointing", GRAD_CHECKPOINTING),
        ("nan_guard",            f"{NAN_GUARD} (abort after {NAN_ABORT_AFTER})" if NAN_GUARD else False),
        ("oom_guard",            f"{OOM_GUARD} (abort after {OOM_ABORT_AFTER})" if OOM_GUARD else False),
        ("max_loss",             MAX_LOSS or "off"),
        ("resume_on_corrupt",    RESUME_ON_CORRUPT),
        ("save_every",           SAVE_EVERY),
        ("max_checkpoints_to_keep", MAX_CKPT_KEEP or "keep all"),
        ("val_split",            f"{VAL_SPLIT} every {VALIDATE_EVERY} steps" if VALIDATE_EVERY else "off"),
        ("loss_display",         f"{LOSS_DISPLAY} (window {LOSS_WINDOW})"),
        ("csv_log",              CSV_LOG),
        ("preview_every",        f"{PREVIEW_EVERY} (source {PREVIEW_SOURCE})" if PREVIEW_EVERY else "off"),
        ("seed",                 SEED),
    ]
    if MULTIPHASE:
        rows.append(("phase", f"{PHASE_INDEX+1}/{PHASE_COUNT} ({PHASE_LABEL}²), global {GLOBAL_STEP_OFFSET}-{GLOBAL_STEP_OFFSET + TOTAL_STEPS}/{GLOBAL_TOTAL_STEPS}"))
    if RUN_ID:
        rows.append(("run_id", RUN_ID))
    width = max(len(k) for k, _ in rows)
    print("\n" + "=" * 78)
    print(f"EFFECTIVE CONFIG" + (f"  [preset: {PRESET_NAME}]" if PRESET_NAME else ""))
    print("=" * 78)
    for key, val in rows:
        print(f"  {key:<{width}}  {val}   [{_CFG_SOURCE.get(key, 'derived')}]")
    if INIT_LORA_FROM:
        print(f"  {'init_lora_from':<{width}}  {INIT_LORA_FROM}   [{_CFG_SOURCE.get('init_lora_from')}]")
    print("=" * 78 + "\n")


_print_effective_config()

os.makedirs(OUTPUT_DIR, exist_ok=True)
RESUME_DIR   = os.path.join(OUTPUT_DIR, "resume_checkpoint")
OPT_FILE     = os.path.join(OUTPUT_DIR, "optimizer.pt")
STEP_FILE    = os.path.join(OUTPUT_DIR, "current_step.txt")
RUN_ID_FILE  = os.path.join(OUTPUT_DIR, "run_id.txt")


def checkpoint_belongs_to_this_run():
    """Is the checkpoint in this folder from the current pipeline run?

    Without RUN_ID (standalone trainer) it's always accepted. With RUN_ID, a checkpoint from
    a previous run must be discarded: otherwise, the phase would restore it with
    start_step == total_steps, ignore init_lora_from and run an empty loop.
    """
    if not RUN_ID:
        return True
    try:
        with open(RUN_ID_FILE, "r", encoding="utf-8") as f:
            return f.read().strip() == RUN_ID
    except Exception:
        return False


torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def patch_attention_for_low_vram():
    """Avoid the `math` backend of SDPA when you need to preserve the mask.

    PyTorch does not support `attn_mask` together with `enable_gqa=True` in either flash or
    mem-efficient modes, so it falls back to `math`, which materializes the full matrix
    [B, heads, S, S]. By expanding K/V to the number of heads of Q, you can pass
    `enable_gqa=False` and mem-efficient becomes available again: the cost is
    a few dozen MB of K/V versus hundreds of MB of scores.

    Without a mask (see COMPACT_TEXT), flash already works with GQA and this doesn't apply.
    """
    from diffusers.models.transformers import transformer_krea2

    original = transformer_krea2.dispatch_attention_fn

    def dispatch(query, key, value, *args, attn_mask=None, enable_gqa=False, **kwargs):
        if attn_mask is not None and enable_gqa and key.shape[2] != query.shape[2]:
            repeats = query.shape[2] // key.shape[2]
            key = key.repeat_interleave(repeats, dim=2)
            value = value.repeat_interleave(repeats, dim=2)
            enable_gqa = False
        return original(query, key, value, *args, attn_mask=attn_mask, enable_gqa=enable_gqa, **kwargs)

    transformer_krea2.dispatch_attention_fn = dispatch
    print("[OK] Attention patched to avoid the SDPA math backend / Atención parcheada para evitar el backend math.")


def ensure_model_downloaded(local_path, repo_id):
    if os.path.exists(local_path) and os.path.isdir(local_path):
        has_content = any(
            os.path.exists(os.path.join(local_path, f))
            for f in ["index.json", "model_index.json", "config.json"]
        ) or len(os.listdir(local_path)) > 0
        if has_content:
            print(f"[OK] Local model found at / Modelo local encontrado en: {local_path}")
            return local_path

    print(f"⚠ Local model not found at / No se encontró modelo local en: {local_path}")
    print(f"  Downloading from Hugging Face / Descargando desde Hugging Face: {repo_id}")
    print(f"  This may take several minutes / Esto puede tardar varios minutos...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with / Se requiere 'huggingface_hub':\n"
            "  pip install huggingface_hub"
        )

    hf_token = cfg.get("hf_token") or os.environ.get("HF_TOKEN")
    dl_kwargs = {
        "repo_id": repo_id,
        "local_dir": local_path,
    }
    if hf_token:
        dl_kwargs["token"] = hf_token

    downloaded_path = snapshot_download(**dl_kwargs)

    print(f"[OK] Model downloaded to / Modelo descargado en: {downloaded_path}")
    return downloaded_path


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=6400,
                    base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def sample_sigma(batch_size, image_seq_len, device, shift_cfg):
    """Sample sigma for flow-matching, with the shift dependent on resolution.

    `content_or_style` biases the uniform before the shift (arXiv 2302.08453 §3.4):
    `content` (u³) concentrates in low sigma = fine detail and similarity; `style`
    (1−u³) in high sigma = composition and color.
    """
    if TIMESTEP_SAMPLING == "logit_normal":
        u = torch.sigmoid(LOGIT_NORMAL_MU + LOGIT_NORMAL_SIGMA
                          * torch.randn(batch_size, device=device))
    else:
        u = torch.rand(batch_size, device=device)
        if CONTENT_OR_STYLE == "content":
            u = u ** 3
        elif CONTENT_OR_STYLE == "style":
            u = 1.0 - u ** 3

    mu = calculate_shift(image_seq_len, *shift_cfg)
    e_mu = math.exp(mu)
    sigma = e_mu / (e_mu + (1.0 / u.clamp(1e-6, 1 - 1e-6) - 1.0))
    sigma = sigma.clamp(1e-4, 1.0 - 1e-4)
    if SIGMA_MIN > 0.0 or SIGMA_MAX < 1.0:
        sigma = sigma.clamp(max(1e-4, SIGMA_MIN), min(1.0 - 1e-4, SIGMA_MAX))
    return sigma


# ∫₀¹ exp(-2(s-0.5)²) ds and its half-bell variant. They normalize the mean
# weight to 1 so that switching schemes does not de facto alter the learning rate
# (without normalization, `bell` would inflate it by 17%).

_BELL_MEAN = 0.8556243918920983
_HALF_BELL_MEAN = 0.9278121959460491


def timestep_weight(sigma):
    """Timestep weighting BSMNTW/HBSMNTW of ai-toolkit, reexpressed over sigma ∈ (0,1).

    Normalized, the weight goes from 0.709 to 1.169: with batch 1 it's a modulation of
    ±20% of the LR per step, and its real effect comes from reweighting the micro-batches
    within the window of gradient accumulation.
    """
    if TIMESTEP_WEIGHTING == "bell":
        return torch.exp(-2.0 * (sigma - 0.5) ** 2) / _BELL_MEAN
    if TIMESTEP_WEIGHTING == "half_bell":
        bell = torch.exp(-2.0 * (sigma - 0.5) ** 2)
        return torch.where(sigma < 0.5, torch.ones_like(bell), bell) / _HALF_BELL_MEAN
    return torch.ones_like(sigma)


def pack_latents(x):
    B, C, H, W = x.shape
    x = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // 2) * (W // 2), C * 4)


def unpack_latents(x, H, W):
    B, _, C = x.shape
    x = x.view(B, H // 2, W // 2, C // 4, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, C // 4, H, W)


def prepare_position_ids(text_seq_len, grid_h, grid_w, device):
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    image_ids = torch.zeros(grid_h, grid_w, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_h, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_w, device=device)[None, :]
    return torch.cat([text_ids, image_ids.reshape(grid_h * grid_w, 3)], dim=0)


SKIP_QUANT = ("img_in", "time_embed", "time_mod_proj", "txt_in", "final_layer")


def quantize_to_nf4_(module, prefix=""):
    from bitsandbytes.nn import Linear4bit, Params4bit
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear) and not any(s in full for s in SKIP_QUANT):
            w = child.weight.data.float().contiguous()
            new_layer = Linear4bit(
                child.in_features, child.out_features,
                bias=child.bias is not None, quant_type="nf4",
                compute_dtype=torch.bfloat16,
            )
            new_layer.weight = Params4bit(w, requires_grad=False, quant_type="nf4")
            if child.bias is not None:
                new_layer.bias = torch.nn.Parameter(child.bias.data, requires_grad=False)
            setattr(module, name, new_layer)
            del child, w
        else:
            quantize_to_nf4_(child, full)


def load_nf4_cache_(transformer, cache_dir):
    index_path = os.path.join(cache_dir, "index.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.json not found in NF4 cache / No existe index.json en caché NF4: {cache_dir}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    quantized = index.get("quantized", {})
    weights_dir = os.path.join(cache_dir, "weights")
    replaced = 0

    def get_parent_module(root, module_name):
        parts = module_name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    for name, info in quantized.items():
        filepath = os.path.join(weights_dir, info["file"])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"NF4 weight file not found / No existe archivo NF4: {filepath}")

        parent, child_name = get_parent_module(transformer, name)

        with safe_open(filepath, framework="pt", device="cpu") as f:
            weight_data = f.get_tensor("weight")
            bias_data = None
            if info.get("bias", False):
                bias_data = f.get_tensor("bias")

            qs_dict = {}
            for key in f.keys():
                if not key.startswith("quant_state."):
                    continue
                qs_key = key[len("quant_state."):]
                qs_dict[qs_key] = f.get_tensor(key)

            packed_qs = {
                "absmax": qs_dict["absmax"],
                "nested_absmax": qs_dict["nested_absmax"],
                "nested_quant_map": qs_dict["nested_quant_map"],
                "quant_map": qs_dict["quant_map"],
                "quant_state.bitsandbytes__nf4": qs_dict["quant_state.bitsandbytes__nf4"],
            }

            quant_state = QuantState.from_dict(packed_qs, device="cpu")

        new_weight = Params4bit(
            weight_data,
            requires_grad=False,
            quant_type="nf4",
            quant_storage=torch.uint8,
        )
        new_weight.quant_state = quant_state
        new_weight.bnb_quantized = True

        new_layer = Linear4bit(
            info["in_features"],
            info["out_features"],
            bias=info["bias"],
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
        )
        new_layer.weight = new_weight
        if bias_data is not None:
            new_layer.bias = torch.nn.Parameter(bias_data, requires_grad=False)

        setattr(parent, child_name, new_layer)
        replaced += 1

    print(f"Reconstructed NF4 layers / Capas NF4 reconstruidas: {replaced}")

    verified = 0
    for name, layer in transformer.named_modules():
        if isinstance(layer, Linear4bit):
            if getattr(layer.weight, "bnb_quantized", False):
                if layer.weight.quant_state is not None:
                    verified += 1

    print(f"Verified NF4 layers / Capas NF4 verificadas: {verified}")

    if verified != replaced:
        raise RuntimeError("NF4 Verification mismatch / La verificación NF4 no coincide")

    print("[OK] NF4 cache loaded successfully / Caché NF4 cargada correctamente.")
    return transformer


def _lora_b_norm(model):
    """Global ‖lora_B‖. PEFT initializes lora_B to exact zero, so a value > 0
    proves that a weight load actually landed on the adapter."""
    total = 0.0
    for name, p in model.named_parameters():
        if "lora_B" in name:
            total += p.detach().float().pow(2).sum().item()
    return total ** 0.5


def load_lora_weights(model, blob, src):
    """Load LoRA weights into the adapter and verify that the load took effect.

    set_peft_model_state_dict uses load_state_dict(strict=False): if the keys don't
    match (rank/alpha/lora_target different from those in the checkpoint) it wouldn't
    raise any error and the training phase would start from a random initialization in silence. We prefer
    to fail hard: the orchestrator aborts the pipeline upon a non-zero exit code.
    """
    res = set_peft_model_state_dict(model, load(blob))
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    if unexpected:
        print(f"[!] {len(unexpected)} unexpected keys loading / claves inesperadas al cargar {src}")
        for k in unexpected[:5]:
            print(f"    - {k}")
        sys.exit(1)

    b_norm = _lora_b_norm(model)
    if b_norm == 0.0:
        print(f"[!] LoRA load was a no-op (‖lora_B‖ = 0) / the load didn't take effect: {src}")
        print("[!] Do the lora_rank/lora_alpha/lora_target match those in the checkpoint?")
        sys.exit(1)
    print(f"    ‖lora_B‖ = {b_norm:.4f}")
    return b_norm


def _export_lora(model, path, step=None, epoch=None, num_images=None):
    """Exports the LoRA in plain bf16 format, with training metadata.
    The safetensors metadata is a str→str map: no loader iterating over
    tensors can trip on it. Per-module `.alpha` tensors are opt-in,
    because adding keys that are not `lora_*` can confuse loaders
    expecting only A/B pairs.
    """
    clean = {}
    for k, v in model.state_dict().items():
        if "lora_" not in k:
            continue
        new_key = "transformer." + k.replace("base_model.model.", "")
        clean[new_key] = v.to(torch.bfloat16).cpu().contiguous()

    if EXPORT_ALPHA_TENSORS:
        alpha = torch.tensor(float(LORA_ALPHA), dtype=torch.bfloat16)
        for key in [k for k in clean if k.endswith("lora_A.default.weight")]:
            clean[key.replace("lora_A.default.weight", "alpha")] = alpha.clone()

    meta = {"format": "pt"}
    if EXPORT_METADATA:
        # `ss_network_alpha` matters: with rank 16 and alpha 32 the correct scale is
        # 2.0, but a loader without alpha information assumes alpha=rank and applies
        # 1.0, meaning the LoRA would run at half strength.
        meta.update({
            "ss_network_dim":           str(LORA_RANK),
            "ss_network_alpha":         str(LORA_ALPHA),
            "ss_network_module":        "peft.LoraConfig",
            "ss_base_model_version":    "krea2",
            "ss_output_name":           PROJECT_NAME or "krea2_lora",
            "ss_learning_rate":         str(LR),
            "ss_optimizer":             OPTIMIZER_NAME,
            "ss_lr_scheduler":          LR_SCHEDULER,
            "ss_seed":                  str(SEED),
            "ss_training_comment":      f"AcademiaSD LoRAlab-Krea2 target={LORA_TARGET} "
                                        f"dtype={LORA_DTYPE_NAME} ema={USE_EMA}",
            "modelspec.architecture":   "krea2/lora",
            "modelspec.implementation": "AcademiaSD_LoRAlab-Krea2",
            "modelspec.title":          PROJECT_NAME or "krea2_lora",
        })
        if num_images is not None:
            meta["ss_num_train_images"] = str(num_images)
        # Dummy entry `1_<trigger>`: it is the trick that makes the trigger word
        # visible in the metadata panels of ComfyUI and A1111.
        if TRIGGER_WORD:
            meta["ss_tag_frequency"] = json.dumps({f"1_{TRIGGER_WORD}": {TRIGGER_WORD: 1}})
        if step is not None:
            meta["training_info"] = json.dumps({"step": int(step),
                                                "epoch": int(epoch or 0)})

    save_file(clean, path, metadata=meta)


# ── UTILIDADES DE ESTADO / STATE UTILITIES ──────────────────────────────────

def _atomic_write(path, writer):
    """Writes via temporary file + os.replace (atomic on POSIX and Windows)."""
    tmp = f"{path}.tmp"
    try:
        writer(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def rotate_checkpoints(output_dir, keep):
    """Keeps only the most recent `keep` per-step checkpoints.

    Sorts by the parsed step number in the name, not by ctime: ctime lies after
    a copy or an rsync, the step number does not. It never touches the FINAL
    checkpoint.
    """
    if keep <= 0:
        return
    found = []
    for name in os.listdir(output_dir):
        match = re.fullmatch(r"Krea2_LoRA_step_(\d+)\.safetensors", name)
        if match:
            found.append((int(match.group(1)), os.path.join(output_dir, name)))
    found.sort(key=lambda pair: pair[0])
    for _, path in found[:-keep]:
        try:
            os.remove(path)
            print(f"  ↳ pruned old checkpoint / checkpoint antiguo eliminado: {os.path.basename(path)}")
        except OSError as exc:
            print(f"  [!] Could not prune {os.path.basename(path)}: {exc}")


def _curation_group(score, threshold, override):
    """Effective group of an image: "good" or "bad".

    Exact replica of resolve_curation_group() in scripts/python/server.py. If
    one changes, the other must change, or the UI would show a different split
    than what gets trained. Without a detected face (score None) it goes to the
    good group: non-scorable is not the same as bad.
    """
    if override in ("good", "bad"):
        return override
    if score is None or threshold is None:
        return "good"
    return "good" if score >= threshold else "bad"


def load_curation_weights(dataset_path, cache_names):
    """Weight per cache entry based on dataset curation.

    Returns (weights, summary) or (None, None) if there is nothing to apply: missing
    report, option disabled, or if all weights turn out to be 1.0 — in those cases,
    training must remain bit-for-bit as it was before this existed.

    The report is written by 0_curate_dataset.py and regenerated on every scan; the
    effective threshold and manual reassignments live in curation_overrides.json,
    owned by the UI, and override the automatic verdict.
    """
    if not CURATION_WEIGHTS:
        return None, None
    report_path = os.path.join(dataset_path, "curation_report.json")
    if not os.path.exists(report_path):
        return None, None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception as exc:
        print(f"[!] curation_report.json ilegible ({exc}) — se entrena sin pesos / training unweighted.")
        return None, None

    overrides = {}
    ovr_path = os.path.join(dataset_path, "curation_overrides.json")
    if os.path.exists(ovr_path):
        try:
            with open(ovr_path, "r", encoding="utf-8") as f:
                overrides = json.load(f) or {}
        except Exception as exc:
            print(f"[!] curation_overrides.json ilegible ({exc}) — se usa el umbral automático.")

    images = report.get("images") or {}
    if not images:
        return None, None
    manual = overrides.get("threshold")
    threshold = manual if isinstance(manual, (int, float)) else report.get("auto_threshold")
    group_ovr = overrides.get("groups") or {}
    weights_cfg = report.get("weights") or {}
    w_good = float(weights_cfg.get("good", 1.0))
    w_bad = float(weights_cfg.get("bad", 0.5))

    weights, counts, unscored = {}, {"good": 0, "bad": 0}, []
    for name in cache_names:
        # With flip_x the cache stores `img_001` and `img_001__flip` as separate entries,
        # but the curation scored the source image: without removing the suffix, half
        # the dataset would train at weight 1.0 in silence. Same strip used by the
        # orphan check below.
        stem = name[:-6] if name.endswith("__flip") else name
        entry = images.get(stem)
        if entry is None:
            # In the cache but not in the report: added after the last scan.
            # Full weight (never penalize for missing data) and warning.
            unscored.append(stem)
            weights[name] = 1.0
            continue
        group = _curation_group(entry.get("score"), threshold, group_ovr.get(stem))
        counts[group] += 1
        weights[name] = w_good if group == "good" else w_bad

    if unscored:
        uniq = sorted(set(unscored))
        print(f"[!] {len(uniq)} image(s) in the cache are not in curation_report.json "
              f"(añadidas tras el último scan) — entrenan a peso ×1.0: "
              + ", ".join(uniq[:8]) + ("…" if len(uniq) > 8 else ""))
        print("    re-run curation to include them.")

    if all(abs(w - 1.0) < 1e-9 for w in weights.values()):
        return None, None

    summary = {"good": counts["good"], "bad": counts["bad"],
               "w_good": w_good, "w_bad": w_bad, "threshold": threshold}
    return weights, summary


class EpochSampler:
    """Epoch-based sampling: each image is seen exactly once per epoch.

    Replaces the original `random.choice(bucket)` + `random.choice(image)` which
    gave each *bucket* the same probability regardless of how many images it contained:
    a bucket with 1 image would receive as much mass as one with 16.
    Measured in this repo, six individual images would take up 67% of the steps.

    Batches never mix buckets because different shapes cannot be concatenated.
    """

    def __init__(self, buckets, batch_size, seed, repeats=None):
        self.buckets = {k: sorted(v) for k, v in buckets.items()}
        self.batch_size = max(1, int(batch_size))
        self.repeats = repeats or {}
        self.rng = random.Random(seed)
        self.queue = []
        self.epoch = 0

    def _refill(self):
        self.epoch += 1
        batches = []
        for size, names in sorted(self.buckets.items()):
            pool = []
            for name in names:
                pool.extend([name] * max(1, int(self.repeats.get(name, 1))))
            self.rng.shuffle(pool)
            for i in range(0, len(pool), self.batch_size):
                chunk = pool[i:i + self.batch_size]
                if len(chunk) < self.batch_size:
                    # Pad the last batch of the bucket with samples from the bucket itself:
                    # filling from another bucket would break the torch.cat.
                    chunk = chunk + self.rng.choices(pool, k=self.batch_size - len(chunk))
                batches.append((size, chunk))
        self.rng.shuffle(batches)   # intercala resoluciones a lo largo de la época
        self.queue = batches

    def next(self):
        if not self.queue:
            self._refill()
        return self.queue.pop()

    def state_dict(self):
        state = self.rng.getstate()
        return {
            "epoch": self.epoch,
            "rng": json.dumps([state[0], list(state[1]), state[2]]),
            "queue": json.dumps([[list(size), names] for size, names in self.queue]),
        }

    def load_state_dict(self, sd):
        self.epoch = int(sd["epoch"])
        raw = json.loads(sd["rng"])
        self.rng.setstate((raw[0], tuple(raw[1]), raw[2]))
        self.queue = [(tuple(size), names) for size, names in json.loads(sd["queue"])]


class LegacySampler:
    """Historical sampling: uniform over buckets, with replacement.

    Only kept to reproduce old runs. See `EpochSampler` for the
    bias it introduces.
    """

    def __init__(self, buckets, batch_size, seed):
        self.buckets = {k: list(v) for k, v in buckets.items()}
        self.batch_size = max(1, int(batch_size))
        self.rng = random.Random(seed)
        self.epoch = 0

    def next(self):
        size = self.rng.choice(list(self.buckets))
        return size, [self.rng.choice(self.buckets[size]) for _ in range(self.batch_size)]

    def state_dict(self):
        state = self.rng.getstate()
        return {"epoch": 0, "rng": json.dumps([state[0], list(state[1]), state[2]]), "queue": "[]"}

    def load_state_dict(self, sd):
        raw = json.loads(sd["rng"])
        self.rng.setstate((raw[0], tuple(raw[1]), raw[2]))


class EMA:
    """Exponential moving average of trainable weights, in shadow fp32.

    `update()` must be called ONLY on real optimizer updates: doing so by
    micro-batch would convert the effective decay into `decay ** grad_accum_steps`.
    """

    def __init__(self, params, decay=0.99, device="cpu"):
        self.decay = float(decay)
        self.params = list(params)
        self.device = torch.device(device)
        self.shadow = [p.detach().to(self.device, torch.float32).clone() for p in self.params]
        self.backup = None
        self.updates = 0

    @torch.no_grad()
    def update(self):
        self.updates += 1
        # Decay warmup: without this, the first updates dominate the shadow model
        # for hundreds of steps and EMA starts biased toward initialization.
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for shadow, param in zip(self.shadow, self.params):
            shadow.mul_(decay).add_(param.detach().to(self.device, torch.float32),
                                    alpha=1.0 - decay)

    @torch.no_grad()
    def apply(self):
        """Installs EMA weights into the model, saving the live weights."""
        self.backup = [p.detach().clone() for p in self.params]
        for shadow, param in zip(self.shadow, self.params):
            param.copy_(shadow.to(param.device, param.dtype))

    @torch.no_grad()
    def restore(self):
        if self.backup is None:
            return
        for backup, param in zip(self.backup, self.params):
            param.copy_(backup)
        self.backup = None

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates,
                "shadow": [s.cpu() for s in self.shadow]}

    def load_state_dict(self, sd):
        self.updates = int(sd.get("updates", 0))
        loaded = sd.get("shadow") or []
        if len(loaded) != len(self.shadow):
            print("[!] EMA state size mismatch; reinitializing from current weights "
                  "(this is normal if the LoRA rank changed).")
            return
        for dst, src in zip(self.shadow, loaded):
            dst.copy_(src.to(dst.device, dst.dtype))


class VaeHolder:
    vae = None
    @classmethod
    def get(cls):
        if cls.vae is None:
            from diffusers import AutoencoderKLQwenImage
            cls.vae = AutoencoderKLQwenImage.from_pretrained(
                MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16)
        return cls.vae


def run_preview(model, scheduler, embed, mask, neg, size, step, shift_cfg,
                steps=None, cfg_scale=None, seed=None):
    H, W = size
    # The bucket must be a multiple of 16: gh/gw are H/16 and W/16, and pack_latents
    # divides latent dimensions by 2.
    H, W = max(16, (H // 16) * 16), max(16, (W // 16) * 16)
    steps = int(steps or PREVIEW_STEPS)
    cfg_scale = PREVIEW_CFG if cfg_scale is None else float(cfg_scale)
    gh, gw = H // 16, W // 16
    device = "cuda"
    was_training = model.training
    model.eval()

    g = torch.Generator(device=device).manual_seed(int(SEED if seed is None else seed))
    latents = torch.randn((1, 16, H // 8, W // 8), generator=g, device=device, dtype=torch.bfloat16)
    latents = pack_latents(latents)
    pos_ids = prepare_position_ids(embed.shape[1], gh, gw, device)
    embed = embed.to(device)
    mask = mask.to(device) if mask is not None else None
    neg_pos_ids = None
    if neg is not None:
        neg = (neg[0].to(device), neg[1].to(device) if neg[1] is not None else None)
        # When packed, the negative prompt has fewer tokens than the positive one,
        # so it needs its own position_ids.
        neg_pos_ids = (pos_ids if neg[0].shape[1] == embed.shape[1]
                       else prepare_position_ids(neg[0].shape[1], gh, gw, device))

    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    mu = calculate_shift(latents.shape[1], *shift_cfg)
    scheduler.set_timesteps(steps, device=device, sigmas=sigmas, mu=mu)

    with torch.no_grad():
        for t in scheduler.timesteps:
            tt = (t / scheduler.config.num_train_timesteps).expand(1).to(torch.bfloat16)
            pred = model(hidden_states=latents, encoder_hidden_states=embed, timestep=tt,
                         position_ids=pos_ids, encoder_attention_mask=mask, return_dict=False)[0]
            if neg is not None:
                pred_u = model(hidden_states=latents, encoder_hidden_states=neg[0], timestep=tt,
                               position_ids=neg_pos_ids, encoder_attention_mask=neg[1], return_dict=False)[0]
                pred = pred + cfg_scale * (pred - pred_u)
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]

        vae = VaeHolder.get().to(device)
        lat = unpack_latents(latents, H // 8, W // 8).to(vae.dtype).unsqueeze(2)
        mean = torch.tensor(vae.config.latents_mean, device=device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
        std  = torch.tensor(vae.config.latents_std,  device=device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
        img = vae.decode(lat * std + mean, return_dict=False)[0][:, :, 0]
        img = ((img.float() / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        vae.to("cpu")

    from PIL import Image
    out = os.path.join(OUTPUT_DIR, f"preview_step_{step}.png")
    Image.fromarray(img).save(out)
    print(f"\n  ↳ Preview saved to / Preview guardada: {out}")
    if was_training:
        model.train()
    free_vram()


# ── ENTRENAMIENTO / TRAINING ─────────────────────────────────────────────────
def train_krea2():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    patch_attention_for_low_vram()

    if not os.path.exists(CACHE_DIR) or not any(f.endswith("_latent.pt") for f in os.listdir(CACHE_DIR)):
        print(f"\n[!] ERROR: Cache directory '{CACHE_DIR}' is empty or does not exist.")
        print(f"[!] Please run Pre-Cache first! ")
        # Exit code ≠ 0: with `return` this was an invisible failure for
        # run_batch_cli.sh and run_progressive.py, which reported it as success.
        sys.exit(1)

    ensure_model_downloaded(
        local_path=MODEL_ID,
        repo_id="AcademiaSD/Krea-2-NF4-for-LoRA-Training"
    )

    print("Loading Krea-2 Transformer...")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        vae=None,
        text_encoder=None,
        torch_dtype=torch.bfloat16,
    )

    transformer = pipe.transformer
    scheduler   = pipe.scheduler

    del pipe
    free_vram()

    shift_cfg = (
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 6400),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )

    NF4_CACHE_DIR = MODEL_ID

    if os.path.exists(os.path.join(NF4_CACHE_DIR, "index.json")):
        print("\n¡NF4 CACHE DETECTED!")
        print("Skipping NF4 quantization...")
        t0 = time.time()
        transformer = load_nf4_cache_(transformer, NF4_CACHE_DIR)
        print(f"[NF4] Cache loaded in / Caché cargada en {time.time() - t0:.1f}s", flush=True)
        transformer.to("cuda")
        free_vram()
        print(f"Transformer 12B pinned in VRAM. Usage / Uso: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    else:
        print("\nNo NF4 cache found. Performing quantization... / No se encontró caché NF4. Ejecutando cuantización...")
        quantize_to_nf4_(transformer)
        transformer.to("cuda")
        free_vram()
        print(f"Transformer 12B pinned in VRAM. Usage / Uso: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    if GRAD_CHECKPOINTING:
        transformer.enable_gradient_checkpointing()
    else:
        # Without recompute: faster but uses more activation VRAM. Only suitable in
        # low-resolution stages where VRAM is plentiful (decided by the orchestrator).
        print("Gradient checkpointing OFF (faster, more VRAM)")

    all_linears = [name for name, m in transformer.named_modules()
                   if isinstance(m, (torch.nn.Linear, bnb.nn.Linear4bit))]

    def keep(name):
        # 'all' also includes the text_fusion blocks; reduced presets are limited
        # to image transformer_blocks, which is where the LoRA actually performs.
        if LORA_TARGET == "all":
            return True
        if not name.startswith("transformer_blocks."):
            return False
        if ".attn." in name:
            return True
        return LORA_TARGET == "attn+ff" and ".ff." in name

    target_modules = [n for n in all_linears if keep(n)]
    if not target_modules:
        print(f"[!] lora_target '{LORA_TARGET}' matched no layers; falling back to 'all' / no coincidió con ninguna capa.")
        target_modules = all_linears
    print(f"Target LoRA Layers / Capas LoRA objetivo: {len(target_modules)}/{len(all_linears)} ({LORA_TARGET})")

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
        target_modules=target_modules, use_dora=False, init_lora_weights=True,
    )

    model = get_peft_model(transformer, lora_config)

    # A1: dtype of the LoRA master weights. In bf16 (8-bit mantissa) the
    # relative epsilon is ~0.0039, so any update smaller than 0.39% of the
    # weight magnitude rounds down to nothing: at the end of the cosine decay,
    # with the LR already low, a large portion of updates simply fail to land.
    # PEFT casts the input to the dtype of lora_A and the result back, so fp32
    # works on top of the NF4 base without modifying it (cost: ~+235 MB of VRAM).
    for module in model.modules():
        if hasattr(module, "lora_A"):
            for adapter in module.lora_A.values():
                adapter.to(dtype=LORA_DTYPE)
        if hasattr(module, "lora_B"):
            for adapter in module.lora_B.values():
                adapter.to(dtype=LORA_DTYPE)
        if hasattr(module, "lora_embedding_A"):
            for adapter in module.lora_embedding_A.values():
                adapter.data = adapter.data.to(LORA_DTYPE)
        if hasattr(module, "lora_embedding_B"):
            for adapter in module.lora_embedding_B.values():
                adapter.data = adapter.data.to(LORA_DTYPE)

    model.print_trainable_parameters()
    print(f"LoRA master dtype / dtype de los pesos LoRA: {LORA_DTYPE_NAME}")

    def _make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    transformer.img_in.register_forward_hook(_make_inputs_require_grad)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if OPTIMIZER_NAME == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=LR, betas=OPTIMIZER_BETAS,
                                      eps=OPTIMIZER_EPS, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER_NAME == "adamw8bit":
        optimizer = bnb.optim.AdamW8bit(trainable, lr=LR, betas=OPTIMIZER_BETAS,
                                        eps=OPTIMIZER_EPS, weight_decay=WEIGHT_DECAY)
    else:
        # Paged: offloads optimizer state to RAM under VRAM pressure instead
        # of crashing with an OOM during high-resolution peaks.
        optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=LR, betas=OPTIMIZER_BETAS,
                                             eps=OPTIMIZER_EPS, weight_decay=WEIGHT_DECAY)

    ema = None
    if USE_EMA:
        ema = EMA(trainable, decay=EMA_DECAY, device=EMA_DEVICE)
        horizon = 1.0 / max(1e-9, 1.0 - EMA_DECAY)
        if horizon > TOTAL_UPDATES / 3.0:
            print(f"[!] ema_decay {EMA_DECAY} implies a ~{horizon:.0f}-update horizon but this "
                  f"run is only {TOTAL_UPDATES:.0f} updates: the EMA will barely leave its "
                  f"initialization / el EMA apenas saldrá de su inicialización.")
        print(f"[OK] EMA enabled / activada: decay {EMA_DECAY} on {EMA_DEVICE}")

    def lr_at(step):
        # The LR is only applied to actual updates of the optimizer (1 in every
        # GRAD_ACCUM_STEPS loop steps), so the schedule is measured in updates,
        # not loop steps. WARMUP_UPDATES is already normalized to updates.
        update = step / max(1, GRAD_ACCUM_STEPS)
        if update < WARMUP_UPDATES:
            return LR * update / max(1e-9, WARMUP_UPDATES)
        prog = min(1.0, (update - WARMUP_UPDATES) / max(1e-9, TOTAL_UPDATES - WARMUP_UPDATES))
        if LR_SCHEDULER == "constant":
            factor = 1.0
        elif LR_SCHEDULER == "linear":
            factor = 1.0 - prog
        elif LR_SCHEDULER == "cosine_with_restarts":
            factor = 0.5 * (1 + math.cos(math.pi * ((prog * max(1, LR_NUM_CYCLES)) % 1.0)))
        elif LR_SCHEDULER == "step":
            factor = LR_STEP_GAMMA ** int(prog * max(1, LR_STEP_COUNT))
        else:
            factor = 0.5 * (1 + math.cos(math.pi * prog))
        return LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * factor)

    # Configuration identity that the checkpoint must respect for restoration
    # to make sense. A change in rank/target causes the weights to no longer fit;
    # a change in cache or dtype invalidates the optimizer state.
    FINGERPRINT = json.dumps({
        "rank": LORA_RANK, "alpha": LORA_ALPHA, "target": LORA_TARGET,
        "cache_dir": CACHE_DIR, "lora_dtype": LORA_DTYPE_NAME,
        "batch_size": BATCH_SIZE, "optimizer": OPTIMIZER_NAME,
    }, sort_keys=True)

    # ── CHECKPOINT RESUME ─────────────────
    start_step = 0
    pending_state = None       # sampler/EMA/RNG: restored after the first step to avoid overwriting the LoRA weights
    lora_weights_path = os.path.join(RESUME_DIR, "adapter_model.safetensors")
    have_checkpoint = (os.path.exists(STEP_FILE) and os.path.exists(OPT_FILE)
                       and os.path.exists(lora_weights_path))
    if have_checkpoint and not checkpoint_belongs_to_this_run():
        print("=" * 65)
        print("[i] Checkpoint from a previous pipeline run; discarding it  ")
        print(f"    {RESUME_DIR}")
        print("=" * 65)
        have_checkpoint = False

    if have_checkpoint:
        print("=" * 65)
        print("¡Checkpoint detected! Restoring state... ")
        try:
            state = torch.load(OPT_FILE, weights_only=True)
            if not isinstance(state, dict) or "format_version" not in state:
                # Legacy format: optimizer.pt was the raw state_dict and the step
                # lived only in current_step.txt. Kept for backward compatibility to avoid
                # breaking in-flight runs.
                print("[i] Legacy checkpoint format : no RNG or sampler state.")
                with open(STEP_FILE, "r", encoding="utf-8") as f:
                    start_step = int(f.read().strip())
                optimizer.load_state_dict(state)
            else:
                start_step = int(state["step"])
                if state.get("fingerprint") != FINGERPRINT:
                    print("[!] WARNING: checkpoint fingerprint differs from current config.")
                    print(f"    ckpt = {state.get('fingerprint')}")
                    print(f"    now  = {FINGERPRINT}")
                optimizer.load_state_dict(state["optimizer"])
                if ema is not None and state.get("ema"):
                    ema.load_state_dict(state["ema"])
                pending_state = state

            with open(lora_weights_path, "rb") as f:
                load_lora_weights(model, f.read(), lora_weights_path)
            print(f"Resuming training from step {start_step}...")
        except Exception as exc:
            # Previously, this set start_step = 0 *after* loading weights: when
            # changing lora_rank and resuming, you would silently train an entire
            # run on top of pre-trained weights with the LR reset from warmup.
            print(f"[!] ERROR: checkpoint exists but could not be restored {exc}")
            if RESUME_ON_CORRUPT != "restart":
                print("[!] Refusing to silently restart from step 0. Delete the checkpoint or set "
                      "resume_on_corrupt='restart'")
                sys.exit(2)
            print("[!] resume_on_corrupt='restart': starting from step 0 anyway.")
            start_step = 0
            pending_state = None
        print("=" * 65)

        if start_step >= TOTAL_STEPS:
            print(f"[!] Checkpoint step {start_step} >= total_steps {TOTAL_STEPS}; nothing to do "
                  f"Increase total_steps to continue.")
            return

    # ── PHASE HAND-OFF (progressive resolution) ──────────────────────────────
    # When there is no native checkpoint for this phase but init_lora_from is
    # provided, ONLY the adapter weights from the previous phase are loaded.
    # The optimizer starts fresh (Adam momentum reset) and start_step=0: each
    # phase re-warms up on the new gradient scale instead of carrying over past inertia.
    if start_step == 0 and INIT_LORA_FROM:
        prev_adapter = os.path.join(INIT_LORA_FROM, "adapter_model.safetensors")
        if os.path.exists(prev_adapter):
            print("=" * 65)
            print(f"Phase hand-off: loading LoRA weights from previous phase\n  {prev_adapter}")
            with open(prev_adapter, "rb") as f:
                load_lora_weights(model, f.read(), prev_adapter)
            print("[OK] LoRA initialized from previous phase; optimizer starts fresh.")
            print("=" * 65)
        else:
            print(f"[!] init_lora_from set but adapter not found {prev_adapter}")
            sys.exit(1)

    last_step_executed = start_step
    sampler = None             # lo construye el cargador de caché, más abajo
    saving = {"busy": False, "done_on_exit": False}

    def save_checkpoint_now(current_s):
        """Saves the complete state atomically and non-reentrantly.

        Deliberate order: weights first, then state, and `current_step.txt`
        last. That file is the commit marker: if it exists, everything before it exists.
        """
        if current_s <= 0 or saving["busy"]:
            return
        saving["busy"] = True
        try:
            print(f"\nSaving checkpoint state at step {current_s}...")

            # save_pretrained escribe in-place: 
            stage = RESUME_DIR + ".stage"
            shutil.rmtree(stage, ignore_errors=True)
            os.makedirs(stage, exist_ok=True)
            model.save_pretrained(stage)
            os.makedirs(RESUME_DIR, exist_ok=True)
            for name in os.listdir(stage):
                os.replace(os.path.join(stage, name), os.path.join(RESUME_DIR, name))
            shutil.rmtree(stage, ignore_errors=True)

            _py_rng = random.getstate()
            state = {
                "format_version": 2,
                "step": int(current_s),
                "optimizer": optimizer.state_dict(),
                "sampler": sampler.state_dict() if sampler is not None else None,
                "ema": ema.state_dict() if ema is not None else None,
                # El RNG de python va como JSON y el de torch como ByteTensor,
                # para que torch.load(weights_only=True) siga funcionando.
                "rng_python": json.dumps([_py_rng[0], list(_py_rng[1]), _py_rng[2]]),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "fingerprint": FINGERPRINT,
            }
            _atomic_write(OPT_FILE, lambda p: torch.save(state, p))

            if RUN_ID:
                _atomic_write(RUN_ID_FILE, lambda p: open(p, "w", encoding="utf-8").write(RUN_ID))

            ckpt = os.path.join(OUTPUT_DIR, f"Krea2_LoRA_step_{current_s}.safetensors")
            # The LoRA that gets delivered is the EMA; resume_checkpoint retains the
            # raw weights to prevent double-smoothing on every restart.
            if ema is not None:
                ema.apply()
            try:
                _atomic_write(ckpt, lambda p: _export_lora(
                    model, p, step=current_s,
                    epoch=sampler.epoch if sampler is not None else 0,
                    num_images=len(cache_data)))
            finally:
                if ema is not None:
                    ema.restore()

            # Commit: the step is written last.
            _atomic_write(STEP_FILE, lambda p: open(p, "w", encoding="utf-8").write(str(current_s)))
            rotate_checkpoints(OUTPUT_DIR, MAX_CKPT_KEEP)
            print(f"✓ Checkpoint saved successfully at step {current_s}: {ckpt}")
        finally:
            saving["busy"] = False

    def handle_signal(sig, frame):
        print(f"\n[!] Signal received / Señal de detención recibida ({sig}).")
        save_checkpoint_now(last_step_executed)
        # The external handler would also save upon catching SystemExit; this flag
        # prevents double-saving.
        saving["done_on_exit"] = True
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, handle_signal)
        # Launched manually via SSH without tmux, closing the terminal sends SIGHUP:
        # save a checkpoint instead of just dying. (Doesn't reach via server: the
        # process runs in its own session.)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, handle_signal)
    except Exception:
        pass

    model.train()
    optimizer.zero_grad(set_to_none=True)

    pin = torch.cuda.is_available()
    cache_data, buckets = {}, defaultdict(list)

    def compact(emb, msk):
        """Retains only the real caption tokens and drops the mask.

        This is exact: text tokens do not use RoPE (prepare_position_ids assigns
        position 0 to all of them) even in text_fusion, so attention is
        permutation-equivariant over them, and their outputs are discarded in
        `hidden_states[:, text_seq_len:]`. The mask only served as key-padding,
        so dropping the padding tokens is equivalent to masking them, allowing SDPA
        to use flash attention without a mask.
        """
        idx = msk[0].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:      # caption sin tokens válidos: dejarlo como estaba
            return emb, msk
        return emb[:, idx].contiguous(), None

    def load_cache_entry(directory, name):
        lat = torch.load(f"{directory}/{name}_latent.pt", weights_only=True)
        emb = torch.load(f"{directory}/{name}_embed.pt",  weights_only=True)
        msk = torch.load(f"{directory}/{name}_mask.pt",   weights_only=True).bool()
        lat, emb = lat.to(torch.bfloat16), emb.to(torch.bfloat16)
        if COMPACT_TEXT:
            emb, msk = compact(emb, msk)
        if pin:
            lat, emb = lat.pin_memory(), emb.pin_memory()
            if msk is not None:
                msk = msk.pin_memory()
        return lat, emb, msk

    manifest_names = None
    manifest_path = os.path.join(CACHE_DIR, "cache_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_names = set(json.load(f).get("entries", {}))
        except Exception:
            manifest_names = None

    for f in sorted(os.listdir(CACHE_DIR)):
        if f.startswith(".") or not f.endswith("_latent.pt"):
            continue
        nombre = f.replace("_latent.pt", "")
        cache_data[nombre] = load_cache_entry(CACHE_DIR, nombre)
        buckets[(cache_data[nombre][0].shape[2], cache_data[nombre][0].shape[3])].append(nombre)

    # Orphans: latents from deleted or renamed images. Without a manifest they were
    # never cleaned up and kept being silently trained on.  
    if manifest_names is not None:
        # The manifest tracks the base name; mirrored variants are derived
        # from flip_x, so they are compared by their base name.
        orphans = sorted(name for name in cache_data
                         if (name[:-6] if name.endswith("__flip") else name) not in manifest_names)
        if orphans:
            print(f"\n[!] {len(orphans)} cached entries are not in cache_manifest.json "
                  f"(deleted or renamed source images?) / no están en el manifiesto:")
            for name in orphans[:10]:
                print(f"      {name}")
            if len(orphans) > 10:
                print(f"      ... and {len(orphans) - 10} more")
            print("[!] They ARE being trained on. Re-run Pre-Cache with prune_orphans='delete' "
                  "to remove them / SE están entrenando.")

    # ── C1: validation split ─────────────────────────────────────────────────
    # Deterministic (over sorted names) and applied BEFORE building the
    # sampler, so that the holdout receives no gradients.
    val_data, val_names = {}, []
    if VAL_CACHE_DIR and os.path.isdir(VAL_CACHE_DIR):
        for f in sorted(os.listdir(VAL_CACHE_DIR)):
            if f.startswith(".") or not f.endswith("_latent.pt"):
                continue
            name = f.replace("_latent.pt", "")
            val_data[name] = load_cache_entry(VAL_CACHE_DIR, name)
            val_names.append(name)
        print(f"[OK] Validation set from {VAL_CACHE_DIR}: {len(val_names)} images")
    elif VAL_SPLIT > 0:
        ordered = sorted(cache_data)
        stride = max(2, math.ceil(1.0 / VAL_SPLIT))
        # Original/flipped pairs must fall on the same side of the split.
        picked = {n[:-6] if n.endswith("__flip") else n for n in ordered[::stride]}
        val_names = [n for n in ordered
                     if (n[:-6] if n.endswith("__flip") else n) in picked]
        if len(val_names) >= len(ordered):
            print("[!] val_split would hold out the whole dataset; disabling.")
            val_names = []
        for name in val_names:
            val_data[name] = cache_data.pop(name)
            size = (val_data[name][0].shape[2], val_data[name][0].shape[3])
            buckets[size].remove(name)
            if not buckets[size]:
                del buckets[size]
        if val_names:
            print(f"[OK] Holdout split: {len(val_names)} validation / {len(cache_data)} training images")

    if not cache_data:
        print("[!] ERROR: no training images left after the validation split.")
        sys.exit(1)

    # ── Curaduría: peso por imagen ───────────────────────────────────────────
    # Se resuelve sobre el set de entrenamiento ya definitivo (post-split), para
    # que el recuento por grupo refleje lo que de verdad recibe gradiente.
    curation_w, curation_summary = load_curation_weights(DATASET_PATH, cache_data.keys())
    if curation_summary:
        thr = curation_summary["threshold"]
        print(f"\n[OK] Curaduría activa / curation active — umbral "
              f"{'auto' if thr is None else f'{thr * 100:.0f}%'}:")
        print(f"     Buena calificación : {curation_summary['good']:>4} imagen(es) · "
              f"peso ×{curation_summary['w_good']}")
        print(f"     Baja calificación  : {curation_summary['bad']:>4} imagen(es) · "
              f"peso ×{curation_summary['w_bad']}")

    neg = None
    if os.path.exists(f"{CACHE_DIR}/_neg_embed.pt"):
        neg_emb = torch.load(f"{CACHE_DIR}/_neg_embed.pt", weights_only=True)
        neg_msk = torch.load(f"{CACHE_DIR}/_neg_mask.pt",  weights_only=True).bool()
        if COMPACT_TEXT:
            neg_emb, neg_msk = compact(neg_emb, neg_msk)
        neg = (neg_emb, neg_msk)

    if CAPTION_DROPOUT > 0 and neg is None:
        print("[!] caption_dropout_rate is set but no _neg_embed.pt in the cache; ")
        CAPTION_DROPOUT_ACTIVE = 0.0
    else:
        CAPTION_DROPOUT_ACTIVE = CAPTION_DROPOUT

    pos_cache = {}
    def get_pos_ids(text_len, lh, lw):
        key = (text_len, lh, lw)
        if key not in pos_cache:
            pos_cache[key] = prepare_position_ids(text_len, lh // 2, lw // 2, "cuda")
        return pos_cache[key]

    # ── C5: preview prompts pre-encoded by the pre-cache ─────────────────────
    # The trainer does not have a text encoder loaded, so the prompts are
    # encoded in stage 1 and read here, just like the negative embedding.
    sample_prompts = []
    for i in range(64):
        emb_path = os.path.join(CACHE_DIR, f"_sample{i}_embed.pt")
        if not os.path.exists(emb_path):
            break
        s_emb = torch.load(emb_path, weights_only=True).to(torch.bfloat16)
        s_msk = torch.load(os.path.join(CACHE_DIR, f"_sample{i}_mask.pt"), weights_only=True).bool()
        if COMPACT_TEXT:
            s_emb, s_msk = compact(s_emb, s_msk)
        meta = {}
        meta_path = os.path.join(CACHE_DIR, f"_sample{i}_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        sample_prompts.append((s_emb, s_msk, meta))

    if PREVIEW_SOURCE == "prompts" and not sample_prompts:
        print("[!] preview_source='prompts' but no _sample*_embed.pt in the cache; "
              "add sample_prompts to pre_cache_settings.json and re-run Pre-Cache. "
              "Falling back to captions / usando captions.")
    elif sample_prompts:
        print(f"[OK] {len(sample_prompts)} sample prompts loaded for previews / "
              f"prompts de preview cargados.")

    all_preview_names = sorted(cache_data.keys())

    def get_preview_sample(step):
        if PREVIEW_CAPTION_MODE == "random":
            return random.choice(all_preview_names)
        elif PREVIEW_CAPTION_MODE == "rotate4":
            idx = (step // max(1, PREVIEW_EVERY)) % min(4, len(all_preview_names))
            return all_preview_names[idx]
        else:
            return all_preview_names[0]

    # ── A2: sampler ──────────────────────────────────────────────────────────
    if SAMPLER_MODE == "epoch":
        sampler = EpochSampler(buckets, BATCH_SIZE, SEED)
    else:
        sampler = LegacySampler(buckets, BATCH_SIZE, SEED)
        # The historical default is a measurable bug: warn with the real number.
        counts = sorted(len(v) for v in buckets.values())
        if len(counts) > 1 and counts[-1] > counts[0]:
            bias = counts[-1] / counts[0]
            print(f"\n[!] sampler='legacy' samples buckets uniformly, so an image in the "
                  f"smallest bucket ({counts[0]} img) gets {bias:.0f}x the gradient steps of one "
                  f"in the largest ({counts[-1]} img).")
            print(f"[!] Muestreo sesgado {bias:.0f}x. Use sampler='epoch' (or preset 'stable_v2') "
                  f"for full coverage / para cobertura completa.")

    # ── A6: restore RNG and sampler state after a resume ──────────────────────
    if pending_state is not None:
        try:
            if pending_state.get("sampler"):
                sampler.load_state_dict(pending_state["sampler"])
            raw = json.loads(pending_state["rng_python"])
            random.setstate((raw[0], tuple(raw[1]), raw[2]))
            torch.set_rng_state(pending_state["rng_torch"].to(torch.uint8))
            if torch.cuda.is_available() and pending_state.get("rng_cuda"):
                torch.cuda.set_rng_state_all([s.to(torch.uint8) for s in pending_state["rng_cuda"]])
            print(f"[OK] RNG and sampler state restored (epoch {sampler.epoch}) / "
                  f"estado RNG y del sampler restaurado.")
        except Exception as exc:
            print(f"[!] Could not restore RNG/sampler state: {exc} — continuing with a fresh ")

    # ── C1: validation loss ──────────────────────────────────────────────────
    @torch.no_grad()
    def validation_loss():
        """Loss in fixed sigmas and fixed noise per image.

        Fixing both is the whole trick: it converts a scalar dominated by the
        variance of sigma into a readable curve for deciding early stopping.
        """
        if ema is not None:
            ema.apply()
        was_training = model.training
        model.eval()
        total, count = 0.0, 0
        try:
            for name in val_names:
                lat, emb, msk = val_data[name]
                lat = lat.to("cuda", non_blocking=True)
                emb = emb.to("cuda", non_blocking=True)
                msk_c = msk.to("cuda", non_blocking=True) if (msk is not None and not COMPACT_TEXT) else None
                x = pack_latents(lat).float()
                pos = get_pos_ids(emb.shape[1], lat.shape[2], lat.shape[3])
                gen = torch.Generator(device="cuda").manual_seed(
                    VAL_SEED + (zlib.crc32(name.encode("utf-8")) & 0xFFFF))
                eps = torch.randn(x.shape, generator=gen, device="cuda", dtype=torch.float32)
                tgt = eps - x
                for s in VALIDATION_SIGMAS:
                    sig = torch.full((x.shape[0],), float(s), device="cuda")
                    noisy_v = ((1 - s) * x + s * eps).to(MODEL_DTYPE)
                    pred_v = model(hidden_states=noisy_v, encoder_hidden_states=emb,
                                   timestep=sig, position_ids=pos,
                                   encoder_attention_mask=msk_c, return_dict=False)[0]
                    total += F.mse_loss(pred_v.float(), tgt).item()
                    count += 1
        finally:
            if was_training:
                model.train()
            if ema is not None:
                ema.restore()
            free_vram()
        return total / max(1, count)

    # ── C2: logging  ─────────────────────────────────────────────
    train_log = val_log = train_log_file = val_log_file = None
    if CSV_LOG:
        train_log_path = os.path.join(OUTPUT_DIR, "train_log.csv")
        val_log_path   = os.path.join(OUTPUT_DIR, "val_log.csv")
        # Append: reanudar no debe perder la historia previa.
        new_train = not os.path.exists(train_log_path)
        new_val   = not os.path.exists(val_log_path)
        train_log_file = open(train_log_path, "a", newline="", encoding="utf-8")
        val_log_file   = open(val_log_path,   "a", newline="", encoding="utf-8")
        train_log = csv.writer(train_log_file)
        val_log   = csv.writer(val_log_file)
        if new_train:
            train_log.writerow(["step", "update", "epoch", "loss", "loss_avg", "grad_norm",
                                "lr", "sigma", "bucket_h", "bucket_w", "secs", "vram_peak_gb"])
        if new_val:
            val_log.writerow(["step", "update", "epoch", "val_loss"])

    loss_hist = collections.deque(maxlen=max(1, LOSS_WINDOW))
    nan_count = oom_streak = skipped_outlier = 0
    accum_count = 0

    def on_oom(step, size):
        """Discards the window, frees VRAM, and decides whether to give up.

        Returns True if the OOM should propagate (too many consecutive occurrences).
        """
        nonlocal oom_streak, accum_count
        oom_streak += 1
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)
        free_vram()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        print(f"\n[!] CUDA OOM at step {step} (bucket {size[0]}x{size[1]}) — batch skipped "
              f"({oom_streak}/{OOM_ABORT_AFTER} consecutive)")
        if oom_streak >= OOM_ABORT_AFTER:
            print("[!] Persistent OOM; saving checkpoint and aborting ")
            save_checkpoint_now(last_step_executed)
            return True
        return False
    running_loss, t_step_avg = 0.0, 0.0
    print(f"\nSTARTING TRAINING {len(cache_data)} images in "
          f"{len(buckets)} buckets, sampler '{SAMPLER_MODE}'.")

    try:
        for step in range(start_step + 1, TOTAL_STEPS + 1):
            last_step_executed = step
            t0 = time.time()

            size, names = sampler.next()
            latents = torch.cat([cache_data[n][0] for n in names]).to("cuda", non_blocking=True)
            masks   = None
            # D3: caption dropout reusing the empty prompt embedding that the
            # pre-cache already writes for the CFG of previews. get_pos_ids is
            # cached by (text_len, lh, lw), so the negative gets its own.
            if CAPTION_DROPOUT_ACTIVE > 0 and random.random() < CAPTION_DROPOUT_ACTIVE:
                # Los embeds son 4-D [B, T, 12, 2560]: se repite de forma agnóstica
                # al rango, y sólo cuando el batch lo exige (con batch 1 no se copia).
                neg_emb = neg[0].to("cuda", non_blocking=True)
                embeds = (neg_emb if len(names) == 1
                          else neg_emb.repeat(len(names), *([1] * (neg_emb.dim() - 1))))
                if not COMPACT_TEXT and neg[1] is not None:
                    neg_msk = neg[1].to("cuda", non_blocking=True)
                    masks = (neg_msk if len(names) == 1
                             else neg_msk.repeat(len(names), *([1] * (neg_msk.dim() - 1))))
            else:
                embeds = torch.cat([cache_data[n][1] for n in names]).to("cuda", non_blocking=True)
                if not COMPACT_TEXT:
                    masks = torch.cat([cache_data[n][2] for n in names]).to("cuda", non_blocking=True)

            try:
                latent_patched = pack_latents(latents)
                B, seq_img, _ = latent_patched.shape

                sigma  = sample_sigma(B, seq_img, "cuda", shift_cfg)
                # A10: in fp32 the target stops carrying the rounding errors of bf16. The
                # input to the model is still cast to MODEL_DTYPE.
                noise_dtype = torch.float32 if HIGH_PREC_TARGETS else latent_patched.dtype
                noise = torch.randn(latent_patched.shape, device=latent_patched.device,
                                    dtype=noise_dtype)
                if NOISE_OFFSET > 0:
                    # The noise is generated already packed, so channel c of the latent
                    # occupies indices c*4 ... c*4+3. See B3: discouraged in rectified
                    # flow, where sigma=1 is already pure noise in distribution.
                    channels = latents.shape[1]
                    offset = torch.randn((B, 1, channels), device=noise.device, dtype=noise.dtype)
                    noise = noise + NOISE_OFFSET * offset.repeat_interleave(4, dim=2)

                t_exp = sigma.view(-1, 1, 1).to(noise.dtype)
                base  = latent_patched.to(noise.dtype)
                noisy = ((1 - t_exp) * base + t_exp * noise).to(MODEL_DTYPE)
                target = noise - base

                pos_ids = get_pos_ids(embeds.shape[1], size[0], size[1])

                pred = model(
                    hidden_states=noisy,
                    encoder_hidden_states=embeds,
                    timestep=sigma,
                    position_ids=pos_ids,
                    encoder_attention_mask=masks,
                    return_dict=False,
                )[0]

                if TIMESTEP_WEIGHTING == "none" and curation_w is None:
                    raw_loss = F.mse_loss(pred.float(), target.float())
                    loss_value = raw_loss.detach()
                else:
                    per_sample = F.mse_loss(pred.float(), target.float(),
                                            reduction="none").mean(dim=(1, 2))
                    if TIMESTEP_WEIGHTING != "none":
                        per_sample = per_sample * timestep_weight(sigma.float())
                    # El guard de max_loss, el NaN guard y el CSV leen la loss SIN
                    # el peso de curaduría, por dos motivos: atenuar una imagen la
                    # volvería inmune al descarte de outliers, y la curva del log
                    # dejaría de ser comparable con la de un run sin curar.
                    loss_value = per_sample.mean().detach()
                    if curation_w is not None:
                    # Scaling a sample's loss IS a per-image LR. It is not
                    # renormalized to a mean of 1.0 per batch: that would negate the effect,
                    # and having a dampened batch contribute less to the accumulated
                    # update is precisely the point.
                        w = torch.tensor([curation_w.get(n, 1.0) for n in names],
                                         device=per_sample.device, dtype=per_sample.dtype)
                        per_sample = per_sample * w
                    raw_loss = per_sample.mean()
            except torch.cuda.OutOfMemoryError:
                if not OOM_GUARD or on_oom(step, size):
                    raise
                continue

            # ── A3: loss guards ──────────────────────────────────────────────
            if MAX_LOSS > 0 and torch.isfinite(loss_value) and loss_value.item() > MAX_LOSS:
                skipped_outlier += 1
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                print(f"\n[!] Loss {loss_value.item():.3f} > max_loss {MAX_LOSS} at step {step} "
                      f"— window discarded ({skipped_outlier} total)")
                continue
            if NAN_GUARD and not torch.isfinite(loss_value):
                nan_count += 1
                # Discard the entire window: a single inf would cause clip_grad_norm_
                # to calculate a NaN norm and scale ALL gradients to NaN,
                # and AdamW would permanently write NaN into exp_avg.
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                print(f"\n[!] Non-finite loss at step {step} — batch skipped "
                      f"({nan_count}/{NAN_ABORT_AFTER})")
                if nan_count >= NAN_ABORT_AFTER:
                    print("[!] Too many non-finite losses; saving and aborting / abortando.")
                    save_checkpoint_now(last_step_executed)
                    return
                continue

            try:
                # The backward pass is the actual memory peak, so it needs the
                # same protection as the forward pass.
                (raw_loss / GRAD_ACCUM_STEPS).backward()
            except torch.cuda.OutOfMemoryError:
                if not OOM_GUARD or on_oom(step, size):
                    raise
                continue

            step_loss = loss_value.item()
            running_loss += step_loss
            loss_hist.append(step_loss)
            accum_count += 1
            oom_streak = 0

            grad_norm = 0.0
            did_update = False
            if accum_count >= GRAD_ACCUM_STEPS:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
                if NAN_GUARD and not torch.isfinite(gnorm):
                    nan_count += 1
                    print(f"\n[!] Non-finite grad norm at step {step} — update skipped "
                          f"({nan_count}/{NAN_ABORT_AFTER})")
                    if nan_count >= NAN_ABORT_AFTER:
                        print("[!] Too many non-finite gradients; saving and aborting / abortando.")
                        save_checkpoint_now(last_step_executed)
                        return
                else:
                    grad_norm = gnorm.item()
                    for gparam in optimizer.param_groups:
                        gparam["lr"] = lr_at(step)
                    optimizer.step()
                    did_update = True
                    if ema is not None:
                        # Sólo en updates reales: por micro-batch el decay efectivo
                        # sería decay ** GRAD_ACCUM_STEPS.
                        ema.update()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0

            t_step     = time.time() - t0
            t_step_avg = t_step if t_step_avg == 0 else 0.1 * t_step + 0.9 * t_step_avg
            # In multi-stage, the bar, %, and ETA are global: the hand-off preserves
            # weights, so resetting them per stage would make it look like starting from scratch.
            done_steps  = GLOBAL_STEP_OFFSET + step
            total_shown = GLOBAL_TOTAL_STEPS if MULTIPHASE else TOTAL_STEPS
            eta_s      = (total_shown - done_steps) * t_step_avg
            eta        = f"{int(eta_s//3600):02d}:{int((eta_s%3600)//60):02d}:{int(eta_s%60):02d}"
            pct        = done_steps / max(1, total_shown)
            barra      = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))

            if LOSS_DISPLAY == "window":
                # The cumulative mean flattens out by design and hides
                # late movement; the sliding window actually shows it.
                avg_loss = sum(loss_hist) / max(1, len(loss_hist))
            else:
                avg_loss = running_loss / max(1, step - start_step)

            if MULTIPHASE:
                head = (f"[F{PHASE_INDEX+1}/{PHASE_COUNT} {PHASE_LABEL}²] "
                        f"Paso {step:4d}/{TOTAL_STEPS} · global {done_steps:5d}/{GLOBAL_TOTAL_STEPS}")
            else:
                head = f"Step/Paso {step:4d}/{TOTAL_STEPS}"
            progress_line = (
                f"{head} [{barra}] {pct*100:5.1f}% | "
                f"Loss {avg_loss:.4f} | gnorm {grad_norm:.3f} | "
                f"lr {lr_at(step):.2e} | ep {sampler.epoch} | {t_step_avg:.2f}s/it | ETA {eta}"
            )
            print(f"\r{progress_line}", end="", flush=True)

            if train_log is not None and did_update:
                train_log.writerow([
                    step, int(step / max(1, GRAD_ACCUM_STEPS)), sampler.epoch,
                    f"{step_loss:.6f}", f"{avg_loss:.6f}", f"{grad_norm:.4f}",
                    f"{lr_at(step):.3e}", f"{sigma.mean().item():.4f}",
                    size[0], size[1], f"{t_step:.3f}",
                    f"{torch.cuda.max_memory_allocated() / 1e9:.2f}",
                ])
                train_log_file.flush()

            if step % SAVE_EVERY == 0:
                print()
                save_checkpoint_now(step)

            if VALIDATE_EVERY > 0 and val_names and step % VALIDATE_EVERY == 0:
                vloss = validation_loss()
                print(f"\n  [Val] step {step} | val_loss {vloss:.4f} | {len(val_names)} images")
                if val_log is not None:
                    val_log.writerow([step, int(step / max(1, GRAD_ACCUM_STEPS)),
                                      sampler.epoch, f"{vloss:.6f}"])
                    val_log_file.flush()

            if PREVIEW_EVERY > 0 and step % PREVIEW_EVERY == 0:
                if PREVIEW_SOURCE == "prompts" and sample_prompts:
                    # Rotates through configured prompts; each can define its
                    # own size, seed, steps, and CFG.
                    idx = (step // PREVIEW_EVERY) % len(sample_prompts)
                    emb0, msk0, meta = sample_prompts[idx]
                    ref = cache_data[all_preview_names[0]][0]
                    size_px = (int(meta.get("height", ref.shape[2] * 8)),
                               int(meta.get("width",  ref.shape[3] * 8)))
                    p_seed = int(meta.get("seed", SEED)) + (step if PREVIEW_WALK_SEED else 0)
                    print(f"\n  [Preview] Prompt {idx}: {str(meta.get('prompt', ''))[:60]}")
                else:
                    p_name = get_preview_sample(step)
                    lat0, emb0, msk0 = cache_data[p_name]
                    meta, size_px, p_seed = {}, (lat0.shape[2] * 8, lat0.shape[3] * 8), SEED
                    print(f"\n  [Preview] Mode: {PREVIEW_CAPTION_MODE} | Sample: {p_name}")

                if ema is not None:
                    ema.apply()
                try:
                    run_preview(model, scheduler, emb0, msk0, neg, size_px, step, shift_cfg,
                                steps=int(meta.get("steps", PREVIEW_STEPS)),
                                cfg_scale=float(meta.get("cfg", PREVIEW_CFG)),
                                seed=p_seed)
                finally:
                    if ema is not None:
                        ema.restore()

    except torch.cuda.OutOfMemoryError:
        # Should not reach here (the internal guard captures it), but if the guard
        # is disabled it is advisable to save work before dying.        
        print(f"\n[!] CUDA OOM at step {last_step_executed}; saving checkpoint / guardando.")
        save_checkpoint_now(last_step_executed)
        raise
    except (KeyboardInterrupt, SystemExit):
        if not saving["done_on_exit"]:
            save_checkpoint_now(last_step_executed)
        return

    # A5: the final accumulation window was dropped if total_steps was not
    # a multiple of grad_accum_steps.
    if accum_count > 0:
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        if torch.isfinite(gnorm):
            for gparam in optimizer.param_groups:
                gparam["lr"] = lr_at(TOTAL_STEPS)
            optimizer.step()
            if ema is not None:
                ema.update()
        optimizer.zero_grad(set_to_none=True)
        print(f"\n[i] Flushed final partial accumulation window ({accum_count} micro-steps).")

    print("\n\nTraining completed! ")
    if nan_count or skipped_outlier:
        print(f"[i] Skipped batches: {nan_count} non-finite, {skipped_outlier} over max_loss.")
    if VALIDATE_EVERY > 0 and val_names:
        print(f"[i] Final validation loss: {validation_loss():.4f}")
    # Guardar también el resume_checkpoint al terminar: en el pipeline progresivo
    # es el hand-off de pesos que carga la fase siguiente vía init_lora_from.
    save_checkpoint_now(TOTAL_STEPS)
    final = os.path.join(OUTPUT_DIR, "Krea2_FINAL_LoRA.safetensors")
    if ema is not None:
        ema.apply()
    try:
        _atomic_write(final, lambda p: _export_lora(
            model, p, step=TOTAL_STEPS, epoch=sampler.epoch, num_images=len(cache_data)))
    finally:
        if ema is not None:
            ema.restore()
    print(f"✓ Final LoRA saved to {final}")


if __name__ == "__main__":
    train_krea2()