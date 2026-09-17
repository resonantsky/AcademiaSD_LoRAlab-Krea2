# -*- coding: utf-8 -*-
"""
0_curate_dataset.py — Dataset curation by face identity (runs before training)

Scores every dataset image by ArcFace similarity against 3 baseline images and
splits the dataset into two groups: good and low rating. The trainer applies
a different weight to each group (see curation_weights in 2_train_lora_krea2.py).
Nothing is deleted or moved: the low rating group still trains, but attenuated.

The 0_ prefix indicates that it runs BEFORE pre-caching: it decides the training weight
for each image, not what gets cached. It is CPU-only, so it does not compete for VRAM.

Reads configuration from pre_cache_settings.json (key "curation").
"""
import datetime
import json
import os
import sys

import numpy as np
from PIL import Image

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
    """Resolves a relative path against project root (absolute paths left intact)."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# ── DEFAULTS ────────────────────────────────────────────────────────────────
DEFAULTS = {
    "dataset_path": "./dataset",
    "curation": {
        "baselines": [],       # 3 images that best represent the target look
        "weight_good": 1.0,
        "weight_bad": 0.5,
    },
}

# The floor for the automatic threshold: below ~0.25 cosine similarity, ArcFace
# no longer considers them to be the same person. Serves as a baseline so a highly
# dispersed dataset doesn't end up sending valid images to the low group.
DIFFERENT_PERSON_FLOOR = 0.25

# Minimum number of scored faces required for outlier statistics to be meaningful.
MIN_SCORES_FOR_THRESHOLD = 4

REPORT_NAME = "curation_report.json"
CACHE_NAME = ".curation_cache.npz"

# ── LOAD CONFIG ─────────────────────────────────────────────────────────────
CONFIG_PATH = os.environ.get("PRECACHE_SETTINGS_PATH",
                             os.path.join(PROJECT_ROOT, "pre_cache_settings.json"))

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"✓ Configuration loaded from {CONFIG_PATH}")
else:
    cfg = {}
    print(f"⚠ {CONFIG_PATH} not found, using defaults.")

DATASET_PATH = from_root(cfg.get("dataset_path", DEFAULTS["dataset_path"]))

_cur = cfg.get("curation") or {}
BASELINES = list(_cur.get("baselines") or [])
WEIGHT_GOOD = float(_cur.get("weight_good", DEFAULTS["curation"]["weight_good"]))
WEIGHT_BAD = float(_cur.get("weight_bad", DEFAULTS["curation"]["weight_bad"]))


# =============================================================================
# FACE EMBEDDINGS
# =============================================================================

class FaceEmbedder:
    """ArcFace (InsightFace) embedding extractor for scoring identity similarity.

    Loads detection + recognition modules on CPU: curation runs before
    training, so it must not compete for VRAM. Embeddings are returned
    L2-normalized (`normed_embedding`), making cosine similarity a direct dot product.
    """

    def __init__(self):
        self._app = None

    def _ensure_loaded(self):
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            print("\n[!] InsightFace is not installed.")
            print("    Run ./install_LoRAlab-Krea2.sh, or install manually:")
            print("    pip install insightface onnxruntime opencv-python\n")
            sys.exit(1)
        print("Loading face model (first run downloads ~300 MB)...")
        self._app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        # ctx_id=-1 forces CPU execution.
        self._app.prepare(ctx_id=-1)

    def _detect_with_pad_retry(self, img_bgr):
        """Detects faces, retrying with an added border if none are found.

        RetinaFace (buffalo_l) fails on faces that FILL the frame: extreme close-ups
        are too large for its anchor scales and return empty results. Adding a
        border reduces the face's bounding box fraction back into the detector's
        useful range; it shifts coordinates without changing content or embeddings.
        This fallback only triggers when the initial pass fails, preventing regression.

        Without this retry, close-up shots—often the best dataset images—would be lost.
        """
        import cv2
        faces = self._app.get(img_bgr)
        if faces:
            return faces
        for pad in (0.25, 0.5):
            h, w = img_bgr.shape[:2]
            py, px = int(h * pad), int(w * pad)
            padded = cv2.copyMakeBorder(img_bgr, py, py, px, px, cv2.BORDER_REPLICATE)
            faces = self._app.get(padded)
            if faces:
                return faces
        return []

    def embed(self, image_path):
        """Embedding for the LARGEST face in the image, or None if no face is found.

        Loads via PIL (supports unicode paths, unlike cv2.imread) and converts
        to the BGR array expected by InsightFace.
        """
        self._ensure_loaded()
        import cv2
        try:
            with Image.open(image_path) as pil:
                img_bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            print(f"\n[!] Failed to read {os.path.basename(image_path)}: {exc}")
            return None
        faces = self._detect_with_pad_retry(img_bgr)
        if not faces:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = getattr(largest, "normed_embedding", None)
        return None if emb is None else np.asarray(emb, dtype=np.float32)


# =============================================================================
# EMBEDDING CACHE
# =============================================================================

def fingerprint(path):
    """Image fingerprint: mtime+size. Same criterion as file_fingerprint()
    in pre-cache — fast and reliable enough to know if re-embedding is required."""
    st = os.stat(path)
    return f"{int(st.st_mtime)}:{st.st_size}"


def load_cache(dataset_dir):
    """Pre-computed embeddings, indexed by filename.

    Returns (embeddings, fingerprints). An embedding with length 0 means
    "inspected, but no face detected" — cached as a hit to avoid repeating
    slow face detection steps on subsequent re-scans.
    """
    path = os.path.join(dataset_dir, CACHE_NAME)
    if not os.path.exists(path):
        return {}, {}
    try:
        with np.load(path, allow_pickle=False) as data:
            fps = json.loads(str(data["__fingerprints__"]))
            embs = {k: data[k] for k in data.files if k != "__fingerprints__"}
        return embs, fps
    except Exception:
        print("[i] Unreadable embedding cache; recomputing.")
        return {}, {}


def save_cache(dataset_dir, embs, fps):
    path = os.path.join(dataset_dir, CACHE_NAME)
    tmp = path + ".tmp.npz"
    try:
        np.savez_compressed(tmp, __fingerprints__=np.array(json.dumps(fps)), **embs)
        os.replace(tmp, path)
    except Exception as exc:
        print(f"[i] Could not save embedding cache ({exc}) — non-critical.")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


# =============================================================================
# CURATION
# =============================================================================

def resolve_baseline(name, dataset_dir):
    """A baseline can be defined as a stem ('img_003'), full filename
    ('img_003.png'), or path. Supported outside dataset: a clean reference photo
    is a valid baseline even if not used directly for training."""
    candidates = [name, os.path.join(dataset_dir, name)]
    for ext in IMAGE_EXTS:
        candidates.append(os.path.join(dataset_dir, name + ext))
    for c in candidates:
        resolved = from_root(c) if not os.path.isabs(c) else c
        if os.path.isfile(resolved):
            return resolved
    return None


def auto_threshold(scores):
    """Automatic cutoff calculation based on lower outliers:

        cutoff = max(median − 1.5 · IQR, 0.25)

    An absolute threshold is unreliable because each dataset varies in variance.
    A robust fence adapts dynamically; the lower floor prevents poor datasets
    from normalizing identity drift.

    Returns None when there aren't enough faces to produce meaningful stats —
    in that case, everything defaults to the good rating group.
    """
    ss = sorted(s for s in scores if s is not None)
    if len(ss) < MIN_SCORES_FOR_THRESHOLD:
        return None
    n = len(ss)
    med = ss[n // 2]
    q1, q3 = ss[n // 4], ss[(3 * n) // 4]
    return max(med - 1.5 * (q3 - q1), DIFFERENT_PERSON_FLOOR)


def curate():
    if not os.path.isdir(DATASET_PATH):
        print(f"[!] Dataset folder does not exist: {DATASET_PATH}")
        return 1

    # Top-level directory only, matching pre-cache behavior: images in subfolders
    # are not trained on, so they are not scored here.
    files = sorted(f for f in os.listdir(DATASET_PATH)
                   if not f.startswith(".")
                   and os.path.isfile(os.path.join(DATASET_PATH, f))
                   and f.lower().endswith(IMAGE_EXTS))
    if not files:
        print(f"[!] No images found in: {DATASET_PATH}")
        return 1

    if len(BASELINES) != 3:
        print(f"\n[!] Exactly 3 baselines required (found {len(BASELINES)}).")
        print("    Select 3 representative target images in the UI,")
        print("    or add them manually to pre_cache_settings.json:")
        print('      "curation": { "baselines": ["img_003", "img_017", "img_042"] }')
        print("\n    Using 3 averaged baselines prevents angle, expression,")
        print("    and lighting bias from dominating individual scores.\n")
        return 1

    resolved = [(b, resolve_baseline(b, DATASET_PATH)) for b in BASELINES]
    missing = [b for b, p in resolved if p is None]
    if missing:
        print(f"\n[!] Baselines not found: {', '.join(missing)}")
        return 1

    embedder = FaceEmbedder()
    embs, fps = load_cache(DATASET_PATH)
    cache_hits = 0

    def embed_cached(path, key):
        """Retrieves embedding via cache matching (mtime, size).
        Model loading and face detection are slow, so changing baselines or
        thresholds allows re-scoring within seconds."""
        nonlocal cache_hits
        fp = fingerprint(path)
        if key in embs and fps.get(key) == fp:
            cache_hits += 1
            cached = embs[key]
            return None if cached.size == 0 else cached
        emb = embedder.embed(path)
        embs[key] = np.zeros(0, dtype=np.float32) if emb is None else emb
        fps[key] = fp
        return emb

    # ── Baselines ────────────────────────────────────────────────────────────
    base_embs = []
    base_missing_face = []
    for name, path in resolved:
        # Cache using absolute paths if located outside dataset directory
        # to avoid key collisions with dataset images sharing the same name.
        key = os.path.basename(path) if os.path.dirname(path) == DATASET_PATH else path
        emb = embed_cached(path, key)
        if emb is None:
            base_missing_face.append(os.path.basename(path))
        base_embs.append(emb)

    if base_missing_face:
        print(f"\n[!] No face detected in baselines: {', '.join(base_missing_face)}")
        print("    Please select baseline images with clear, visible faces.\n")
        save_cache(DATASET_PATH, embs, fps)
        return 1

    # ── Scoring ──────────────────────────────────────────────────────────────
    print(f"Scoring {len(files)} image(s) against 3 baselines...")
    scores = {}
    for i, fname in enumerate(files, 1):
        path = os.path.join(DATASET_PATH, fname)
        stem = os.path.splitext(fname)[0]
        emb = embed_cached(path, fname)
        # Averaging 3 similarities is equivalent to computing similarity against
        # the unnormalized centroid of the baselines: prevents single shot bias.
        scores[stem] = None if emb is None else float(np.mean([float(np.dot(b, emb))
                                                               for b in base_embs]))
        print(f"\rScoring... {i}/{len(files)}", end="", flush=True)
    print()

    save_cache(DATASET_PATH, embs, fps)

    threshold = auto_threshold(scores.values())
    scored = [s for s in scores.values() if s is not None]
    no_face = len(scores) - len(scored)

    # ── Report ───────────────────────────────────────────────────────────────
    # Saved directly inside the dataset folder so scores travel with the dataset
    # and persist through project renames. Effective threshold and manual
    # overrides live in curation_overrides.json (owned by UI, preserved across re-runs).
    report = {
        "version": 1,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset_path": DATASET_PATH,
        "baselines": [os.path.basename(p) for _, p in resolved],
        "auto_threshold": threshold,
        "weights": {"good": WEIGHT_GOOD, "bad": WEIGHT_BAD},
        "images": {stem: {"score": (None if s is None else round(s, 6))}
                   for stem, s in scores.items()},
    }
    report_path = os.path.join(DATASET_PATH, REPORT_NAME)
    tmp = report_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, report_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    # Images with no detected faces go to GOOD group: non-scorable is not the same
    # as bad quality (back-facing shots or extreme profiles are valid training material).
    if threshold is None:
        good = len(scores)
        bad = 0
    else:
        bad = sum(1 for s in scores.values() if s is not None and s < threshold)
        good = len(scores) - bad

    print()
    print("=" * 70)
    print("  CURATION COMPLETE")
    print("=" * 70)
    print(f"  Scored images       : {len(scored)} of {len(scores)}"
          + (f" ({cache_hits} from cache)" if cache_hits else ""))
    if no_face:
        print(f"  No face detected    : {no_face} → GOOD group (non-scorable ≠ bad)")
    if threshold is None:
        print(f"  Automatic threshold : unavailable (requires at least {MIN_SCORES_FOR_THRESHOLD} "
              f"faces) → assigning all to good group")
    else:
        print(f"  Automatic threshold : {threshold * 100:.0f}%")
        print(f"  Good rating         : {good} image(s) · weight ×{WEIGHT_GOOD}")
        print(f"  Low rating          : {bad} image(s) · weight ×{WEIGHT_BAD}")
    print(f"  Report              : {report_path}")
    print("=" * 70)
    print("\nAdjust the threshold and reassign images in the UI before training.\n")
    return 0


if __name__ == "__main__":
    sys.exit(curate())