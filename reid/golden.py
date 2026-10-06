# Mirror of apps/vision/src/vision/ml/golden.py in the PetTrace app repo; see reid/pipeline.py.

"""Golden fingerprint: fixed inputs whose embeddings identify a model version.

Every registered `pettrace-embedder` version stores `golden.npy` = the embeddings of
`golden_images()` produced by that version. Anyone can recompute them from the downloaded artifacts
and compare, independent of library versions or ONNX byte layout. Used by the vision service at
startup (self-test), `ml/scripts/register_v1.py` and `ml/scripts/verify_model.py` — keep one copy.

The images are generated, not stored, so the set is reproducible from this file alone. Changing
this generator changes every fingerprint: bump GOLDEN_SET_VERSION if you ever must.
"""

import hashlib

import numpy as np
from PIL import Image

GOLDEN_SET_VERSION = "golden-v1"
COSINE_MIN = 0.9999  # per image, recomputed vs stored
MAX_ABS_DIFF = 1e-3  # per component


def golden_images() -> list[Image.Image]:
    """12 deterministic RGB images: noise, gradients, shapes, solid colours, odd aspect ratios."""
    rng = np.random.default_rng(20261003)
    imgs = []
    for w, h in [(224, 224), (640, 480), (480, 640), (1600, 1200)]:
        imgs.append(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    x = np.linspace(0, 255, 512, dtype=np.float32)
    grad = np.stack(
        [np.tile(x, (384, 1)), np.tile(x[::-1], (384, 1)), np.full((384, 512), 128.0)], -1
    )
    imgs.append(grad.astype(np.uint8))
    checker = (np.indices((400, 400)).sum(axis=0) // 25 % 2 * 255).astype(np.uint8)
    imgs.append(np.stack([checker] * 3, -1))
    yy, xx = np.mgrid[:500, :500]
    disc = ((yy - 250) ** 2 + (xx - 250) ** 2 < 160**2).astype(np.uint8)
    imgs.append(np.stack([disc * 220, disc * 120 + 30, 255 - disc * 200], -1))
    for colour in [(200, 120, 40), (30, 30, 30), (240, 240, 240)]:
        imgs.append(np.full((300, 300, 3), colour, dtype=np.uint8))
    imgs.append(rng.integers(0, 256, (900, 300, 3), dtype=np.uint8))
    imgs.append(rng.integers(0, 256, (300, 1200, 3), dtype=np.uint8))
    return [Image.fromarray(a, "RGB") for a in imgs]


def fingerprint(embeddings: np.ndarray) -> str:
    """Short, stable id of a version's behaviour: hash of embeddings rounded to 3 decimals."""
    rounded = np.round(embeddings.astype(np.float64), 3).astype("<f4")
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def compare(stored: np.ndarray, fresh: np.ndarray) -> tuple[bool, float, float]:
    """(ok, min cosine, max |diff|) between stored golden embeddings and freshly computed ones."""
    cos = np.sum(stored * fresh, axis=1) / (
        np.linalg.norm(stored, axis=1) * np.linalg.norm(fresh, axis=1)
    )
    min_cos = float(cos.min())
    max_abs = float(np.abs(stored - fresh).max())
    return (min_cos >= COSINE_MIN and max_abs <= MAX_ABS_DIFF), min_cos, max_abs
