"""Cache frozen model embeddings with data and model fingerprints."""

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

from pettrace.data.curation import sha256


def feature_cache(base_pipeline, rows, output, base_model_dir, batch=16):
    import numpy as np

    from pettrace.evaluation.core import file_hashes

    # Bind the cache to every image, label/split, spec and exact frozen ONNX artifact.
    key = {"rows": [{k: r[k] for k in ["animal", "species", "split", "sha256"]} for r in rows],
           "models": file_hashes(base_model_dir), "spec": base_pipeline.spec}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    path = Path(output) / f"frozen-e2-{digest[:16]}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["fingerprint"].item()) != digest:
                raise ValueError("Feature cache fingerprint mismatch")
            return cached["features"], digest
    embeddings = []
    for i in range(0, len(rows), batch):
        images = []
        for row in rows[i:i + batch]:
            if sha256(row["path"]) != row["sha256"]:
                raise ValueError(f"Image changed since manifest: {row['path']}")
            with Image.open(row["path"]) as im:
                images.append(ImageOps.exif_transpose(im).convert("RGB"))
        embeddings.append(base_pipeline.embed(images))
        if i % (batch * 20) == 0:
            print(f"Frozen embeddings: {i + len(images)}/{len(rows)}", flush=True)
    features = np.concatenate(embeddings).astype(np.float32)
    if not np.isfinite(features).all() or features.shape != (len(rows), 512):
        raise ValueError("Invalid frozen encoder output")
    np.savez_compressed(path, features=features, fingerprint=np.asarray(digest))
    return features, digest
