"""Incremental frozen embeddings keyed by image bytes and inference implementation."""

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, __version__ as pillow_version

from pettrace.data.curation import sha256
from pettrace.models import pipeline


def feature_cache(base_pipeline, rows, output, base_model_dir, batch=16, *, checkpoint=None):
    from pettrace.evaluation.core import file_hashes

    if not rows or batch < 1:
        raise ValueError("Nonempty rows and a positive batch size are required")
    identity = {"models": file_hashes(base_model_dir), "spec": base_pipeline.spec,
                "pipeline_sha256": sha256(pipeline.__file__), "implementation_sha256": sha256(__file__),
                "numpy": np.__version__, "onnxruntime": ort.__version__, "pillow": pillow_version}
    model_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    dataset = [{k: r[k] for k in ("animal", "species", "split", "sha256")} for r in rows]
    digest = hashlib.sha256(json.dumps({"model": model_key, "rows": dataset}, sort_keys=True).encode()).hexdigest()
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    # Labels/splits belong to the dataset digest; frozen image embeddings can be reused across volumes.
    features = np.empty((len(rows), 512), dtype=np.float32)
    with sqlite3.connect(directory / f"frozen-e2-{model_key}.sqlite3") as cache:
        cache.execute("CREATE TABLE IF NOT EXISTS embeddings (image TEXT PRIMARY KEY, vector BLOB, checksum TEXT)")
        missing = []
        for index, row in enumerate(rows):
            if sha256(row["path"]) != row["sha256"]:
                raise ValueError(f"Image changed since manifest: {row['path']}")
            saved = cache.execute("SELECT vector, checksum FROM embeddings WHERE image=?", (row["sha256"],)).fetchone()
            if saved is None:
                missing.append(index)
            else:
                vector, checksum = saved
                if len(vector) != 512 * 4 or hashlib.sha256(vector).hexdigest() != checksum:
                    raise ValueError("Feature cache checksum mismatch")
                features[index] = np.frombuffer(vector, dtype=np.float32)
        print(f"Frozen embeddings: {len(rows)-len(missing)} cached, {len(missing)} new", flush=True)
        for offset in range(0, len(missing), batch):
            indices = missing[offset:offset + batch]
            images = []
            for index in indices:
                with Image.open(rows[index]["path"]) as im:
                    images.append(ImageOps.exif_transpose(im).convert("RGB"))
            values = np.asarray(base_pipeline.embed(images), dtype=np.float32)
            if values.shape != (len(indices), 512) or not np.isfinite(values).all():
                raise ValueError("Invalid frozen encoder output")
            for index, value in zip(indices, values):
                features[index] = value
                vector = value.tobytes()
                cache.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                              (rows[index]["sha256"], vector, hashlib.sha256(vector).hexdigest()))
            cache.commit()  # Completed batches survive an interrupted extraction.
            if checkpoint:
                checkpoint()
            if offset % (batch * 20) == 0:
                print(f"New frozen embeddings: {offset+len(indices)}/{len(missing)}", flush=True)
    if not np.isfinite(features).all():
        raise ValueError("Invalid cached encoder output")
    return features, digest
