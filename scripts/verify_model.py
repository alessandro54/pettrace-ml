"""Verify a downloaded `pettrace-embedder` version reproduces its golden fingerprint.

    python scripts/verify_model.py                                  # @production
    python scripts/verify_model.py models:/pettrace-embedder/1
    python scripts/verify_model.py /path/to/downloaded/model/dir    # offline copy

Recomputes the embeddings of the generated golden set with the version's own pipeline.yaml + ONNX
and compares them with the stored golden.npy. Exit 0 = same model behaviour, 1 = different.
Works for another developer with only the artifacts (no MLflow access needed for a local dir).
"""

import hashlib
import json
import os
import sys

import numpy as np

from reid.golden import compare, fingerprint, golden_images
from reid.pipeline import Pipeline


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "models:/pettrace-embedder@production"
    if os.path.isdir(target):
        model_dir = target
    else:
        import mlflow

        model_dir = mlflow.artifacts.download_artifacts(target)
    meta = json.load(open(os.path.join(model_dir, "golden.json")))
    stored = np.load(os.path.join(model_dir, "golden.npy"))
    fresh = Pipeline(model_dir).embed(golden_images())
    ok, min_cos, max_abs = compare(stored, fresh)
    sha = hashlib.sha256(open(os.path.join(model_dir, "encoder.onnx"), "rb").read()).hexdigest()
    print(f"model      {target}")
    print(f"encoder    sha256 {sha[:16]}…")
    print(
        f"golden     stored {meta['fingerprint']} · recomputed {fingerprint(fresh)} "
        f"({meta['golden_set']})"
    )
    print(f"agreement  min cosine {min_cos:.6f} · max |Δ| {max_abs:.1e}")
    print("RESULT     " + ("✓ same model behaviour" if ok else "✗ DIFFERENT model behaviour"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
