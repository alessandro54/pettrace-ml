"""Register experiment E2 as `pettrace-embedder` (@candidate): YOLOv8n crop → the same CLIP as v1.

E2 = detect the animal (YOLOv8n, COCO classes cat/dog, highest-confidence box, 10% margin; no box
→ full image) and embed the crop with v1's encoder, unchanged. Only the crop differs from E1, so a
comparison isolates its effect. The detector is an ONNX export of the COCO-pretrained yolov8n.pt
(Ultralytics 8.3, opset 17, 640×640), produced once on the host:

    uv run --group export --with "ultralytics==8.3.253" --with onnx --with onnxslim python -c \\
        "from ultralytics import YOLO; YOLO('notebooks/yolov8n.pt')\
            .export(format='onnx', imgsz=640, opset=17, simplify=True)"
    cp notebooks/yolov8n.onnx artifacts/yolov8n.onnx
    make register-e2

A re-export is not guaranteed to be byte-identical (onnx/onnxslim versions): the validated file is
the one inside pettrace-embedder v2 (detector sha256 da8f9b9e…); reuse it from there to reproduce.

Static like v1: if a version tagged model_version=clip-vit-b32-crop@v1 exists, its golden
embeddings are recomputed and compared; nothing new is registered. Licence: YOLOv8 weights are
AGPL-3.0 — fine for research; decide before serving E2 in production (docs/ml.md).
"""

import hashlib
import json
import os
import platform
import shutil
import statistics
import tempfile
import time

import mlflow
import numpy as np
import onnxruntime as ort
import yaml
from mlflow import MlflowClient

from reid.golden import GOLDEN_SET_VERSION, compare, fingerprint, golden_images
from reid.pipeline import Pipeline

MODEL_NAME = "pettrace-embedder"
EXPERIMENT = "pettrace-embedder"
ALIAS = os.environ.get("REGISTER_ALIAS", "candidate")
BASE = os.environ.get("BASE_MODEL", "models:/pettrace-embedder/1")  # E1: its encoder is reused
DETECTOR = os.environ.get("DETECTOR_ONNX", "artifacts/yolov8n.onnx")
MODEL_VERSION = "clip-vit-b32-crop@v1"
DETECT_STEP = {
    "op": "detect_crop",
    "enabled": True,
    "file": "detector.onnx",
    "model": "yolov8n (COCO, Ultralytics 8.3 export, opset 17)",
    "input_size": 640,
    "conf": 0.25,
    "classes": [15, 16],  # COCO cat, dog
    "margin": 0.10,
    "fallback": "full_image",
}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not os.path.exists(DETECTOR):
        raise SystemExit(f"detector not found at {DETECTOR} (see the module docstring)")
    base_dir = mlflow.artifacts.download_artifacts(BASE)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join(base_dir, "encoder.onnx"), os.path.join(tmp, "encoder.onnx"))
        shutil.copy(DETECTOR, os.path.join(tmp, "detector.onnx"))
        with open(os.path.join(base_dir, "pipeline.yaml")) as f:
            spec = yaml.safe_load(f)
        spec["model_version"] = MODEL_VERSION
        spec["experiment"] = "E2"
        spec["steps"] = [DETECT_STEP if s["op"] == "detect_crop" else s for s in spec["steps"]]
        with open(os.path.join(tmp, "pipeline.yaml"), "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False)

        encoder_sha = sha256_of(os.path.join(tmp, "encoder.onnx"))
        detector_sha = sha256_of(os.path.join(tmp, "detector.onnx"))
        runner = Pipeline(tmp)
        golden = runner.embed(golden_images())
        golden_fp = fingerprint(golden)
        np.save(os.path.join(tmp, "golden.npy"), golden)

        # CPU latency of the whole pipeline (detect + crop + encode) on one golden image.
        img = golden_images()[3]
        for _ in range(3):
            runner.embed([img])
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            runner.embed([img])
            times.append((time.perf_counter() - t0) * 1000)
        p50, p95 = statistics.median(times), sorted(times)[int(0.95 * len(times)) - 1]

        mlflow.set_experiment(EXPERIMENT)
        client = MlflowClient()
        existing = client.search_model_versions(
            f"name='{MODEL_NAME}' and tag.model_version='{MODEL_VERSION}'"
        )
        if existing:
            v = existing[0]
            stored = np.load(
                os.path.join(
                    mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/{v.version}"),
                    "golden.npy",
                )
            )
            ok, min_cos, max_abs = compare(stored, golden)
            if not ok:
                raise SystemExit(
                    f"REFUSED: {MODEL_VERSION} is v{v.version} and this build differs "
                    f"(golden min cosine {min_cos:.6f}, max |Δ| {max_abs:.2e}). E2 is static."
                )
            client.set_registered_model_alias(MODEL_NAME, ALIAS, v.version)
            print(
                f"E2 already registered as v{v.version} — verified identical. "
                f"Alias @{ALIAS} → v{v.version}."
            )
            return

        with open(os.path.join(tmp, "golden.json"), "w") as f:
            json.dump(
                {
                    "golden_set": GOLDEN_SET_VERSION,
                    "fingerprint": golden_fp,
                    "images": len(golden),
                    "runner": "reid.pipeline (= vision.ml.pipeline)",
                    "tolerance": {"cosine_min": 0.9999, "max_abs_diff": 1e-3},
                },
                f,
                indent=2,
            )
        with open(os.path.join(tmp, "thresholds.json"), "w") as f:
            f.write(
                '{"high": 0.84, "possible": 0.74, "calibrated": false, '
                '"note": "uncalibrated; E2 inflates all scores (two-cat exploration): '
                'calibrate at the EER per version"}\n'
            )

        with mlflow.start_run(run_name="v2-e2-yolov8n-crop-clip-vit-b32") as run:
            mlflow.log_params(
                {
                    "experiment": "E2",
                    "model_version": MODEL_VERSION,
                    "base_model": BASE,
                    "detector": DETECT_STEP["model"],
                    "detector_conf": DETECT_STEP["conf"],
                    "detector_classes": "15,16 (cat,dog)",
                    "crop_margin": DETECT_STEP["margin"],
                    "detector_input": DETECT_STEP["input_size"],
                    "fallback": "full_image",
                    "onnxruntime": ort.__version__,
                    "detector_licence": "AGPL-3.0 (Ultralytics YOLOv8)",
                }
            )
            mlflow.log_metrics(
                {
                    "cpu_latency_p50_ms": p50,
                    "cpu_latency_p95_ms": p95,
                    "detector_size_mb": os.path.getsize(os.path.join(tmp, "detector.onnx")) / 1e6,
                }
            )
            mlflow.set_tags(
                {
                    "sha256": encoder_sha,
                    "detector_sha256": detector_sha,
                    "golden_fingerprint": golden_fp,
                    "git_sha": os.environ.get("GIT_SHA", "unknown"),
                    "host": platform.node(),
                }
            )
            for name in (
                "encoder.onnx",
                "detector.onnx",
                "pipeline.yaml",
                "thresholds.json",
                "golden.npy",
                "golden.json",
            ):
                mlflow.log_artifact(os.path.join(tmp, name), artifact_path="model")
            run_id = run.info.run_id

    mv = client.create_model_version(
        MODEL_NAME,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
        tags={
            "sha256": encoder_sha,
            "detector_sha256": detector_sha,
            "model_version": MODEL_VERSION,
            "experiment": "E2",
            "golden_fingerprint": golden_fp,
        },
        description=(
            "E2: YOLOv8n animal crop (10% margin, full-image fallback) "
            "→ CLIP ViT-B/32 (v1 encoder), 512-d."
        ),
    )
    client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
    print(f"registered {MODEL_NAME} v{mv.version} @{ALIAS} · {MODEL_VERSION}")
    print(
        f"  encoder sha256 {encoder_sha[:12]}… (= v1) · detector sha256 {detector_sha[:12]}… "
        f"· golden {golden_fp}"
    )
    print(f"  pipeline CPU latency p50 {p50:.0f} ms · p95 {p95:.0f} ms")


if __name__ == "__main__":
    main()
