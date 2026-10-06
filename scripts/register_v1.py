"""Register the model running in production today as `pettrace-embedder` v1 (@production).

v1 = experiment E1: full image → CLIP ViT-B/32 image tower → visual projection → L2 norm → 512-d.
Deterministic and idempotent, so the same command reproduces v1 in any environment:

    MLFLOW_TRACKING_URI=<env> MLFLOW_TRACKING_USERNAME=… MLFLOW_TRACKING_PASSWORD=… \
        python scripts/register_v1.py

Steps:
  1. load CLIP at a pinned Hugging Face revision (the one vision serves today)
  2. export image tower + projection + L2 norm to ONNX (preprocessing stays outside the graph and
     is described exactly in pipeline.yaml)
  3. parity vs the production function reid.clip_reference.get_embedding, twice: ONNX fed by the HF
     processor, and the self-contained runner (pipeline.yaml preprocessing + ONNX, pipeline.py)
  4. golden fingerprint: embeddings of the generated golden set (golden.py) → golden.npy
  5. CPU latency of the ONNX encoder
  6. log run + artifacts (encoder.onnx, pipeline.yaml, thresholds.json, golden.npy, golden.json)
     to experiment `pettrace-embedder`, register, tag sha256/model_version/golden_fingerprint,
     set alias @production

v1 is STATIC: if a version tagged model_version=clip-vit-b32@v1 already exists, nothing new is
registered. The fresh golden embeddings are compared with the stored ones; if they match (within
golden.py tolerances) only the alias is (re)pointed, otherwise the script refuses and exits 1.

Runs inside the vision image (torch, transformers, cached weights, the `vision` package) with
`onnxruntime` and `mlflow-skinny` added — see docs/ml.md.
"""

import hashlib
import json
import os
import platform
import statistics
import tempfile
import time

import mlflow
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import transformers
import yaml
from mlflow import MlflowClient
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from reid.golden import GOLDEN_SET_VERSION, compare, fingerprint, golden_images
from reid.pipeline import Pipeline

MODEL_NAME = "pettrace-embedder"
EXPERIMENT = "pettrace-embedder"
ALIAS = os.environ.get("REGISTER_ALIAS", "production")

HF_REPO = "openai/clip-vit-base-patch32"
HF_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"  # served by vision since 2026-09
MODEL_VERSION = "clip-vit-b32@v1"  # tag already stored on every Qdrant point and pets row
OPSET = 17
PARITY_MIN_COSINE = 0.9999


class Encoder(torch.nn.Module):
    """Image tower + projection + L2 norm = reid.clip_reference.get_embedding after preprocessing."""

    def __init__(self, clip: CLIPModel):
        super().__init__()
        self.vision = clip.vision_model
        self.proj = clip.visual_projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.vision(pixel_values=pixel_values).pooler_output
        return F.normalize(self.proj(pooled), p=2, dim=-1)


def sample_images() -> list[Image.Image]:
    """Deterministic test images of different sizes and aspect ratios (preprocessing coverage)."""
    rng = np.random.default_rng(42)
    sizes = [(224, 224), (640, 480), (480, 640), (1600, 1200), (300, 900), (1024, 1024)]
    return [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for w, h in sizes]


def pipeline_spec(proc: CLIPProcessor) -> dict:
    ip = proc.image_processor
    return {
        "name": MODEL_NAME,
        "experiment": "E1",
        "model_version": MODEL_VERSION,
        "embedding_dims": 512,
        "normalized": "l2",
        "steps": [
            {"op": "decode", "mode": "RGB"},
            {"op": "detect_crop", "enabled": False},
            {
                "op": "clip_preprocess",
                "resize_shortest_edge": ip.size["shortest_edge"],
                "resample": "bicubic",
                "center_crop": [ip.crop_size["height"], ip.crop_size["width"]],
                "rescale_factor": ip.rescale_factor,
                "mean": list(ip.image_mean),
                "std": list(ip.image_std),
                "layout": "NCHW float32",
            },
            {"op": "onnx", "file": "encoder.onnx", "input": "pixel_values", "output": "embedding"},
        ],
        "source": {"hf_repo": HF_REPO, "hf_revision": HF_REVISION},
    }


def main() -> None:
    torch.manual_seed(0)
    clip = CLIPModel.from_pretrained(HF_REPO, revision=HF_REVISION).eval()
    proc = CLIPProcessor.from_pretrained(HF_REPO, revision=HF_REVISION)
    encoder = Encoder(clip).eval()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "encoder.onnx")
        dummy = torch.zeros(1, 3, 224, 224)
        torch.onnx.export(
            encoder,
            (dummy,),
            onnx_path,
            opset_version=OPSET,
            input_names=["pixel_values"],
            output_names=["embedding"],
            dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        )
        sha256 = hashlib.sha256(open(onnx_path, "rb").read()).hexdigest()
        size_mb = os.path.getsize(onnx_path) / 1e6

        # Parity against the production code path (same function vision serves).
        from reid.clip_reference import get_embedding

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        cosines, max_abs = [], 0.0
        for img in sample_images():
            ref = np.array(get_embedding(img), dtype=np.float32)
            pixels = proc(images=img, return_tensors="np")["pixel_values"].astype(np.float32)
            out = sess.run(["embedding"], {"pixel_values": pixels})[0][0]
            cosines.append(float(np.dot(ref, out)))
            max_abs = max(max_abs, float(np.max(np.abs(ref - out))))
        parity = min(cosines)
        if parity < PARITY_MIN_COSINE:
            raise SystemExit(f"parity failed: min cosine {parity:.6f} < {PARITY_MIN_COSINE}")

        # Self-contained runner (spec preprocessing, no HF) must also match production.
        with open(os.path.join(tmp, "pipeline.yaml"), "w") as f:
            yaml.safe_dump(pipeline_spec(proc), f, sort_keys=False)
        runner = Pipeline(tmp)
        imgs = sample_images()
        ref = np.array([get_embedding(i) for i in imgs], dtype=np.float32)
        runner_parity = float(np.min(np.sum(ref * runner.embed(imgs), axis=1)))
        if runner_parity < PARITY_MIN_COSINE:
            raise SystemExit(
                f"runner parity failed: min cosine {runner_parity:.6f} < {PARITY_MIN_COSINE}"
            )

        # Golden fingerprint, computed by the self-contained runner.
        golden = runner.embed(golden_images())
        golden_fp = fingerprint(golden)
        np.save(os.path.join(tmp, "golden.npy"), golden)

        # CPU latency of the encoder alone (single image), after warm-up.
        x = proc(images=sample_images()[1], return_tensors="np")["pixel_values"].astype(np.float32)
        for _ in range(3):
            sess.run(None, {"pixel_values": x})
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            sess.run(None, {"pixel_values": x})
            times.append((time.perf_counter() - t0) * 1000)
        p50 = statistics.median(times)
        p95 = sorted(times)[int(0.95 * len(times)) - 1]

        mlflow.set_experiment(EXPERIMENT)
        client = MlflowClient()

        # v1 is static: verify an existing v1 instead of registering a second one.
        try:
            existing = client.search_model_versions(
                f"name='{MODEL_NAME}' and tag.model_version='{MODEL_VERSION}'"
            )
        except mlflow.exceptions.MlflowException:
            existing = []
        if existing:
            v = existing[0]
            stored_dir = mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/{v.version}")
            stored = np.load(os.path.join(stored_dir, "golden.npy"))
            ok, min_cos, max_abs = compare(stored, golden)
            same_bytes = v.tags.get("sha256") == sha256
            if not ok:
                raise SystemExit(
                    f"REFUSED: {MODEL_VERSION} already registered as v{v.version} and this build "
                    f"differs (golden min cosine {min_cos:.6f}, max |Δ| {max_abs:.2e}). "
                    "v1 is static."
                )
            client.set_registered_model_alias(MODEL_NAME, ALIAS, v.version)
            print(
                f"v1 already registered as v{v.version} — verified identical "
                f"(golden min cosine {min_cos:.6f}, max |Δ| {max_abs:.1e}; "
                f"ONNX bytes {'identical' if same_bytes else 'differ, outputs identical'}). "
                f"Alias @{ALIAS} → v{v.version}."
            )
            return

        with open(os.path.join(tmp, "golden.json"), "w") as f:
            f.write(
                json.dumps(
                    {
                        "golden_set": GOLDEN_SET_VERSION,
                        "fingerprint": golden_fp,
                        "images": len(golden),
                        "generator": "ml/scripts/golden.py",
                        "runner": "ml/scripts/pipeline.py",
                        "tolerance": {"cosine_min": 0.9999, "max_abs_diff": 1e-3},
                    },
                    indent=2,
                )
            )
        with open(os.path.join(tmp, "thresholds.json"), "w") as f:
            f.write(
                '{"high": 0.84, "possible": 0.74, "calibrated": false, '
                '"note": "spike values; the two-cat exploration showed they do not separate identities"}\n'
            )

        with mlflow.start_run(run_name="v1-e1-clip-vit-b32") as run:
            mlflow.log_params(
                {
                    "experiment": "E1",
                    "hf_repo": HF_REPO,
                    "hf_revision": HF_REVISION,
                    "model_version": MODEL_VERSION,
                    "onnx_opset": OPSET,
                    "input": "1x3x224x224 float32",
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "onnxruntime": ort.__version__,
                }
            )
            mlflow.log_metrics(
                {
                    "parity_min_cosine": parity,
                    "parity_max_abs_diff": max_abs,
                    "runner_parity_min_cosine": runner_parity,
                    "onnx_size_mb": size_mb,
                    "cpu_latency_p50_ms": p50,
                    "cpu_latency_p95_ms": p95,
                }
            )
            mlflow.set_tags(
                {
                    "sha256": sha256,
                    "golden_fingerprint": golden_fp,
                    "git_sha": os.environ.get("GIT_SHA", "unknown"),
                    "host": platform.node(),
                }
            )
            for name in (
                "encoder.onnx",
                "pipeline.yaml",
                "thresholds.json",
                "golden.npy",
                "golden.json",
            ):
                mlflow.log_artifact(os.path.join(tmp, name), artifact_path="model")
            run_id = run.info.run_id

    try:
        client.create_registered_model(
            MODEL_NAME, description="Image → identity embedding pipeline served by vision."
        )
    except mlflow.exceptions.MlflowException:
        pass
    mv = client.create_model_version(
        MODEL_NAME,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
        tags={
            "sha256": sha256,
            "model_version": MODEL_VERSION,
            "experiment": "E1",
            "golden_fingerprint": golden_fp,
        },
        description=(
            "E1 baseline: full image, CLIP ViT-B/32, 512-d. "
            "What vision served before the registry."
        ),
    )
    client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
    print(f"registered {MODEL_NAME} v{mv.version} @{ALIAS}")
    print(f"  model_version {MODEL_VERSION} · sha256 {sha256[:12]}… · {size_mb:.0f} MB")
    print(
        f"  parity min cosine {parity:.6f} (max |Δ| {max_abs:.2e}) vs reid.clip_reference.get_embedding"
    )
    print(f"  self-contained runner parity {runner_parity:.6f} · golden fingerprint {golden_fp}")
    print(f"  ONNX CPU latency p50 {p50:.0f} ms · p95 {p95:.0f} ms")


if __name__ == "__main__":
    main()
