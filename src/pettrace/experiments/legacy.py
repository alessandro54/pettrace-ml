"""Legacy E3 protocol: public-validation calibration and private test-split evaluation.

This differs from the thesis notebooks' two-domain, all-eligible-photo evaluation.
"""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from pettrace.data import benchmark as data
from pettrace.evaluation import metrics
from pettrace.evaluation.core import embed, load_model
from pettrace.training.features import feature_cache


def evaluate_embeddings(features, rows, threshold):
    labels, species = [r["animal"] for r in rows], [r["species"] for r in rows]
    same, diff = metrics.pair_scores(features, labels, species)
    return {"retrieval": metrics.retrieval(features, labels, species),
            "verification_at_public_val_eer": metrics.at_threshold(same, diff, threshold)}


def run(manifest, base_uri, dataset_version, output, config=None, log_mlflow=True, require_gpu=True):
    if not log_mlflow:
        raise ValueError("MLflow publication is mandatory for every experiment run")

    import mlflow
    import torch

    from pettrace.training.triplet import Config, export_encoder, fast_eer, fit, project

    config = config or Config()
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab before running E3")
    output = Path(output)
    if (output / "head.pt").exists() or (output / "evaluation.json").exists():
        raise ValueError("Output already contains a run; choose a new directory to preserve provenance")
    output.mkdir(parents=True, exist_ok=True)
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if {r["species"] for r in rows} != {"cat", "dog"}:
        raise ValueError("Full E3 needs public identity data for both cats and dogs")
    by_id = {}
    for row in rows:
        by_id.setdefault(row["animal"], set()).add(row["split"])
    if any(len(s) != 1 for s in by_id.values()):
        raise ValueError("An identity crosses manifest splits")
    trainval = [r for r in rows if r["split"] in {"train", "val"}]
    public_test = [r for r in rows if r["split"] == "test"]
    if {r["species"] for r in public_test} != {"cat", "dog"}:
        raise ValueError("Public test must contain both species")
    base = load_model(base_uri)
    if not base.golden_ok:
        raise RuntimeError("Frozen E2 artifact failed its golden fingerprint check")
    if base.pipeline.spec.get("experiment") != "E2" or base.pipeline.detector is None:
        raise ValueError("E3 requires an explicit E2 detector-crop base, not E1")
    # The serving runner remains unchanged. Colab can select GPU sessions for feature extraction.
    import onnxruntime as ort

    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        base.pipeline.session.set_providers(providers)
        base.pipeline.detector.session.set_providers(providers)
    print("Encoder providers:", base.pipeline.session.get_providers())
    cache_dir = Path(manifest).parent / "feature-cache"
    features, cache_digest = feature_cache(base.pipeline, trainval, cache_dir, base.dir)
    (output / "data").mkdir(exist_ok=True)
    for name in ["manifest.csv", "audit.json", "sources.json"]:
        source = Path(manifest).parent / name
        if source.exists():
            shutil.copy2(source, output / "data" / name)
    manifest_sha = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    from pettrace.tracking import source_identity

    source = source_identity()
    provenance = {"experiment": "E3", "base_uri": base_uri, "base_sha256": base.sha256,
                  "manifest_sha256": manifest_sha, "feature_cache_sha256": cache_digest,
                  "source_sha256": source["source_sha256"], "dataset_version": dataset_version,
                  "torch": torch.__version__, "onnxruntime": ort.__version__,
                  "augmentation": "none; frozen E2 embeddings cached once",
                  "sampling": "same-species P x K, alternating species",
                  "selection": "public validation macro-species mAP only"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    active_run = None
    if log_mlflow:
        mlflow.set_experiment("pettrace-embedder")
        active_run = mlflow.start_run(run_name=f"e3-mlp-triplet-seed-{config.seed}")
        mlflow.log_params({**config.__dict__, "manifest_sha256": manifest_sha,
                           "base_uri": base_uri, "source_sha256": source["source_sha256"]})
        mlflow.set_tags({"experiment": "E3", "git_dirty": "true",
                         "source_sha256": source["source_sha256"], "training_source": "public_body_identities"})
        import pandas as pd

        for split, context in [("train", "training"), ("val", "validation")]:
            frame = pd.DataFrame([{k: r[k] for k in ["source", "animal", "species", "sha256"]}
                                  for r in trainval if r["split"] == split])
            mlflow.log_input(mlflow.data.from_pandas(frame, name=f"e3-public-{split}"), context=context)

    def callback(record):
        if log_mlflow:
            mlflow.log_metrics({k: v for k, v in record.items() if k != "epoch"}, step=record["epoch"])

    try:
        head, training = fit(features, trainval, output, config, callback=callback)
        (output / "training.json").write_text(json.dumps(training, indent=2))
        version_id = active_run.info.run_id[:12] if active_run else provenance["source_sha256"][:12] + f"-s{config.seed}"
        artifact_dir = output / "model"
        hashes = export_encoder(base.dir, head, artifact_dir, f"clip-vit-b32-crop-mlp-e3@{version_id}")
        val_idx = [i for i, r in enumerate(trainval) if r["split"] == "val"]
        validation_rows = [trainval[i] for i in val_idx]
        ev2, ev3 = features[val_idx], project(head, features[val_idx])
        calibration = {}
        for name, embeddings in [("E2", ev2), ("E3", ev3)]:
            positive, negative = metrics.pair_scores(embeddings, [r["animal"] for r in validation_rows],
                                                    [r["species"] for r in validation_rows])
            error, threshold = fast_eer(positive, negative)
            calibration[name] = {"eer": error, "threshold": threshold, "source": "public_val"}
        (artifact_dir / "thresholds.json").write_text(json.dumps({
            "high": calibration["E3"]["threshold"], "possible": calibration["E3"]["threshold"],
            "calibrated": True, "source": "public_val_eer", "note": "Single operating point; both levels share the EER threshold."}, indent=2))
        # The checkpoint is fixed before either public or local test is loaded/evaluated.
        test_features, _ = feature_cache(base.pipeline, public_test, cache_dir, base.dir)
        local = data.load(dataset_version)
        local_photos = local.split("test")
        if not local_photos:
            raise ValueError("Pinned local dataset has no test identities")
        public_hashes = {r["sha256"] for r in rows}
        from pettrace.data.curation import sha256

        if any(sha256(p.path) in public_hashes for p in local_photos):
            raise ValueError("An image is shared by public data and the local held-out test set")
        local_rows = [{"animal": p.animal, "species": p.species} for p in local_photos]
        local_features = embed(base.pipeline, local_photos, batch=8)
        results = {"experiment": "E3", "calibration": calibration, "base_sha256": base.sha256,
                   "e3_sha256": hashes, "public_test": {}, "local_test": {},
                   "local_dataset": {"repo": local.repo, "version": local.version, "commit": local.commit,
                                     "fingerprint": local.fingerprint, "photos": len(local_photos),
                                     "identities": len({p.animal for p in local_photos})}}
        for domain, frozen, evaluation_rows in [("public_test", test_features, public_test),
                                                ("local_test", local_features, local_rows)]:
            for name, embeddings in [("E2", frozen), ("E3", project(head, frozen))]:
                results[domain][name] = evaluate_embeddings(embeddings, evaluation_rows, calibration[name]["threshold"])
        # Real-image parity after export, in addition to the synthetic golden set.
        from pettrace.models.pipeline import Pipeline

        images = []
        for photo in local_photos[:5]:
            with Image.open(photo.path) as im:
                images.append(ImageOps.exif_transpose(im).convert("RGB"))
        served = Pipeline(str(artifact_dir)).embed(images)
        reference = project(head, local_features[:len(images)])
        if not np.allclose(served, reference, atol=1e-3):
            raise RuntimeError("E3 serving parity failed on real test images")
        (output / "evaluation.json").write_text(json.dumps(results, indent=2))
        if log_mlflow:
            for domain in ["public_test", "local_test"]:
                for name in ["E2", "E3"]:
                    for sp, values in results[domain][name]["retrieval"].items():
                        if sp == "per_query":
                            continue
                        mlflow.log_metrics({f"{domain}.{name}.{sp}.{k}": float(v) for k, v in values.items()})
            mlflow.log_artifacts(str(artifact_dir), artifact_path="model")
            for filename in ["training.json", "history.json", "evaluation.json", "provenance.json"]:
                mlflow.log_artifact(str(output / filename))
            mlflow.log_artifacts(str(output / "data"), artifact_path="data")
            (output / "mlflow_run.json").write_text(json.dumps({"run_id": active_run.info.run_id,
                "model_uri": f"runs:/{active_run.info.run_id}/model"}, indent=2))
        return results
    finally:
        if active_run:
            mlflow.end_run(status="FINISHED" if (output / "evaluation.json").exists() else "FAILED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base", default="models:/pettrace-embedder/2")
    parser.add_argument("--dataset", default="v3")
    parser.add_argument("--output", default="reports/e3_first_run")
    args = parser.parse_args()
    run(args.manifest, args.base, args.dataset, args.output, log_mlflow=True)
