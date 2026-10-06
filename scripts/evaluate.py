"""Evaluate a `pettrace-embedder` version on a `pettrace-reid` dataset version (thesis protocol).

    python scripts/evaluate.py --model models:/pettrace-embedder@production --dataset v1
    python scripts/evaluate.py --model /path/to/model/dir --dataset /path/to/dataset/dir --no-mlflow
    make evaluate MODEL=models:/pettrace-embedder/1 DATASET=v2 PROTOCOL=zero-shot   # in the project image

Embeds with the version's own pipeline.yaml + ONNX (reid.pipeline = the app's vision.ml.pipeline, exactly
what vision serves), then on the dataset's splits:
  test  → Rank-1/5/10 (by animal) and mAP (by photo), per species + global, same-species gallery
  val   → match threshold at the EER; FAR/FRR of that threshold on test
Logs one MLflow run (experiment "pettrace-reid-eval") with the dataset as an input (repo, version,
Hub commit, fingerprint of metadata.csv), the model as params (URI, resolved registry version, sha256
of every ONNX file actually evaluated) and the code as tags (GIT_SHA/GIT_DIRTY, set by `make
evaluate`; ONNX Runtime version, CPU), so each number traces back to the exact data, model and
code. The promotion gate compares two such runs on the same dataset version.
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import tempfile

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # ml/ → reid
from reid import data, metrics  # noqa: E402

EXPERIMENT = "pettrace-reid-eval"


def file_hashes(mdir: str) -> dict[str, str]:
    """sha256 of every ONNX file of the evaluated pipeline: the model's identity in any registry."""
    out = {}
    for path in sorted(glob.glob(os.path.join(mdir, "**", "*.onnx"), recursive=True)):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[os.path.basename(path)] = h.hexdigest()
    return out


def registry_version(uri: str) -> str:
    """models:/name@alias or models:/name/N → the version number it points to (aliases move)."""
    if not uri.startswith("models:/"):
        return ""
    from mlflow import MlflowClient

    name, _, ref = uri[len("models:/"):].partition("@")
    if ref:
        return MlflowClient().get_model_version_by_alias(name, ref).version
    return name.rsplit("/", 1)[1] if "/" in name else ""


def model_dir(uri: str) -> str:
    if os.path.isdir(uri):
        return uri
    import mlflow

    return mlflow.artifacts.download_artifacts(uri)


def embed(pipeline, photos: list[data.Photo], batch: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(photos), batch):
        images = [ImageOps.exif_transpose(Image.open(p.path)).convert("RGB") for p in photos[i : i + batch]]
        out.append(pipeline.embed(images))
    return np.concatenate(out) if out else np.zeros((0, 0))


def evaluate(ds: data.Dataset, pipeline, zero_shot: bool = False) -> dict:
    result = {"dataset": ds.name, "splits": {}, "protocol": "zero-shot" if zero_shot else "split"}
    if zero_shot:
        return evaluate_zero_shot(ds, pipeline, result)
    threshold = float("nan")
    val = ds.split("val")
    if val:
        E = embed(pipeline, val)
        same, diff = metrics.pair_scores(E, [p.animal for p in val], [p.species for p in val])
        result["val"] = dict(zip(("eer", "threshold"), metrics.eer(same, diff)), pairs_same=len(same), pairs_diff=len(diff))
        threshold = result["val"]["threshold"]
    test = ds.split("test")
    if not test:
        raise SystemExit(f"{ds.name} has no test animals")
    E = embed(pipeline, test)
    animals, species = [p.animal for p in test], [p.species for p in test]
    result["test"] = metrics.retrieval(E, animals, species)
    same, diff = metrics.pair_scores(E, animals, species)
    result["test_threshold"] = metrics.at_threshold(same, diff, threshold)
    for row in result["test"]["per_query"]:
        row["photo"] = os.path.relpath(test[row["query"]].path, ds.root)
    return result


def evaluate_zero_shot(ds: data.Dataset, pipeline, result: dict) -> dict:
    """Pretrained models (E1, E2) never see the dataset, so every photo is a query whatever its
    split. The EER threshold is fitted on the same pairs it is reported on (optimistic: say so)."""
    photos = ds.photos
    E = embed(pipeline, photos)
    animals, species = [p.animal for p in photos], [p.species for p in photos]
    result["test"] = metrics.retrieval(E, animals, species)
    same, diff = metrics.pair_scores(E, animals, species)
    result["val"] = dict(zip(("eer", "threshold"), metrics.eer(same, diff)), pairs_same=len(same), pairs_diff=len(diff))
    result["test_threshold"] = metrics.at_threshold(same, diff, result["val"]["threshold"])
    for row in result["test"]["per_query"]:
        row["photo"] = os.path.relpath(photos[row["query"]].path, ds.root)
    return result


def flat_metrics(result: dict) -> dict[str, float]:
    out = {}
    for group, summary in result["test"].items():
        if group == "per_query":
            continue
        for k, v in summary.items():
            out[f"test.{group}.{k}"] = v
    for k, v in result.get("val", {}).items():
        out[f"val.{k}"] = v
    for k, v in result["test_threshold"].items():
        out[f"test.global.{k}_at_val_eer"] = v
    return {k: float(v) for k, v in out.items() if v == v}  # drop NaN


def report(result: dict) -> None:
    print(f"\n{result['dataset']}")
    print(f"{'group':8} {'queries':>7} {'gallery':>7} {'Rank-1':>7} {'Rank-5':>7} {'Rank-10':>7} {'mAP':>7}")
    for group, s in result["test"].items():
        if group == "per_query" or not s.get("queries"):
            continue
        print(f"{group:8} {s['queries']:7d} {s['gallery_animals']:7d} {s['rank1']:7.1%} {s['rank5']:7.1%} {s['rank10']:7.1%} {s['mAP']:7.3f}")
    if "val" in result:
        v, t = result["val"], result["test_threshold"]
        where = "all pairs (optimistic)" if result.get("protocol") == "zero-shot" else "val"
        print(f"EER on {where} {v['eer']:.1%} at threshold {v['threshold']:.3f} → FAR {t['far']:.1%}, FRR {t['frr']:.1%}")
    else:
        print("no val animals: threshold not calibrated")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="models:/pettrace-embedder@production")
    ap.add_argument("--dataset", required=True, help="version tag (v1) or a local dataset directory")
    ap.add_argument("--repo", default=data.REPO)
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--protocol", choices=("split", "zero-shot"), default="split",
                    help="zero-shot: every photo is a query (pretrained models only); EER on the same pairs")
    args = ap.parse_args()

    from reid.pipeline import Pipeline

    ds = data.from_dir(args.dataset) if os.path.isdir(args.dataset) else data.load(args.dataset, args.repo)
    mdir = model_dir(args.model)
    pipeline = Pipeline(mdir)
    result = evaluate(ds, pipeline, zero_shot=args.protocol == "zero-shot")
    report(result)
    if args.no_mlflow:
        return 0

    import mlflow
    import onnxruntime
    from mlflow.data.http_dataset_source import HTTPDatasetSource
    from mlflow.data.meta_dataset import MetaDataset

    mlflow.set_experiment(EXPERIMENT)
    model_version = pipeline.spec.get("model_version", "")
    with mlflow.start_run(run_name=f"{model_version or 'model'} on {ds.name}"):
        mlflow.log_input(MetaDataset(HTTPDatasetSource(ds.url), name=ds.name, digest=ds.fingerprint), context="evaluation")
        mlflow.log_params({
            "model.uri": args.model, "model.model_version": model_version,
            "dataset.repo": ds.repo, "dataset.version": ds.version, "dataset.commit": ds.commit,
            "dataset.fingerprint": ds.fingerprint, "protocol": f"{args.protocol}: same-species gallery, leave-one-out, "
            + ("EER on all pairs (optimistic)" if args.protocol == "zero-shot" else "EER on val"),
            "model.registry_version": registry_version(args.model),
            **{f"model.sha256.{name}": h for name, h in file_hashes(mdir).items()},
        })
        mlflow.set_tags({
            "dataset": ds.name, "model_version": model_version,
            "git_sha": os.environ.get("GIT_SHA", "unknown"), "git_dirty": os.environ.get("GIT_DIRTY", "unknown"),
            "onnxruntime": onnxruntime.__version__, "cpu": f"{platform.machine()} {platform.system()}",
        })
        mlflow.log_metrics(flat_metrics(result))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "evaluation.json")
            with open(path, "w") as f:
                json.dump(result, f, indent=2, default=float)
            mlflow.log_artifact(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
