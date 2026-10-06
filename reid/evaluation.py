"""The thesis evaluation protocol on a pinned `pettrace-reid` version, shared by the experiment
notebooks (notebooks/eK_*.ipynb) and the CLI (scripts/evaluate.py).

    model = load_model("models:/pettrace-embedder/1")        # download + identity + golden check
    ev = evaluate(data.load("v3"), model.pipeline, zero_shot=True)
    run_id = log_run(ev, model, experiment="E1", artifacts="reports/e1_v3")

Protocols:
  split      queries = `test` photos, gallery = the other `test` photos of the same species
             (leave-one-out), threshold at the EER on `val`, FAR/FRR on `test`. Trained models.
  zero-shot  every photo is a query whatever its split; the EER threshold is fitted on the same
             pairs it is reported on (optimistic). Pretrained models only (E1, E2).
Each animal scores with its best photo, as the app does. Metrics: Rank-1/5/10 per animal, mAP per
photo, per species and global, always with the gallery size.
"""

import glob
import hashlib
import json
import os
import platform
import tempfile
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from reid import data, metrics
from reid.golden import compare, fingerprint, golden_images
from reid.pipeline import Pipeline

EXPERIMENT = "pettrace-reid-eval"


@dataclass
class Model:
    uri: str
    dir: str
    pipeline: Pipeline
    model_version: str  # e.g. clip-vit-b32@v1: the tag stored with every vector
    registry_version: str  # the registry's version number (local to one MLflow server)
    sha256: dict[str, str]  # per ONNX file: the identity that holds across registries
    golden_ok: bool
    golden_fingerprint: str


@dataclass
class Evaluation:
    dataset: data.Dataset
    protocol: str
    result: dict  # test/val/test_threshold metrics + per_query rows
    photos: list[data.Photo]  # the photos that were embedded (queries and gallery)
    embeddings: np.ndarray
    same: np.ndarray  # cosine scores of same-animal pairs (same species)
    diff: np.ndarray  # cosine scores of different-animal pairs (same species)


def file_hashes(mdir: str) -> dict[str, str]:
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

    name, _, ref = uri[len("models:/") :].partition("@")
    if ref:
        return MlflowClient().get_model_version_by_alias(name, ref).version
    return name.rsplit("/", 1)[1] if "/" in name else ""


def model_dir(uri: str) -> str:
    if os.path.isdir(uri):
        return uri
    import mlflow

    return mlflow.artifacts.download_artifacts(uri)


def load_model(uri: str) -> Model:
    """Download a registered version and check it is the model it claims to be: the golden set
    re-embedded with its own pipeline must match the stored golden embeddings."""
    mdir = model_dir(uri)
    pipeline = Pipeline(mdir)
    stored = np.load(os.path.join(mdir, "golden.npy"))
    fresh = pipeline.embed(golden_images())
    ok, _, _ = compare(stored, fresh)
    return Model(uri, mdir, pipeline, pipeline.spec.get("model_version", ""), registry_version(uri),
                 file_hashes(mdir), ok, fingerprint(fresh))


def embed(pipeline: Pipeline, photos: list[data.Photo], batch: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(photos), batch):
        images = [ImageOps.exif_transpose(Image.open(p.path)).convert("RGB") for p in photos[i : i + batch]]
        out.append(pipeline.embed(images))
    return np.concatenate(out) if out else np.zeros((0, 0))


def evaluate(ds: data.Dataset, pipeline: Pipeline, zero_shot: bool = False) -> Evaluation:
    protocol = "zero-shot" if zero_shot else "split"
    result: dict = {"dataset": ds.name, "protocol": protocol}
    if zero_shot:
        photos = ds.photos
        E = embed(pipeline, photos)
        animals, species = [p.animal for p in photos], [p.species for p in photos]
        same, diff = metrics.pair_scores(E, animals, species)
        result["val"] = dict(zip(("eer", "threshold"), metrics.eer(same, diff)),
                             pairs_same=len(same), pairs_diff=len(diff))
    else:
        val = ds.split("val")
        if val:
            Ev = embed(pipeline, val)
            s, d = metrics.pair_scores(Ev, [p.animal for p in val], [p.species for p in val])
            result["val"] = dict(zip(("eer", "threshold"), metrics.eer(s, d)), pairs_same=len(s), pairs_diff=len(d))
        photos = ds.split("test")
        if not photos:
            raise ValueError(f"{ds.name} has no test animals")
        E = embed(pipeline, photos)
        animals, species = [p.animal for p in photos], [p.species for p in photos]
        same, diff = metrics.pair_scores(E, animals, species)
    result["test"] = metrics.retrieval(E, animals, species)
    thr = result.get("val", {}).get("threshold", float("nan"))
    result["test_threshold"] = metrics.at_threshold(same, diff, thr)
    for row in result["test"]["per_query"]:
        row["photo"] = os.path.relpath(photos[row["query"]].path, ds.root)
    return Evaluation(ds, protocol, result, photos, E, same, diff)


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


def report(result: dict) -> str:
    lines = [result["dataset"], f"{'group':8} {'queries':>7} {'gallery':>7} {'Rank-1':>7} {'Rank-5':>7} {'Rank-10':>7} {'mAP':>7}"]
    for group, s in result["test"].items():
        if group == "per_query" or not s.get("queries"):
            continue
        lines.append(f"{group:8} {s['queries']:7d} {s['gallery_animals']:7d} {s['rank1']:7.1%} {s['rank5']:7.1%} "
                     f"{s['rank10']:7.1%} {s['mAP']:7.3f}")
    if "val" in result:
        v, t = result["val"], result["test_threshold"]
        where = "all pairs (optimistic)" if result.get("protocol") == "zero-shot" else "val"
        lines.append(f"EER on {where} {v['eer']:.1%} at threshold {v['threshold']:.3f} → FAR {t['far']:.1%}, FRR {t['frr']:.1%}")
    else:
        lines.append("no val animals: threshold not calibrated")
    return "\n".join(lines)


def log_run(ev: Evaluation, model: Model, experiment: str = "", artifacts: str | None = None,
            run_name: str | None = None, params: dict | None = None) -> str:
    """One MLflow run in `pettrace-reid-eval` that traces every number to the exact data (Hub
    commit + fingerprint), model (registry version + sha256 of each ONNX file + golden check) and
    code (GIT_SHA/GIT_DIRTY). `artifacts` = a directory of figures/tables attached under report/."""
    import mlflow
    import onnxruntime
    from mlflow.data.http_dataset_source import HTTPDatasetSource
    from mlflow.data.meta_dataset import MetaDataset

    ds = ev.dataset
    mlflow.set_experiment(EXPERIMENT)
    name = run_name or f"{experiment + ' · ' if experiment else ''}{model.model_version or 'model'} on {ds.name}"
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_input(MetaDataset(HTTPDatasetSource(ds.url), name=ds.name, digest=ds.fingerprint), context="evaluation")
        mlflow.log_params({
            "model.uri": model.uri, "model.model_version": model.model_version,
            "model.registry_version": model.registry_version,
            **{f"model.sha256.{f}": h for f, h in model.sha256.items()},
            "model.golden_fingerprint": model.golden_fingerprint,
            "dataset.repo": ds.repo, "dataset.version": ds.version, "dataset.commit": ds.commit,
            "dataset.fingerprint": ds.fingerprint,
            "protocol": f"{ev.protocol}: same-species gallery, leave-one-out, "
            + ("EER on all pairs (optimistic)" if ev.protocol == "zero-shot" else "EER on val"),
            **(params or {}),
        })
        mlflow.set_tags({
            "experiment": experiment, "dataset": ds.name, "model_version": model.model_version,
            "golden_ok": str(model.golden_ok).lower(),
            "git_sha": os.environ.get("GIT_SHA", "unknown"), "git_dirty": os.environ.get("GIT_DIRTY", "unknown"),
            "onnxruntime": onnxruntime.__version__, "cpu": f"{platform.machine()} {platform.system()}",
        })
        mlflow.log_metrics(flat_metrics(ev.result))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "evaluation.json")
            with open(path, "w") as f:
                json.dump(ev.result, f, indent=2, default=float)
            mlflow.log_artifact(path)
        if artifacts:
            mlflow.log_artifacts(artifacts, artifact_path="report")
        return run.info.run_id
