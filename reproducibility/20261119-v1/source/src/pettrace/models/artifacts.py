"""Verified local, Hugging Face and MLflow model artifacts."""

import glob
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, snapshot_download

from pettrace.models.golden import compare, fingerprint, golden_images
from pettrace.models.pipeline import Pipeline


@dataclass
class Model:
    uri: str
    dir: str
    pipeline: Pipeline
    model_version: str
    registry_version: str
    sha256: dict[str, str]
    golden_ok: bool
    golden_fingerprint: str


def file_hashes(model_dir: str | os.PathLike) -> dict[str, str]:
    """Return SHA-256 identities for every ONNX file in an artifact directory."""
    hashes = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "**", "*.onnx"), recursive=True)):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        hashes[os.path.basename(path)] = digest.hexdigest()
    return hashes


def registry_version(uri: str) -> str:
    """Resolve an MLflow model URI to its registry version number."""
    if not uri.startswith("models:/"):
        return ""
    try:
        from mlflow import MlflowClient
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install pettrace-ml[tracking] to load an MLflow model URI") from exc

    name, _, alias = uri[len("models:/") :].partition("@")
    if alias:
        return MlflowClient().get_model_version_by_alias(name, alias).version
    return name.rsplit("/", 1)[1] if "/" in name else ""


def model_dir(uri: str | os.PathLike) -> str:
    """Return a local artifact directory, downloading an MLflow URI when needed."""
    uri = str(uri)
    if os.path.isdir(uri):
        return uri
    if not (uri.startswith(("models:/", "runs:/")) or "://" in uri):
        raise FileNotFoundError(f"Model artifact directory does not exist: {uri}")
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install pettrace-ml[tracking] to load an MLflow model URI") from exc
    return mlflow.artifacts.download_artifacts(uri)


def download_model(repo_id: str, *, revision: str, subfolder: str | None = None,
                   cache_dir: str | os.PathLike | None = None, token: str | None = None) -> Path:
    """Download a Hugging Face model at the immutable commit resolved by ``revision``."""
    repo_id, revision = repo_id.strip(), revision.strip()
    if not repo_id or not revision:
        raise ValueError("repo_id and revision are required")
    relative = Path(subfolder or ".")
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe model subfolder: {relative}")

    commit = HfApi(token=token).model_info(repo_id, revision=revision).sha
    if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
        raise ValueError("Hugging Face did not resolve the model to an immutable commit")
    kwargs = {"repo_type": "model", "revision": commit, "token": token}
    if subfolder:
        kwargs["allow_patterns"] = [f"{relative.as_posix()}/**"]
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    root = Path(snapshot_download(repo_id, **kwargs)).resolve()
    artifact = (root / relative).resolve()
    if root not in artifact.parents and artifact != root:
        raise ValueError(f"Unsafe model subfolder: {relative}")
    required = [artifact / "pipeline.yaml", artifact / "golden.npy"]
    if any(not path.is_file() for path in required) or not list(artifact.rglob("*.onnx")):
        raise FileNotFoundError("The model release is missing pipeline.yaml, golden.npy or ONNX weights")
    return artifact


def load_model(uri: str | os.PathLike) -> Model:
    """Load an artifact and verify its stored golden embeddings."""
    uri = str(uri)
    artifact = model_dir(uri)
    pipeline = Pipeline(artifact)
    stored = np.load(os.path.join(artifact, "golden.npy"))
    fresh = pipeline.embed(golden_images())
    ok, _, _ = compare(stored, fresh)
    return Model(uri, artifact, pipeline, pipeline.spec.get("model_version", ""), registry_version(uri),
                 file_hashes(artifact), ok, fingerprint(fresh))
