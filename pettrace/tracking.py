"""Shared production tracking and provenance for independently executable notebooks."""

import hashlib
import json
from pathlib import Path

PRODUCTION_URI = "https://ml.pettrace.app"
PUBLICATION_TAG = "pettrace.publication_key"


def production_context():
    """Read-only preflight against the existing experiment; never create an experiment."""
    import mlflow
    from pettrace.evaluation import EXPERIMENT

    if mlflow.get_tracking_uri().rstrip("/") != PRODUCTION_URI:
        raise ValueError(f"Evaluation requires the production tracking server: {PRODUCTION_URI}")
    if mlflow.get_registry_uri().rstrip("/") != PRODUCTION_URI:
        raise ValueError("Evaluation requires the model registry on the production tracking server.")
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None or experiment.lifecycle_stage != "active":
        raise ValueError(f"The existing production experiment {EXPERIMENT!r} is unavailable.")
    return client, experiment


def publish_evaluation(ev, model, report, source_sha256, *, experiment,
                       expected_model_uri=None, expected_model_sha256=None,
                       expected_dataset_version=None, expected_protocol=None):
    """Publish only evaluation artifacts; sequential retries reuse or resume the same run.

    Model and dataset choices belong to the calling notebook. Optional expectations enforce them
    again at publication. Concurrent publication of the same result is unsupported because MLflow
    does not enforce uniqueness of run tags.
    """
    import mlflow
    from pettrace import __version__
    from pettrace.evaluation import log_run

    client, tracking_experiment = production_context()
    if mlflow.active_run() is not None:
        raise RuntimeError("Finish the active MLflow run before publishing an evaluation.")
    if not model.golden_ok or (expected_model_uri and model.uri != expected_model_uri):
        raise ValueError("Publication requires the verified expected model.")
    if expected_model_sha256 is not None and model.sha256 != expected_model_sha256:
        raise ValueError("Publication requires the verified expected model hashes.")
    if (not ev.dataset.commit
            or (expected_dataset_version and ev.dataset.version != expected_dataset_version)
            or (expected_protocol and ev.protocol != expected_protocol)):
        raise ValueError("Publication requires the expected pinned dataset and evaluation protocol.")
    report = Path(report)
    if not (report / "summary.json").is_file() or not (report / "runtime.json").is_file():
        raise ValueError("Generate the report and runtime provenance before publishing.")
    identity = {
        "experiment": experiment, "dataset_repo": ev.dataset.repo, "dataset_commit": ev.dataset.commit,
        "dataset_fingerprint": ev.dataset.fingerprint, "model_sha256": model.sha256,
        "source_sha256": source_sha256, "result": ev.result,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=float).encode()).hexdigest()
    matches = client.search_runs(
        [tracking_experiment.experiment_id], filter_string=f"tags.`{PUBLICATION_TAG}` = '{key}'", max_results=2,
    )
    if len(matches) > 1:
        raise RuntimeError("Multiple evaluation runs have this publication key; inspect them before publishing.")
    run = matches[0] if matches else client.create_run(
        tracking_experiment.experiment_id,
        tags={PUBLICATION_TAG: key, "experiment": experiment, "purpose": "evaluation",
              "mlflow.runName": f"{experiment} · {model.model_version} on {ev.dataset.name}"},
    )
    run_id = run.info.run_id
    manifest = {
        "run_id": run_id, "experiment": experiment, "tracking_uri": PRODUCTION_URI,
        "tracking_experiment_id": tracking_experiment.experiment_id, "publication_key": key,
        "dataset": ev.dataset.name, "dataset_commit": ev.dataset.commit,
        "dataset_fingerprint": ev.dataset.fingerprint, "model": model.uri,
        "protocol": ev.protocol, "source_sha256": source_sha256, "package_version": __version__,
    }
    pending = report / "run.json.tmp"
    pending.write_text(json.dumps(manifest, indent=2) + "\n")
    pending.replace(report / "run.json")
    if run.info.status == "FINISHED":
        return run_id
    return log_run(
        ev, model, experiment=experiment, artifacts=str(report),
        params={"code.source_sha256": source_sha256, "code.package_version": __version__},
        tracking_experiment_id=tracking_experiment.experiment_id, run_id=run_id,
    )


def source_identity():
    """Identify installed package code; a repository root is never required by the notebook."""
    import subprocess
    from importlib.metadata import PackageNotFoundError, distribution
    from pettrace import __version__

    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    commit, dirty = "unknown", "unknown"
    if (package_dir.parent / ".git").exists():
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package_dir.parent, text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=package_dir.parent, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            pass
    else:
        try:
            direct_url = distribution("pettrace-ml").read_text("direct_url.json")
            if direct_url:
                commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id", "unknown")
        except PackageNotFoundError:
            pass
    return {"source_sha256": digest.hexdigest(), "package_version": __version__,
            "git_sha": commit, "git_dirty": dirty}


def save_provenance(report, model):
    """Write runtime versions and installed source identity without requiring local project files."""
    import importlib.metadata
    import platform

    packages = ("numpy", "pillow", "pyyaml", "onnxruntime", "mlflow-skinny", "huggingface-hub", "requests", "matplotlib")
    runtime = {
        **source_identity(), "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "onnx_providers": model.pipeline.session.get_providers(), "model_sha256": model.sha256,
    }
    for name in ("torch", "onnx", "pandas"):
        try:
            runtime["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    path = Path(report) / "runtime.json"
    path.write_text(json.dumps(runtime, indent=2) + "\n")
    return runtime
