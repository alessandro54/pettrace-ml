"""Run the thesis E1 notebook in an isolated kernel and retain publication evidence."""

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4
import warnings

from pettrace.tracking import PRODUCTION_URI, source_identity

PYTHON = "3.11.15"
MODEL = "models:/pettrace-embedder/1"
ENCODER_SHA256 = "4773e649274ca8774c4a63d45f0438fe598ee7ba67e65def97537aa0eca9a576"
PRIVATE_REPO = "alessandro54/pettrace-reid"
PRIVATE_COMMIT = "8b12db0dce41040a821f9792bea5b352342263ce"


def check_environment():
    if platform.python_version() != PYTHON or sys.prefix == sys.base_prefix:
        raise RuntimeError(f"E1 requires an isolated Python {PYTHON}; use colab/e1.ipynb or tools/run_e1.py")
    if not source_identity()["build"]:
        raise RuntimeError("E1 requires a published release wheel with build provenance")
    # Exercise the imports that can fail when host-kernel packages leak into the environment.
    import numpy.testing  # noqa: F401
    from PIL import ImageDraw, ImageText  # noqa: F401
    from pettrace.data.versions import load_training  # noqa: F401
    from pettrace.evaluation.figures import save  # noqa: F401
    import mlflow  # noqa: F401


def credentials(env_file=None):
    if env_file:
        from dotenv import load_dotenv
        if not Path(env_file).is_file():
            raise FileNotFoundError(env_file)
        load_dotenv(env_file, override=True)
    os.environ.setdefault("MLFLOW_TRACKING_URI", PRODUCTION_URI)
    for name in ("HF_TOKEN", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        if not os.environ.get(name, "").strip():
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                os.environ[name] = getpass.getpass(f"{name}: ").strip()
        if not os.environ[name].strip():
            raise ValueError(f"Missing {name}")


def preflight():
    import mlflow
    from huggingface_hub import HfApi
    from pettrace.evaluation.research import PUBLIC_REPO, PUBLIC_COMMIT
    from pettrace.tracking import production_context

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client, _ = production_context()
    client.get_model_version("pettrace-embedder", "1")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.whoami()
    for repo, commit in ((PRIVATE_REPO, PRIVATE_COMMIT), (PUBLIC_REPO, PUBLIC_COMMIT)):
        if api.dataset_info(repo, revision=commit).sha != commit:
            raise ValueError(f"Dataset commit mismatch: {repo}")
    if api.dataset_info(PRIVATE_REPO, revision="v4").sha != PRIVATE_COMMIT:
        raise ValueError("The v4 tag differs from the pinned benchmark")
    print("Preflight OK: imports, HF snapshots and MLflow registry access.", flush=True)


def save_publication_input(ev, report):
    """Persist sufficient numeric evidence to retry tracking without inference or image downloads."""
    ds = ev.dataset
    payload = {"dataset": {key: getattr(ds, key) for key in ("repo", "version", "commit", "fingerprint")},
               "protocol": ev.protocol, "result": ev.result}
    path = Path(report) / "evaluation-input.json"
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")


def resume_publication(report):
    from pettrace.data.benchmark import Dataset
    from pettrace.evaluation.core import load_model
    from pettrace.evaluation.research import PUBLIC_REPO, PUBLIC_COMMIT
    from pettrace.tracking import publish_evaluation

    report = Path(report)
    saved = []
    # Validate both domains before making any tracking writes.
    for folder, repo, version, commit in ((report / "public_test", PUBLIC_REPO, "v1", PUBLIC_COMMIT),
                                         (report, PRIVATE_REPO, "v4", PRIVATE_COMMIT)):
        payload = json.loads((folder / "evaluation-input.json").read_text())
        runtime = json.loads((folder / "runtime.json").read_text())
        ds = payload["dataset"]
        if (ds["repo"], ds["version"], ds["commit"], payload["protocol"]) != (repo, version, commit, "zero-shot"):
            raise ValueError("Saved evaluation is not the pinned E1 protocol")
        if runtime["source_sha256"] != source_identity()["source_sha256"]:
            raise ValueError("Retry with the same release that computed the evaluation")
        if runtime["model_sha256"] != {"encoder.onnx": ENCODER_SHA256}:
            raise ValueError("Saved evaluation used a different encoder")
        ev = SimpleNamespace(dataset=Dataset(**ds, root=""), protocol=payload["protocol"], result=payload["result"])
        saved.append((folder, ev, runtime))
    model = load_model(MODEL)
    run_ids = []
    for folder, ev, runtime in saved:
        run_id = publish_evaluation(ev, model, folder, runtime["source_sha256"], experiment="E1",
                                   expected_model_uri=MODEL, expected_model_sha256={"encoder.onnx": ENCODER_SHA256},
                                   expected_dataset_version=ev.dataset.version, expected_protocol="zero-shot")
        (folder / "publication.json").write_text(json.dumps({"published": True, "run_id": run_id}))
        run_ids.append(run_id)
    receipt = {"published": True, "run_id": run_ids[1], "public_run_id": run_ids[0]}
    (report / "publication.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def execute_notebook(work):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_client import KernelManager

    work = Path(work)
    source = work / "e1_clip_zero_shot.ipynb"
    notebook = nbformat.read(source, as_version=4)
    # Historical outputs must never be mistaken for evidence from this new execution.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    notebook.metadata.pop("widgets", None)
    kernels = work / "kernels"
    spec = kernels / "pettrace-e1"
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-I", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "PetTrace E1 (isolated)", "language": "python"}))
    manager = KernelManager(kernel_name="pettrace-e1",
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    client = NotebookClient(notebook, km=manager, timeout=3600, allow_errors=False,
                            resources={"metadata": {"path": str(work)}}, record_timing=True)
    output = work / "e1.executed.ipynb"

    def checkpoint(cell, cell_index, **kwargs):
        nbformat.write(notebook, output)
        if cell.cell_type == "code":
            print(f"E1: cell {cell_index + 1}/{len(notebook.cells)} recorded", flush=True)

    client.on_cell_executed = checkpoint
    try:
        client.execute(cwd=str(work))
    finally:
        nbformat.write(notebook, output)
        print(f"Notebook evidence: {output}", flush=True)
        # An explicitly supplied kernel manager is owned by the caller.
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)


def finish(work):
    from pettrace.tracking import production_context

    reports = list((work / "reports/e1_v4").glob("*/publication.json"))
    if len(reports) != 1:
        raise RuntimeError("Expected exactly one E1 publication receipt in this work directory")
    receipt = json.loads(reports[0].read_text())
    if not receipt.get("published") or not receipt.get("public_run_id"):
        raise RuntimeError("Both evaluation domains must be published before E1 is complete")
    client, experiment = production_context()
    for run_id, folder in ((receipt["public_run_id"], reports[0].parent / "public_test"),
                           (receipt["run_id"], reports[0].parent)):
        if client.get_run(run_id).info.status != "FINISHED":
            raise RuntimeError(f"MLflow run {run_id} is incomplete")
        client.log_artifact(run_id, str(work / "e1.executed.ipynb"), artifact_path="notebook")
        client.log_artifact(run_id, str(folder / "publication.json"), artifact_path="report")
        for name in ("release.json", "requirements-evaluation.txt", "execution.json"):
            client.log_artifact(run_id, str(work / name), artifact_path="execution")
        print(f"Published: {PRODUCTION_URI}/#/experiments/{experiment.experiment_id}/runs/{run_id}", flush=True)
    (work / "publication.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    os.environ["HF_DATASET_REPO"] = PRIVATE_REPO
    check_environment()
    build = json.loads((work / "release.json").read_text())
    if source_identity()["build"] != build:
        raise ValueError("Installed wheel does not match release.json")
    inventory = dict((name, digest) for digest, name in
                     (line.split("  ", 1) for line in (work / "SHA256SUMS").read_text().splitlines()))
    for name in ("e1_clip_zero_shot.ipynb", "requirements-evaluation.txt", "release.json"):
        if hashlib.sha256((work / name).read_bytes()).hexdigest() != inventory[name]:
            raise ValueError(f"Release asset changed: {name}")
    credentials(args.env_file)
    preflight()
    os.environ["PETTRACE_OUTPUT_DIR"] = str(work / "reports")
    os.environ["HF_DATASET_REPO"] = PRIVATE_REPO
    os.environ["GIT_SHA"] = build["git_sha"]
    os.environ["GIT_DIRTY"] = "false"
    if args.resume:
        reports = list((work / "reports/e1_v4").glob("*/evaluation-input.json"))
        if len(reports) != 1:
            raise RuntimeError("No complete saved evaluation to publish; inspect the executed notebook")
        resume_publication(reports[0].parent)
    else:
        os.environ["PETTRACE_EXECUTION_ID"] = uuid4().hex
        (work / "execution.json").write_text(json.dumps({
            "execution_id": os.environ["PETTRACE_EXECUTION_ID"],
            "python": platform.python_version(), "platform": platform.platform(), "executable": sys.executable,
            "notebook_sha256": hashlib.sha256((work / "e1_clip_zero_shot.ipynb").read_bytes()).hexdigest(),
            "requirements_sha256": hashlib.sha256((work / "requirements-evaluation.txt").read_bytes()).hexdigest(),
            "packages": json.loads(subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"], text=True)),
            "build": build}, indent=2) + "\n")
        execute_notebook(work)
    finish(work)


if __name__ == "__main__":
    main()
