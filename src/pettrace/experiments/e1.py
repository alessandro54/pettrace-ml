"""Run the frozen E1 baseline without a notebook, UI, Drive or repository checkout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from pettrace.tracking import PRODUCTION_URI, source_identity

MODEL = "models:/pettrace-embedder/1"
ENCODER_SHA256 = "4773e649274ca8774c4a63d45f0438fe598ee7ba67e65def97537aa0eca9a576"
PRIVATE_REPO = "alessandro54/pettrace-reid"
PRIVATE_VERSION = "v5"
PRIVATE_COMMIT = "1777734dbffda0df0bf43305233809ee73157d94"


def require_environment():
    """Validate non-interactive credentials before downloading or evaluating anything."""
    os.environ.setdefault("MLFLOW_TRACKING_URI", PRODUCTION_URI)
    if os.environ["MLFLOW_TRACKING_URI"].rstrip("/") != PRODUCTION_URI:
        raise ValueError(f"Official E1 runs must publish to {PRODUCTION_URI}")
    missing = [name for name in ("HF_TOKEN", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
               if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    os.environ.setdefault("MPLBACKEND", "Agg")


def _private_dataset(version, commit, cache_dir):
    from huggingface_hub import HfApi, snapshot_download
    from pettrace.data.benchmark import from_dir

    token = os.environ["HF_TOKEN"]
    resolved = HfApi(token=token).dataset_info(PRIVATE_REPO, revision=version).sha
    if resolved != commit:
        raise ValueError(f"{PRIVATE_REPO}@{version} does not resolve to the requested commit")
    root = snapshot_download(PRIVATE_REPO, repo_type="dataset", revision=commit,
                             cache_dir=str(cache_dir), token=token)
    return from_dir(root, repo=PRIVATE_REPO, version=version, commit=commit)


def _domains(private_version, private_commit, cache_dir):
    from pettrace.data.snapshots import load_training
    from pettrace.data.verification import sha256
    from pettrace.evaluation.research import (
        PUBLIC_COMMIT, PUBLIC_FINGERPRINT, PUBLIC_REPO, PUBLIC_VERSION, public_test_dataset,
    )

    prepared, receipt = load_training(
        repo_id=PUBLIC_REPO, version=PUBLIC_VERSION, revision=PUBLIC_COMMIT,
        expected_fingerprint=PUBLIC_FINGERPRINT, cache_dir=cache_dir / "public",
    )
    public = public_test_dataset(prepared, receipt)
    private = _private_dataset(private_version, private_commit, cache_dir / "private")
    public_hashes = {row["sha256"] for row in prepared.rows}
    if public_hashes.intersection(sha256(photo.path) for photo in private.photos):
        raise ValueError("An exact image is shared by the public and private benchmarks")
    return public, private, receipt


def _dataset_selection(public, private, receipt, public_report, report):
    for dataset, folder, source in ((public, public_report, receipt), (private, report, None)):
        payload = {
            "scope": "published_test_only" if dataset is public else "complete_private_benchmark",
            "dataset": {key: getattr(dataset, key) for key in ("repo", "version", "commit", "fingerprint")},
            "photos": len(dataset.photos), "identities": len(dataset.animals),
        }
        if source:
            payload["source"] = source
        (folder / "dataset-selection.json").write_text(json.dumps(payload, indent=2) + "\n")


def run(output, *, private_version=PRIVATE_VERSION, private_commit=PRIVATE_COMMIT, cache_dir=None):
    """Evaluate and publish both E1 domains; local evidence survives publication failures."""
    from pettrace.evaluation.core import evaluate, load_model, report as text_report
    from pettrace.evaluation.figures import save
    from pettrace.evaluation.research import save_benchmark_comparison
    from pettrace.experiments.zero_shot import publish_results
    from pettrace.ui import task

    require_environment()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    public_report = output / "public_test"
    public_report.mkdir()
    cache = Path(cache_dir or Path.home() / ".cache/pettrace/e1").resolve()
    os.environ["PETTRACE_EXECUTION_ID"] = uuid4().hex

    with task("Running E1", 6, unit="step") as progress:
        public, private, receipt = _domains(private_version, private_commit, cache)
        _dataset_selection(public, private, receipt, public_report, output)
        progress.update()
        model = load_model(MODEL)
        if model.sha256 != {"encoder.onnx": ENCODER_SHA256} or not model.golden_ok:
            raise ValueError("The frozen E1 encoder failed artifact verification")
        if model.pipeline.detector is not None:
            raise ValueError("E1 requires full-image CLIP inference without a detector")
        progress.update()
        public_ev = evaluate(public, model.pipeline, zero_shot=True)
        progress.update()
        private_ev = evaluate(private, model.pipeline, zero_shot=True)
        progress.update()
        save(public_ev, model, str(public_report))
        save(private_ev, model, str(output))
        table, figure = save_benchmark_comparison(public_ev, private_ev, output, "E1")
        import matplotlib.pyplot as plt

        plt.close(figure)
        (output / "benchmark-comparison.md").write_text(table + "\n")
        (output / "results.txt").write_text(
            "PUBLIC\n" + text_report(public_ev.result)
            + "\n\nPRIVATE\n" + text_report(private_ev.result) + "\n")
        progress.update()
        publication = publish_results(
            public_ev, private_ev, model, output, public_report, source_identity()["source_sha256"],
            experiment="E1", expected_model_uri=MODEL,
            expected_model_sha256={"encoder.onnx": ENCODER_SHA256},
            private_dataset_version=private_version,
            extra_design={"execution": "headless", "private_dataset_commit": private_commit},
        )
        progress.update()
    print(table, flush=True)
    print("Published:", publication["url"], flush=True)
    return publication


def resume_publication(report):
    """Retry both MLflow writes from saved metrics without loading images or recomputing embeddings."""
    from pettrace.data.benchmark import Dataset
    from pettrace.evaluation.core import load_model
    from pettrace.experiments.zero_shot import load_publication_input, publish_results

    require_environment()
    report = Path(report).resolve()
    public_report = report / "public_test"
    public_ev = load_publication_input(public_report)
    private_ev = load_publication_input(report)
    runtime = json.loads((report / "runtime.json").read_text())
    if runtime["source_sha256"] != source_identity()["source_sha256"]:
        raise ValueError("Retry with the same PetTrace package that computed the evaluation")
    if runtime["model_sha256"] != {"encoder.onnx": ENCODER_SHA256}:
        raise ValueError("Saved evaluation used a different encoder")
    for evaluation in (public_ev, private_ev):
        evaluation.dataset = Dataset(**evaluation.dataset.__dict__, root="")
    model = load_model(MODEL)
    return publish_results(
        public_ev, private_ev, model, report, public_report, runtime["source_sha256"],
        experiment="E1", expected_model_uri=MODEL,
        expected_model_sha256={"encoder.onnx": ENCODER_SHA256},
        private_dataset_version=private_ev.dataset.version,
        extra_design={"execution": "headless", "private_dataset_commit": private_ev.dataset.commit},
    )


def notebook_artifacts(work, experiment="E1"):
    """Archive old executed notebooks without making notebook execution part of E1."""
    work = Path(work)
    filename = f"{experiment.lower()}.executed.ipynb"
    contents = (work / filename).read_bytes()
    folder = work / "notebook-artifact"
    folder.mkdir(exist_ok=True)
    parts = []
    for index, offset in enumerate(range(0, len(contents), 512 * 1024)):
        name = f"{filename}.part{index:04d}"
        chunk = contents[offset:offset + 512 * 1024]
        (folder / name).write_bytes(chunk)
        parts.append({"file": name, "sha256": hashlib.sha256(chunk).hexdigest()})
    manifest = {"file": filename, "sha256": hashlib.sha256(contents).hexdigest(), "parts": parts}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (folder / "restore_notebook.py").write_text(
        "from pathlib import Path\nimport hashlib, json\n"
        "root=Path(__file__).resolve().parent\nm=json.loads((root/'manifest.json').read_text())\n"
        "chunks=[]\n"
        "for p in m['parts']:\n"
        " d=(root/p['file']).read_bytes()\n"
        " assert hashlib.sha256(d).hexdigest()==p['sha256'], 'Notebook part checksum mismatch'\n"
        " chunks.append(d)\n"
        "data=b''.join(chunks)\nassert hashlib.sha256(data).hexdigest()==m['sha256'], 'Notebook checksum mismatch'\n"
        "(root/m['file']).write_bytes(data)\nprint(root/m['file'])\n")
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--private-version", default=PRIVATE_VERSION)
    parser.add_argument("--private-commit", default=PRIVATE_COMMIT)
    parser.add_argument("--resume", type=Path, help="Retry publication from an existing report")
    args = parser.parse_args(argv)
    if args.resume:
        resume_publication(args.resume)
        return 0
    output = args.output or Path("reports") / f"e1-{uuid4().hex[:12]}"
    run(output, private_version=args.private_version, private_commit=args.private_commit,
        cache_dir=args.cache_dir)
    return 0


if __name__ == "__main__":
    main()
