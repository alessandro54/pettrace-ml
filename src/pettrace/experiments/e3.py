"""Shared loading, publication and registration steps for E3 notebooks."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from pettrace.artifact_upload import compact_json_uploads
from pettrace.tracking import production_context

FROZEN_RUNS = {
    "public": {"E1": "fc9f5ee2f4a04d8dbfbcdd16459a89b3", "E2": "c6f7b7baf0704887818a3d63b0644575"},
    "private": {"E1": "4405cec1e4b64b01991759987cec8a2f", "E2": "e92aeface9c34109883f57952e9844f6"},
}


def select_validation_candidate(candidates):
    """Select the highest validation mAP, breaking exact ties with the lower seed."""
    if not candidates:
        raise ValueError("At least one trained candidate is required")
    seeds = [int(candidate["seed"]) for candidate in candidates]
    scores = [float(candidate["best_val_mAP"]) for candidate in candidates]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Candidate seeds must be unique")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Candidate validation scores must be finite")
    return max(candidates, key=lambda candidate: (
        float(candidate["best_val_mAP"]), -int(candidate["seed"])))


def frozen_controls(evaluation, domain, output):
    """Validate finished runs, immutable data/model and identical queries before comparing."""
    client, _ = production_context()
    evaluations, summaries, receipts = {}, {}, {}
    for label, run_id in FROZEN_RUNS[domain].items():
        run = client.get_run(run_id)
        expected = {"encoder.onnx": "4773e649274ca8774c4a63d45f0438fe598ee7ba67e65def97537aa0eca9a576"}
        if label == "E2":
            expected["detector.onnx"] = "da8f9b9ebea5157cdcefa7bea9fd782249ebdf119f917c07d64e5b41942b9c9d"
        if (run.info.status != "FINISHED" or run.data.tags.get("experiment") != label
                or run.data.params.get("dataset.commit") != evaluation.dataset.commit
                or run.data.params.get("dataset.fingerprint") != evaluation.dataset.fingerprint
                or not run.data.params.get("protocol", "").startswith("zero-shot:")
                or any(run.data.params.get(f"model.sha256.{name}") != digest for name, digest in expected.items())):
            raise ValueError(f"Frozen {label} does not match this benchmark/protocol/model")
        folder = Path(output) / f"baseline_{label.lower()}"
        folder.mkdir(parents=True, exist_ok=True)
        paths = {name: Path(client.download_artifacts(run_id, artifact, str(folder))) for name, artifact in
                 (("evaluation", "evaluation.json"), ("summary", "report/summary.json"))}
        result = json.loads(paths["evaluation"].read_text())
        def signature(rows):
            return [(r["query"], r["photo"], r["animal"], r["species"], r["gallery_animals"]) for r in rows]
        if signature(result["test"]["per_query"]) != signature(evaluation.result["test"]["per_query"]):
            raise ValueError("Frozen baseline queries/gallery differ")
        summaries[label] = json.loads(paths["summary"].read_text())
        evaluations[label] = SimpleNamespace(result=result, dataset=evaluation.dataset)
        receipts[label] = {"run_id": run_id, "sha256": {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in paths.items()}}
    (Path(output) / "frozen-controls.json").write_text(json.dumps(receipts, indent=2))
    return evaluations, summaries


def load_published(runs, model_hashes, model_version, cache_root="reports/published-e3"):
    """Load and verify a published E3 model and both evaluations without creating a run."""
    from pettrace.evaluation.core import file_hashes, load_model, report

    if set(runs) != {"public", "private"} or not model_hashes:
        raise ValueError("Published E3 evidence must include both domains and model hashes")
    client, _ = production_context()
    registered = client.get_model_version("pettrace-embedder", str(model_version))
    if registered.status != "READY" or registered.run_id != runs["private"]["run_id"]:
        raise ValueError("Registered model differs from this notebook's published run")

    results = {}
    cache = Path(cache_root) / runs["private"]["run_id"]
    for domain, expected in runs.items():
        run = client.get_run(expected["run_id"])
        if (run.info.status != "FINISHED" or run.data.tags.get("experiment") != "E3"
                or run.data.params.get("dataset.commit") != expected["commit"]
                or not run.data.params.get("protocol", "").startswith("zero-shot:")
                or any(run.data.params.get(f"model.sha256.{name}") != digest
                       for name, digest in model_hashes.items())):
            raise ValueError("Published run does not match the expected benchmark/model")
        if (domain == "private"
                and registered.source.rstrip("/") != run.info.artifact_uri.rstrip("/") + "/report/model"):
            raise ValueError("Registered version points to a different artifact location")
        folder = cache / domain
        folder.mkdir(parents=True, exist_ok=True)
        receipt = json.loads(Path(client.download_artifacts(
            expected["run_id"], "report/publication.json", str(folder))).read_text())
        if receipt.get("published") is not True or receipt.get("run_id") != expected["run_id"]:
            raise ValueError("Missing successful publication receipt")
        path = folder / "evaluation.json"
        if not path.is_file():
            path = Path(client.download_artifacts(expected["run_id"], "evaluation.json", str(folder)))
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["evaluation_sha256"]:
            raise ValueError("Published evaluation changed or its local cache is corrupt")
        results[domain] = json.loads(path.read_text())
        print(report(results[domain]), "\n")

    model_path = cache / "model"
    required = (*model_hashes, "pipeline.yaml", "golden.npy")
    if not all((model_path / name).is_file() for name in required):
        client.download_artifacts(runs["private"]["run_id"], "report/model", str(cache))
    if file_hashes(str(model_path)) != model_hashes:
        raise ValueError("Published model hashes do not match")
    model = load_model(str(model_path))
    if not model.golden_ok:
        raise ValueError("Published model failed golden verification in this runtime")
    return model, results


def _validate_benchmark_separation(manifest_rows, public_rows, private_hashes):
    public_test = [row for row in public_rows if row["split"] == "test"]
    public_val = [row for row in public_rows if row["split"] == "val"]
    training = [row for row in manifest_rows if row["split"] in {"train", "val"}]
    if ({row["animal"] for row in training} & {row["animal"] for row in public_test}
            or {row["sha256"] for row in training} & {row["sha256"] for row in public_test}):
        raise ValueError("Expanded train/val overlaps the immutable public test")
    def signature(rows):
        return sorted((row["animal"], row["species"], row["sha256"]) for row in rows)
    if signature(row for row in training if row["split"] == "val") != signature(public_val):
        raise ValueError("Training must preserve the published validation photos and identities")
    if ({row["sha256"] for row in manifest_rows} | {row["sha256"] for row in public_rows}) & private_hashes:
        raise ValueError("An exact image is shared with the PetTrace benchmark")
    return public_test


def load_benchmarks(prepared, report, *, private_version, private_commit,
                    private_cache=None):
    """Load the pinned public/private E3 benchmarks and reject identity or image leakage."""
    from pettrace.experiments.e1 import _private_dataset
    from pettrace.data.verification import sha256
    from pettrace.evaluation.research import public_test_dataset, refined_public_receipt
    from pettrace.ui import task

    started = time.monotonic()
    with task("Loading immutable benchmarks", 3, unit="step") as progress:
        private = _private_dataset(
            private_version, private_commit,
            Path(private_cache or Path.home() / ".cache/pettrace/e3/private"))
        progress.update()
        private_hashes = {sha256(photo.path) for photo in private.photos}
        progress.update()
        public_source = refined_public_receipt(prepared)
        public_test_rows = _validate_benchmark_separation(
            prepared.rows, prepared.rows, private_hashes)
        progress.update()
    public = public_test_dataset(prepared, public_source)
    public_report = Path(report) / "public_test"
    public_report.mkdir(parents=True, exist_ok=True)
    selection = {
        "scope": "published_test_only", "source": public_source,
        "prepared_fingerprint": prepared.fingerprint,
        "evaluation_fingerprint": public.fingerprint,
        "photos": len(public.photos), "identities": len(public.animals),
        "rows": [{key: row[key] for key in ("animal", "species", "sha256")}
                 for row in public_test_rows],
    }
    (public_report / "dataset-selection.json").write_text(json.dumps(selection, indent=2))
    print(f"Benchmarks ready in {time.monotonic() - started:.1f}s: "
          f"{len(public.photos)} public · {len(private.photos)} PetTrace.", flush=True)
    return SimpleNamespace(private=private, public=public, public_report=public_report)


def publish_results(public_evaluation, private_evaluation, model, report, public_report, source_sha256, *,
                    experiment, source_sample_pct, config, training_summary, prepared_fingerprint,
                    training_source, coverage_metrics, coverage_params, private_dataset_version):
    """Publish both E3 domains with the same training evidence; safe to retry after interruption."""
    import mlflow
    from mlflow.entities import Metric, Param
    from pettrace.tracking import publish_evaluation, save_provenance

    report, public_report = Path(report), Path(public_report)
    design = {
        "schema_version": 2, "experiment": experiment, "source_sample_pct": source_sample_pct,
        "dataset_coverage": coverage_metrics, "protocol": private_evaluation.protocol,
        "phases": ["data_preparation", "representation_and_training", "evaluation"],
        "public_benchmark": {
            "repo": public_evaluation.dataset.repo, "version": public_evaluation.dataset.version,
            "commit": public_evaluation.dataset.commit, "fingerprint": public_evaluation.dataset.fingerprint,
            "split": "test",
        },
        "task": "individual_pet_retrieval", "gallery": "closed_set_same_species",
        "query_unit": "photograph", "rank_aggregation": "max_score_per_identity",
        "map_aggregation": "photograph_ranking", "capture_session_disjoint": False,
        "private_benchmark": {
            "repo": private_evaluation.dataset.repo, "version": private_evaluation.dataset.version,
            "commit": private_evaluation.dataset.commit, "fingerprint": private_evaluation.dataset.fingerprint,
        },
        "model_sha256": model.sha256, "training": asdict(config),
        "private_identities_used_for_training": False, "eer_scope": "same_evaluation_pairs",
        "mlflow_publication_required": True,
    }
    encoded_design = json.dumps(design, indent=2)
    (report / "experiment-design.json").write_text(encoded_design)
    (public_report / "experiment-design.json").write_text(encoded_design)
    for filename in ("dataset-coverage.json", "training-config.json", "training-summary.json", "sources.json"):
        shutil.copy2(report / filename, public_report / filename)
    if (report / "transport-equivalence.json").is_file():
        shutil.copy2(report / "transport-equivalence.json", public_report / "transport-equivalence.json")
    save_provenance(report, model)

    timestamp = int(time.time() * 1000)
    metric_keys = ("train.loss", "train.active_triplets", "val.macro_species_mAP",
                   "training.optimizer_steps", "training.examples_seen", "training.unique_photos_seen")
    metrics = [Metric(key, float(record[key]), timestamp, int(record["epoch"]))
               for record in training_summary["history"] for key in metric_keys]
    metrics.extend([
        Metric("training.best_epoch", float(training_summary["best_epoch"]), timestamp, 0),
        Metric("training.best_val_mAP", float(training_summary["best_val_mAP"]), timestamp, 0),
    ])
    params = {f"training.{key}": str(value) for key, value in asdict(config).items()}
    params.update({"training.dataset_fingerprint": prepared_fingerprint,
                   "training.dataset_source": json.dumps(training_source, sort_keys=True)})

    with compact_json_uploads():
        receipt = report / "publication.json"
        receipt.write_text(json.dumps({"published": False, "run_id": None}))
        public_run_id = publish_evaluation(
            public_evaluation, model, public_report, source_sha256, experiment=experiment,
            expected_dataset_version=public_evaluation.dataset.version, expected_protocol="zero-shot")
        (public_report / "publication.json").write_text(json.dumps(
            {"published": True, "run_id": public_run_id}))
        run_id = publish_evaluation(
            private_evaluation, model, report, source_sha256, experiment=experiment,
            expected_dataset_version=private_dataset_version, expected_protocol="zero-shot")
        client = mlflow.MlflowClient()
        for published_id in (public_run_id, run_id):
            client.log_batch(
                published_id,
                metrics=metrics + [Metric(key, float(value), timestamp, 0)
                                   for key, value in coverage_metrics.items()],
                params=[Param(key, str(value)) for key, value in {**coverage_params, **params}.items()],
            )
            client.set_tag(published_id, "source_sample_pct", str(source_sample_pct))
        receipt.write_text(json.dumps(
            {"published": True, "run_id": run_id, "public_run_id": public_run_id}))
        for published_id, folder in ((public_run_id, public_report), (run_id, report)):
            if client.get_run(published_id).info.status != "FINISHED":
                raise RuntimeError(f"Publication is incomplete: {published_id}")
            client.log_artifact(published_id, str(folder / "publication.json"), artifact_path="report")
    return {"run_id": run_id, "public_run_id": public_run_id}


def register_candidate(report, run_id, public_run_id, *, experiment="E3"):
    """Register one idempotent candidate after both published runs finish; never promote an alias."""
    import mlflow

    report = Path(report)
    client = mlflow.MlflowClient()
    publication = json.loads((report / "publication.json").read_text())
    if (not publication.get("published") or publication.get("run_id") != run_id
            or publication.get("public_run_id") != public_run_id):
        raise RuntimeError("Publish both domains before registering the model")
    if any(client.get_run(current).info.status != "FINISHED" for current in (public_run_id, run_id)):
        raise RuntimeError("Publication is incomplete; retry publication")
    model_name = "pettrace-embedder"
    model_source = client.get_run(run_id).info.artifact_uri.rstrip("/") + "/report/model"
    versions = [version for version in client.search_model_versions(f"name='{model_name}'")
                if version.run_id == run_id and version.source == model_source]
    if len(versions) > 1:
        raise RuntimeError("Multiple model versions for this run; inspect before registering")
    version = versions[0] if versions else client.create_model_version(
        model_name, source=model_source, run_id=run_id,
        tags={"experiment": experiment, "status": "candidate"})
    version = client.get_model_version(model_name, version.version)
    if version.status != "READY":
        raise RuntimeError(f"Model registration is not ready: {version.status}")
    registration = {"registered": True, "name": model_name, "version": str(version.version),
                    "run_id": run_id, "source": model_source, "alias_promoted": False}
    path = report / "registration.json"
    path.write_text(json.dumps(registration, indent=2))
    client.log_artifact(run_id, str(path), artifact_path="report")
    return registration
