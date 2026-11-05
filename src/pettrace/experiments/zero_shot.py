"""Shared publication mechanics for the E1/E2 zero-shot experiments."""

from io import BytesIO
import json
from pathlib import Path
import shutil

from PIL import Image

from pettrace.artifact_upload import compact_json_uploads
from pettrace.tracking import PRODUCTION_URI, production_context, publish_evaluation, save_provenance


def _prepare_photo_panels(report, public_report):
    """Keep lossless panels locally and publish bounded previews."""
    report, public_report = Path(report), Path(public_report)
    for panel in sorted(public_report.glob("*.png")):
        if not panel.name.startswith(("retrieval-", "preprocessing-")):
            continue
        backup = report.parent / f"{report.name}-original-figures" / panel.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(panel, backup)
        with Image.open(panel) as original:
            preview = original.convert("RGB")
        preview.thumbnail((1800, 1200), Image.Resampling.LANCZOS)
        while True:
            buffer = BytesIO()
            preview.save(buffer, format="JPEG", quality=85, optimize=True)
            if buffer.tell() <= 512 * 1024:
                break
            preview = preview.resize((max(1, preview.width * 3 // 4), max(1, preview.height * 3 // 4)),
                                     Image.Resampling.LANCZOS)
        target = panel.with_suffix(".jpg")
        target.write_bytes(buffer.getvalue())
        for name in ("retrieval-examples.json", "preprocessing-examples.json"):
            manifest = public_report / name
            if not manifest.exists():
                continue
            examples = json.loads(manifest.read_text())
            for example in examples["examples"]:
                if example.get("figure") == panel.name:
                    example["figure"] = target.name
            manifest.write_text(json.dumps(examples, indent=2))
        panel.unlink()


def publish_results(public_evaluation, private_evaluation, model, report, public_report, source_sha256, *,
                    experiment, expected_model_uri, expected_model_sha256, private_dataset_version,
                    extra_design=None):
    """Publish both zero-shot domains and their final receipts; safe to retry."""
    report, public_report = Path(report), Path(public_report)
    if (public_evaluation.protocol != private_evaluation.protocol
            or private_evaluation.protocol != "zero-shot"):
        raise ValueError("E1/E2 publication requires the same zero-shot protocol")
    _prepare_photo_panels(report, public_report)
    design = {
        **(extra_design or {}),
        "schema_version": 2, "experiment": experiment, "protocol": private_evaluation.protocol,
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
        "model_sha256": model.sha256, "training": None,
        "private_identities_used_for_training": False, "eer_scope": "same_evaluation_pairs",
        "mlflow_publication_required": True,
    }
    encoded = json.dumps(design, indent=2)
    (report / "experiment-design.json").write_text(encoded)
    (public_report / "experiment-design.json").write_text(encoded)
    save_provenance(report, model)
    save_provenance(public_report, model)

    from pettrace.experiments.e1 import save_publication_input
    save_publication_input(public_evaluation, public_report)
    save_publication_input(private_evaluation, report)
    receipt = report / "publication.json"
    receipt.write_text(json.dumps({"published": False, "run_id": None}))
    with compact_json_uploads():
        public_run_id = publish_evaluation(
            public_evaluation, model, public_report, source_sha256, experiment=experiment,
            expected_model_uri=expected_model_uri, expected_model_sha256=expected_model_sha256,
            expected_dataset_version=public_evaluation.dataset.version, expected_protocol="zero-shot")
        (public_report / "publication.json").write_text(json.dumps(
            {"published": True, "run_id": public_run_id}))
        run_id = publish_evaluation(
            private_evaluation, model, report, source_sha256, experiment=experiment,
            expected_model_uri=expected_model_uri, expected_model_sha256=expected_model_sha256,
            expected_dataset_version=private_dataset_version, expected_protocol="zero-shot")
        result = {"published": True, "run_id": run_id, "public_run_id": public_run_id}
        receipt.write_text(json.dumps(result))
        client, tracking_experiment = production_context()
        for published_id, folder in ((public_run_id, public_report), (run_id, report)):
            if client.get_run(published_id).info.status != "FINISHED":
                raise RuntimeError(f"Publication is incomplete: {published_id}")
            client.log_artifact(published_id, str(folder / "publication.json"), artifact_path="report")
    return {**result, "url": f"{PRODUCTION_URI}/#/experiments/{tracking_experiment.experiment_id}/runs/{run_id}"}
