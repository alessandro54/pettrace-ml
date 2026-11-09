"""Frozen E4 control, validation calibration and publication for E5."""

from dataclasses import replace
import json
from pathlib import Path

from pettrace.artifact_upload import compact_json_uploads
from pettrace.evaluation.reranking import calibrate, rerank_evaluation
from pettrace.experiments.zero_shot import save_publication_input
from pettrace.tracking import publish_evaluation, save_provenance, tracking_context, tracking_run_url

from pettrace.evaluation.core import load_model

CONTROL_RUNS = {
    "public": "358cb901f3e34483b24acc6ce93649bb",
    "private": "fbf9ca83a0df4c48b1063ea9c609295d",
}
CONTROL_MODEL_VERSION = "23"
CONTROL_MODEL_LABEL = "clip-vit-b32-crop-lora-e3-v4.100@v3"
CONTROL_MODEL_SHA256 = {
    "detector.onnx": "da8f9b9ebea5157cdcefa7bea9fd782249ebdf119f917c07d64e5b41942b9c9d",
    "encoder.onnx": "aaa922be105009c539985d241036999f55f18bef3806f2b02c6894cddc884ebe",
}
CONTROL_SEED = "46"
CONTROL_EPOCH = 27
CONTROL_VALIDATION_MAP = 0.8531207848327025
CONTROL_RESULTS = {
    "public": {"version": "v2", "queries": 277, "identities": 34,
               "rank1": 0.8556, "rank2": 0.9603, "mAP": 0.8345},
    "private": {"version": "v6", "queries": 502, "identities": 33,
                "rank1": 0.8625, "rank2": 0.9422, "mAP": 0.7333,
                "max_impostor_similarity": 0.980469},
}


def load_frozen_e4(model_uri, output):
    """Load the exact E4 winner from a local or configured URI and verify its immutable identity."""
    model = load_model(str(model_uri))
    if (not model.golden_ok or model.model_version != CONTROL_MODEL_LABEL
            or model.sha256 != CONTROL_MODEL_SHA256):
        raise ValueError("E4 control failed identity or golden verification")

    receipt = {
        "experiment": "E4",
        "status": "frozen",
        "runs": CONTROL_RUNS,
        "model_name": "pettrace-embedder",
        "model_version": CONTROL_MODEL_VERSION,
        "model_uri": str(model_uri),
        "model_label": CONTROL_MODEL_LABEL,
        "model_sha256": CONTROL_MODEL_SHA256,
        "seed": int(CONTROL_SEED),
        "selected_epoch": CONTROL_EPOCH,
        "validation_macro_species_mAP": CONTROL_VALIDATION_MAP,
        "inference": {"representation": "single", "identity_aggregation": "max"},
        "test_results": CONTROL_RESULTS,
        "rejected_validation_diagnostics": ["horizontal_flip_tta", "top2_mean", "prototype"],
        "alias_promoted": False,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "e4-control.json").write_text(json.dumps(receipt, indent=2))
    return model, receipt


def select_calibration(validation, attributes, grid, output, *, attribute_cache_sha256):
    """Freeze E5 thresholds using validation only and record every tried condition."""
    selected, candidates = calibrate(validation, attributes, grid)
    evidence = {
        "selection_split": "validation",
        "objective": "maximize Rank-1, then mAP; prefer zero damaged correct matches",
        "attribute_cache_sha256": attribute_cache_sha256,
        "selected": selected,
        "candidates": candidates,
        "test_sets_read": False,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return selected


def apply(public_visual, private_visual, public_attributes, private_attributes, calibration):
    """Evaluate frozen baseline, ungated E5 and frozen double-gated E5 on separate domains."""
    kwargs = {"text_weight": calibration["lambda"], "top_k": 5}
    public = {
        "baseline": public_visual.result["test"],
        "ungated": rerank_evaluation(public_visual, public_attributes, gated=False, **kwargs),
        "gated": rerank_evaluation(
            public_visual, public_attributes, gated=True,
            visual_margin_threshold=calibration["tau_v"],
            text_advantage_threshold=calibration["tau_t"], **kwargs),
    }
    private = {
        "baseline": private_visual.result["test"],
        "ungated": rerank_evaluation(private_visual, private_attributes, gated=False, **kwargs),
        "gated": rerank_evaluation(
            private_visual, private_attributes, gated=True,
            visual_margin_threshold=calibration["tau_v"],
            text_advantage_threshold=calibration["tau_t"], **kwargs),
    }
    return public, private


def as_evaluation(visual, reranked):
    """Keep E4 embeddings/pair evidence while replacing only retrieval rankings."""
    retrieval = {key: value for key, value in reranked.items() if key != "audit"}
    result = {**visual.result, "test": retrieval, "reranking_audit": reranked["audit"]}
    return replace(visual, result=result)


def save_comparison(public, private, output):
    """Persist the required E4/ungated/gated table without mixing benchmark galleries."""
    labels = {"baseline": "E4 Baseline", "ungated": "Ungated E5", "gated": "Double-Gated E5"}
    records = []
    for domain, conditions in (("Public v2", public), ("PetTrace v6", private)):
        for condition, result in conditions.items():
            summary = result["global"]
            audit = result.get("audit", {})
            records.append({
                "dataset": domain, "condition": labels[condition],
                **{key: summary[key] for key in ("queries", "rank1", "rank2", "rank5", "mAP")},
                "gate_activation_rate": audit.get("gate_activation_rate", 0.0),
                "rescued_queries": audit.get("rescued_queries", 0),
                "damaged_queries": audit.get("damaged_queries", 0),
            })
    lines = [
        "| Dataset | Condition | Queries | Rank-1 | Rank-2 | Rank-5 | mAP | Gate | Rescued | Damaged |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['queries']} | {row['rank1']:.2%} | "
            f"{row['rank2']:.2%} | {row['rank5']:.2%} | {row['mAP']:.4f} | "
            f"{row['gate_activation_rate']:.2%} | {row['rescued_queries']} | {row['damaged_queries']} |")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    markdown = "\n".join(lines)
    (output / "comparison.md").write_text(markdown + "\n")
    return markdown


def publish_results(public_evaluation, private_evaluation, model, report, public_report,
                    source_sha256, *, calibration, attribute_receipt):
    """Publish both gated E5 domains; retries reuse completed MLflow runs."""
    report, public_report = Path(report), Path(public_report)
    design = {
        "schema_version": 1,
        "experiment": "E5",
        "intervention": "double-gated identity-level natural-language reranking of visual Top-5",
        "visual_control": {"runs": CONTROL_RUNS, "model": CONTROL_MODEL_LABEL,
                           "inference": "single crop + max identity aggregation"},
        "attribute_model": attribute_receipt,
        "calibration": calibration,
        "selection_split": "public validation only",
        "test_evaluation": "single frozen pass; Public v2 and PetTrace v6 remain separate",
        "text_identity_aggregation": "maximum compatibility across candidate gallery photos",
        "map_aggregation": "photo ranking; identity semantic bonus is shared by Top-5 gallery photos",
        "mlflow_publication_required": True,
    }
    encoded = json.dumps(design, indent=2)
    for folder in (report, public_report):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "experiment-design.json").write_text(encoded + "\n")
        save_provenance(folder, model)
    save_publication_input(public_evaluation, public_report)
    save_publication_input(private_evaluation, report)
    receipt = report / "publication.json"
    receipt.write_text(json.dumps({"published": False, "run_id": None}) + "\n")
    with compact_json_uploads():
        public_run_id = publish_evaluation(
            public_evaluation, model, public_report, source_sha256, experiment="E5",
            expected_model_sha256=CONTROL_MODEL_SHA256,
            expected_dataset_version="v2", expected_protocol=public_evaluation.protocol)
        (public_report / "publication.json").write_text(json.dumps(
            {"published": True, "run_id": public_run_id}) + "\n")
        run_id = publish_evaluation(
            private_evaluation, model, report, source_sha256, experiment="E5",
            expected_model_sha256=CONTROL_MODEL_SHA256,
            expected_dataset_version="v6", expected_protocol=private_evaluation.protocol)
        result = {"published": True, "run_id": run_id, "public_run_id": public_run_id}
        receipt.write_text(json.dumps(result) + "\n")
        client, experiment = tracking_context()
        for published_id, folder in ((public_run_id, public_report), (run_id, report)):
            if client.get_run(published_id).info.status != "FINISHED":
                raise RuntimeError(f"Publication is incomplete: {published_id}")
            client.log_artifact(published_id, str(folder / "publication.json"), artifact_path="report")
    return {**result, "url": tracking_run_url(experiment.experiment_id, run_id)}
