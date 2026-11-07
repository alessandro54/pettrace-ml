"""Report retained training volume against the pinned original source inventories."""

import json
from pathlib import Path


CAPACITY = json.loads(Path(__file__).with_name("source_capacity_v1.json").read_text())


def dataset_coverage(rows, capacities=None):
    capacities = CAPACITY if capacities is None else capacities
    if {row["source"] for row in rows} != set(capacities):
        raise ValueError("Coverage rows must contain exactly the pinned dataset sources")
    metrics = {}
    params = {
        "dataset.reference.unit": "photo_entries",
        "dataset.reference.scope": "original_archives_before_filtering",
    }
    groups = [("dataset", rows, sum(source["photos"] for source in capacities.values()))]
    for source, reference in capacities.items():
        source_rows = [row for row in rows if row["source"] == source]
        if any(row["species"] != reference["species"] for row in source_rows):
            raise ValueError("Source species mismatch")
        prefix = f"dataset.{reference['species']}"
        groups.append((prefix, source_rows, reference["photos"]))
        for key in ("ref", "version", "archive_index_sha256"):
            params[f"{prefix}.reference.{key}"] = str(reference[key])
    splits = {"train": {"train"}, "val": {"val"}, "test_reserved": {"test"},
              "train_val": {"train", "val"}, "retained": {"train", "val", "test"}}
    for prefix, subset, maximum in groups:
        if len(subset) > maximum or any(row["split"] not in {"train", "val", "test"} for row in subset):
            raise ValueError("Invalid coverage numerator")
        metrics[f"{prefix}.source_photos"] = maximum
        metrics[f"{prefix}.retained_identities"] = len({row["animal"] for row in subset})
        for label, allowed in splits.items():
            selected = [row for row in subset if row["split"] in allowed]
            metrics[f"{prefix}.{label}_photos"] = len(selected)
            metrics[f"{prefix}.{label}_pct"] = 100.0 * len(selected) / maximum
            metrics[f"{prefix}.{label}_identities"] = len({row["animal"] for row in selected})
    return metrics, params


def training_volume(prepared, percent, report):
    """Record how the complete public v2 corpus relates to its original archives."""
    if percent != 100 or not hasattr(prepared, "provenance"):
        raise ValueError("Public refined training uses the complete 100% source preparation")
    metrics, params = dataset_coverage(prepared.rows)
    for source, reference in CAPACITY.items():
        prefix = f"dataset.{reference['species']}"
        metrics[f"{prefix}.sampled_photos"] = reference["photos"]
        metrics[f"{prefix}.sampled_pct"] = 100.0
        metrics[f"{prefix}.processed_photos"] = reference["photos"]
        metrics[f"{prefix}.processed_pct"] = 100.0
        metrics[f"{prefix}.excluded_photos"] = reference["photos"] - metrics[f"{prefix}.retained_photos"]
    for key in ("sampled_photos", "processed_photos", "excluded_photos"):
        metrics[f"dataset.{key}"] = sum(metrics[f"dataset.{species}.{key}"] for species in ("cat", "dog"))
    metrics.update({"dataset.sample_requested_pct": 100, "dataset.sampled_pct": 100.0,
                    "dataset.processed_pct": 100.0})
    params.update({
        "dataset.sample_basis": "original_source_photo_entries_before_cleaning",
        "dataset.sample_requested_pct": "100",
        "dataset.transport": "huggingface_refined_v2",
        "dataset.prepared_fingerprint": prepared.fingerprint,
    })
    report = Path(report)
    (report / "sources.json").write_text(json.dumps(prepared.provenance, indent=2))
    (report / "dataset-coverage.json").write_text(json.dumps({
        "metrics": metrics, "params": params, "reference": CAPACITY,
        "transport": prepared.provenance,
    }, indent=2))
    return metrics, params
