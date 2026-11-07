"""Run E2: frozen YOLO localization before the frozen E1 CLIP encoder."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from pettrace.experiments.e1 import (
    ENCODER_SHA256,
    MODEL as BASELINE_MODEL,
    PRIVATE_COMMIT,
    PRIVATE_VERSION,
    _dataset_selection,
    _domains,
    require_environment,
)
from pettrace.tracking import source_identity

MODEL = "models:/pettrace-embedder/2"
DETECTOR_SHA256 = "da8f9b9ebea5157cdcefa7bea9fd782249ebdf119f917c07d64e5b41942b9c9d"
MODEL_SHA256 = {"encoder.onnx": ENCODER_SHA256, "detector.onnx": DETECTOR_SHA256}
DETECTOR_SETTINGS = {
    "enabled": True,
    "file": "detector.onnx",
    "input_size": 640,
    "conf": 0.25,
    "classes": [15, 16],
    "margin": 0.10,
    "fallback": "full_image",
}


def audit_detector(dataset, detector):
    """Record the actual crop decision for every image without changing evaluation."""
    from PIL import Image, ImageOps

    class_species = {15: "cat", 16: "dog"}
    findings = []
    for query, photo in enumerate(dataset.photos):
        with Image.open(photo.path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        found = detector.box(image)
        try:
            photo_name = str(Path(photo.path).relative_to(dataset.root))
        except ValueError:
            photo_name = Path(photo.path).name
        finding = {
            "query": query,
            "photo": photo_name,
            "animal": photo.animal,
            "species": photo.species,
            "status": "no_detection",
            "confidence": None,
            "detected_species": None,
            "species_mismatch": False,
            "retained_area": 1.0,
        }
        if found is not None:
            confidence, class_id, (x1, y1, x2, y2) = found
            margin_x = detector.margin * (x2 - x1)
            margin_y = detector.margin * (y2 - y1)
            box = (
                max(0, x1 - margin_x),
                max(0, y1 - margin_y),
                min(image.width, x2 + margin_x),
                min(image.height, y2 + margin_y),
            )
            valid = box[2] - box[0] >= 2 and box[3] - box[1] >= 2
            detected_species = class_species.get(class_id, f"class-{class_id}")
            finding.update(
                status="crop" if valid else "invalid_box",
                confidence=float(confidence),
                detected_species=detected_species,
                species_mismatch=detected_species != photo.species,
            )
            if valid:
                crop = image.crop(box)
                finding["box"] = list(box)
                finding["retained_area"] = crop.width * crop.height / (image.width * image.height)
        findings.append(finding)
    return findings


def _save_detector_audit(dataset, detector, output):
    findings = audit_detector(dataset, detector)
    Path(output, "detector-audit.json").write_text(json.dumps(findings, indent=2) + "\n")
    rows = []
    for species in ("cat", "dog"):
        selected = [row for row in findings if row["species"] == species]
        counts = Counter(row["status"] for row in selected)
        rows.append({
            "species": species,
            "photos": len(selected),
            "crops": counts["crop"],
            "no_detection": counts["no_detection"],
            "invalid_box": counts["invalid_box"],
            "species_mismatch": sum(row["species_mismatch"] for row in selected),
        })
    Path(output, "detector-audit-summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    return findings


def _save_preprocessing_examples(dataset, findings, model, output):
    """Save deterministic public examples of the exact E2 input transform."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image, ImageOps

    examples = []
    for species in ("cat", "dog"):
        for cropped in (True, False):
            candidates = [row for row in findings
                          if row["species"] == species and (row["status"] == "crop") == cropped]
            if not candidates:
                continue
            finding = min(candidates, key=lambda row: row["query"])
            photo = dataset.photos[finding["query"]]
            with Image.open(photo.path) as raw:
                original = ImageOps.exif_transpose(raw).convert("RGB")
            localized = model.pipeline.detector.crop(original)
            tensor = model.pipeline.preprocess(original)
            pre = model.pipeline.pre
            pixels = tensor.transpose(1, 2, 0) * np.asarray(pre["std"]) + np.asarray(pre["mean"])
            pixels = np.clip(pixels / (pre["rescale_factor"] * 255), 0, 1)
            figure, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(original)
            if cropped:
                left, top, right, bottom = finding["box"]
                axes[0].add_patch(Rectangle(
                    (left, top), right - left, bottom - top,
                    fill=False, edgecolor="yellow", linewidth=2))
            axes[0].set_title(f"{species} · {finding['status']}")
            axes[1].imshow(localized)
            axes[1].set_title("YOLO crop / full-image fallback")
            axes[2].imshow(pixels)
            axes[2].set_title("Actual CLIP input")
            for axis in axes:
                axis.axis("off")
            figure.tight_layout()
            filename = f"preprocessing-{species}-{'crop' if cropped else 'fallback'}.png"
            figure.savefig(Path(output, filename), dpi=160, bbox_inches="tight")
            plt.close(figure)
            examples.append({**finding, "figure": filename, "tensor_shape": list(tensor.shape)})
    Path(output, "preprocessing-examples.json").write_text(json.dumps({
        "dataset_commit": dataset.commit,
        "selection": "lowest query index per species and crop/fallback status, when available",
        "examples": examples,
    }, indent=2) + "\n")


def _validate_models(model, baseline):
    if model.sha256 != MODEL_SHA256 or not model.golden_ok:
        raise ValueError("The frozen E2 model failed artifact verification")
    if baseline.sha256 != {"encoder.onnx": ENCODER_SHA256} or not baseline.golden_ok:
        raise ValueError("The frozen E1 control failed artifact verification")
    if model.pipeline.detector is None or baseline.pipeline.detector is not None:
        raise ValueError("E2 must differ from E1 only by the detector")
    steps = {step["op"]: step for step in model.pipeline.spec["steps"]}
    baseline_steps = {step["op"]: step for step in baseline.pipeline.spec["steps"]}
    for operation in ("clip_preprocess", "onnx"):
        if steps[operation] != baseline_steps[operation]:
            raise ValueError(f"E1 and E2 differ in their shared {operation} step")
    for key, expected in DETECTOR_SETTINGS.items():
        if steps["detect_crop"].get(key) != expected:
            raise ValueError(f"Unexpected detector setting: {key}")


def _query_signature(row):
    return tuple(row[key] for key in ("query", "photo", "animal", "species", "gallery_animals"))


def _save_condition_comparison(baseline, localized, output, label, fallback_queries=()):
    """Persist the paired E1→E2 intervention check for one untouched domain."""
    from pettrace.evaluation.figures import comparison_figure
    from pettrace.ui import markdown_table

    baseline_rows = baseline.result["test"]["per_query"]
    localized_rows = localized.result["test"]["per_query"]
    if baseline.dataset.commit != localized.dataset.commit:
        raise ValueError("E1 and E2 datasets differ")
    if [_query_signature(row) for row in baseline_rows] != [
            _query_signature(row) for row in localized_rows]:
        raise ValueError("E1 and E2 query/gallery coverage differs")
    fallback_queries = list(fallback_queries)
    if fallback_queries and not np.allclose(
            baseline.embeddings[fallback_queries], localized.embeddings[fallback_queries],
            atol=1e-5, rtol=0):
        raise ValueError("E2 fallback embeddings differ from E1")

    records = []
    table_rows = []
    for group in ("global", "cat", "dog"):
        for metric in ("rank1", "rank2", "rank5", "mAP"):
            first = baseline.result["test"][group][metric]
            second = localized.result["test"][group][metric]
            records.append({"group": group, "metric": metric, "e1": first,
                            "e2": second, "delta": second - first})
            table_rows.append([group, metric, f"{first:.4f}", f"{second:.4f}", f"{second-first:+.4f}"])
    ranks = [(first["rank"], second["rank"])
             for first, second in zip(baseline_rows, localized_rows)]
    payload = {
        "dataset_commit": localized.dataset.commit,
        "paired_queries": len(ranks),
        "corrected": sum(first > 1 and second == 1 for first, second in ranks),
        "lost": sum(first == 1 and second > 1 for first, second in ranks),
        "unchanged_rank1": sum((first == 1) == (second == 1) for first, second in ranks),
        "metrics": records,
    }
    output = Path(output)
    (output / "e1-e2-comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    table = markdown_table(["Group", "Metric", "E1", "E2", "Δ E2 − E1"], table_rows)
    (output / "e1-e2-comparison.md").write_text(table + "\n")
    figure = comparison_figure({"E1": baseline, "E2": localized}, f"{label} · paired E1/E2")
    figure.savefig(output / "e1-e2-comparison.png", dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return payload


def run(output, *, private_version=PRIVATE_VERSION, private_commit=PRIVATE_COMMIT, cache_dir=None):
    """Evaluate and publish E2 plus its paired E1 control on both pinned domains."""
    from pettrace.evaluation.core import evaluate, load_model, report as text_report
    from pettrace.evaluation.figures import retrieval_examples, save
    from pettrace.evaluation.research import save_benchmark_comparison
    from pettrace.experiments.zero_shot import publish_results
    from pettrace.ui import task

    require_environment()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    public_report = output / "public_test"
    public_report.mkdir()
    cache = Path(cache_dir or Path.home() / ".cache/pettrace/e2").resolve()
    os.environ["PETTRACE_EXECUTION_ID"] = uuid4().hex

    with task("Running E2", 8, unit="step") as progress:
        public, private, receipt = _domains(private_version, private_commit, cache)
        _dataset_selection(public, private, receipt, public_report, output)
        progress.update()
        model, baseline = load_model(MODEL), load_model(BASELINE_MODEL)
        _validate_models(model, baseline)
        progress.update()
        public_findings = _save_detector_audit(public, model.pipeline.detector, public_report)
        private_findings = _save_detector_audit(private, model.pipeline.detector, output)
        _save_preprocessing_examples(public, public_findings, model, public_report)
        progress.update()
        public_ev = evaluate(public, model.pipeline, zero_shot=True)
        private_ev = evaluate(private, model.pipeline, zero_shot=True)
        progress.update()
        public_baseline = evaluate(public, baseline.pipeline, zero_shot=True)
        private_baseline = evaluate(private, baseline.pipeline, zero_shot=True)
        progress.update()
        save(public_ev, model, str(public_report))
        save(private_ev, model, str(output))
        save(public_baseline, baseline, str(public_report / "baseline_e1"))
        save(private_baseline, baseline, str(output / "baseline_e1"))
        figures, _ = retrieval_examples(public_ev, public_report, public.commit)
        import matplotlib.pyplot as plt

        for figure in figures:
            plt.close(figure)
        progress.update()
        _save_condition_comparison(
            public_baseline, public_ev, public_report, "Public v2",
            [row["query"] for row in public_findings if row["status"] != "crop"])
        _save_condition_comparison(
            private_baseline, private_ev, output, f"PetTrace {private.version}",
            [row["query"] for row in private_findings if row["status"] != "crop"])
        table, figure = save_benchmark_comparison(public_ev, private_ev, output, "E2")
        plt.close(figure)
        (output / "benchmark-comparison.md").write_text(table + "\n")
        (output / "results.txt").write_text(
            "PUBLIC\n" + text_report(public_ev.result)
            + "\n\nPRIVATE\n" + text_report(private_ev.result) + "\n")
        progress.update()
        publication = publish_results(
            public_ev, private_ev, model, output, public_report, source_identity()["source_sha256"],
            experiment="E2", expected_model_uri=MODEL, expected_model_sha256=MODEL_SHA256,
            private_dataset_version=private_version,
            extra_design={
                "execution": "headless",
                "private_dataset_commit": private_commit,
                "intervention": "pinned YOLO crop before the unchanged E1 CLIP encoder",
                "paired_e1_control": True,
            },
        )
        progress.update()
    print("Published:", publication["url"], flush=True)
    return publication


def resume_publication(report):
    """Retry both MLflow writes from saved E2 metrics without repeating inference."""
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
        raise ValueError("Retry with the same PetTrace package that computed E2")
    if runtime["model_sha256"] != MODEL_SHA256:
        raise ValueError("Saved evaluation used a different E2 model")
    for evaluation in (public_ev, private_ev):
        evaluation.dataset = Dataset(**evaluation.dataset.__dict__, root="")
    model = load_model(MODEL)
    return publish_results(
        public_ev, private_ev, model, report, public_report, runtime["source_sha256"],
        experiment="E2", expected_model_uri=MODEL, expected_model_sha256=MODEL_SHA256,
        private_dataset_version=private_ev.dataset.version,
        extra_design={
            "execution": "headless",
            "private_dataset_commit": private_ev.dataset.commit,
            "intervention": "pinned YOLO crop before the unchanged E1 CLIP encoder",
            "paired_e1_control": True,
        },
    )


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
    output = args.output or Path("reports") / f"e2-{uuid4().hex[:12]}"
    run(output, private_version=args.private_version, private_commit=args.private_commit,
        cache_dir=args.cache_dir)
    return 0


if __name__ == "__main__":
    main()
