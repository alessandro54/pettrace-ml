"""Headless E2 evaluates paired E1 controls and can retry publication alone."""

import json
from types import SimpleNamespace

from pettrace.experiments import e2


def dataset(version, commit):
    photo = SimpleNamespace(path="photo.jpg", animal="animal", species="cat", split="test")
    return SimpleNamespace(repo="owner/data", version=version, commit=commit,
                           fingerprint=f"fingerprint-{version}", photos=[photo],
                           animals={"animal": {}}, name=f"data@{version}")


def test_run_evaluates_both_domains_and_mandatorily_publishes(tmp_path, monkeypatch):
    public, private = dataset("v2", "a" * 40), dataset("v5", "b" * 40)
    detector = SimpleNamespace(margin=.1, box=lambda image: None, crop=lambda image: image)
    pipeline = SimpleNamespace(detector=detector, pre={}, spec={"steps": []})
    baseline_pipeline = SimpleNamespace(detector=None, spec={"steps": []})
    model = SimpleNamespace(uri=e2.MODEL, sha256=e2.MODEL_SHA256, golden_ok=True, pipeline=pipeline)
    baseline = SimpleNamespace(uri=e2.BASELINE_MODEL, sha256={"encoder.onnx": e2.ENCODER_SHA256},
                               golden_ok=True, pipeline=baseline_pipeline)
    evaluations, published = [], []

    monkeypatch.setattr(e2, "require_environment", lambda: None)
    monkeypatch.setattr(e2, "_domains", lambda version, commit, cache: (
        public, private, {"repo": "composite/public", "version": "v2", "commit": "a" * 40,
                          "fingerprint": "public-fingerprint", "sources": []}))
    monkeypatch.setattr(e2, "source_identity", lambda: {"source_sha256": "source"})
    monkeypatch.setattr(e2, "_validate_models", lambda *args: None)
    monkeypatch.setattr(e2, "_save_detector_audit",
                        lambda ds, detector, output: [{"query": 0, "status": "no_detection"}])
    monkeypatch.setattr(e2, "_save_preprocessing_examples", lambda *args: None)
    monkeypatch.setattr(e2, "_save_condition_comparison", lambda *args, **kwargs: {})
    monkeypatch.setattr("pettrace.evaluation.core.load_model",
                        lambda uri: model if uri == e2.MODEL else baseline)

    def evaluate(ds, current_pipeline, zero_shot):
        evaluations.append((ds.version, current_pipeline is pipeline))
        return SimpleNamespace(dataset=ds, protocol="zero-shot", result={"test": {}})

    monkeypatch.setattr("pettrace.evaluation.core.evaluate", evaluate)
    monkeypatch.setattr("pettrace.evaluation.core.report", lambda result: "metrics")
    monkeypatch.setattr("pettrace.evaluation.figures.save", lambda *args: {})
    monkeypatch.setattr("pettrace.evaluation.figures.retrieval_examples", lambda *args: ([], []))
    monkeypatch.setattr("pettrace.evaluation.research.save_benchmark_comparison",
                        lambda *args: ("| comparison |", None))

    def publish(*args, **kwargs):
        published.append((args, kwargs))
        return {"published": True, "run_id": "private", "public_run_id": "public", "url": "run-url"}

    monkeypatch.setattr("pettrace.experiments.zero_shot.publish_results", publish)
    report = tmp_path / "e2"
    result = e2.run(report, private_version="v5", private_commit="b" * 40,
                    cache_dir=tmp_path / "cache")

    assert evaluations == [("v2", True), ("v5", True), ("v2", False), ("v5", False)]
    assert len(published) == 1
    assert published[0][1]["private_dataset_version"] == "v5"
    assert published[0][1]["extra_design"]["paired_e1_control"] is True
    assert (report / "benchmark-comparison.md").read_text() == "| comparison |\n"
    assert result["published"] is True


def test_resume_republishes_saved_metrics_without_evaluation(tmp_path, monkeypatch):
    report, public = tmp_path / "e2", tmp_path / "e2" / "public_test"
    public.mkdir(parents=True)
    for folder, version, commit in ((public, "v2", "a" * 40), (report, "v5", "b" * 40)):
        (folder / "evaluation-input.json").write_text(json.dumps({
            "dataset": {"repo": "owner/data", "version": version, "commit": commit,
                        "fingerprint": f"fingerprint-{version}"},
            "protocol": "zero-shot", "result": {"test": {}},
        }))
    (report / "runtime.json").write_text(json.dumps({
        "source_sha256": "source", "model_sha256": e2.MODEL_SHA256}))
    model = SimpleNamespace(uri=e2.MODEL, sha256=e2.MODEL_SHA256, golden_ok=True)
    calls = []
    monkeypatch.setattr(e2, "require_environment", lambda: None)
    monkeypatch.setattr(e2, "source_identity", lambda: {"source_sha256": "source"})
    monkeypatch.setattr("pettrace.evaluation.core.load_model", lambda uri: model)
    monkeypatch.setattr("pettrace.experiments.zero_shot.publish_results",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or {"published": True})

    assert e2.resume_publication(report)["published"] is True
    assert calls[0][1]["private_dataset_version"] == "v5"
    assert calls[0][0][1].dataset.version == "v5"
