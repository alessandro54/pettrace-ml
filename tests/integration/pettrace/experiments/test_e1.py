"""Headless E1 evaluates both pinned domains and can retry publication alone."""

import json
from types import SimpleNamespace

from pettrace.experiments import e1


def dataset(version, commit):
    photo = SimpleNamespace(path="photo.jpg", animal="animal", species="cat", split="test")
    return SimpleNamespace(repo="owner/data", version=version, commit=commit,
                           fingerprint=f"fingerprint-{version}", photos=[photo],
                           animals={"animal": {}}, name=f"data@{version}")


def test_run_generates_both_reports_and_mandatorily_publishes(tmp_path, monkeypatch, capsys):
    public, private = dataset("v2", "a" * 40), dataset("v5", "b" * 40)
    evaluations, saved, published = [], [], []
    model = SimpleNamespace(uri=e1.MODEL, sha256={"encoder.onnx": e1.ENCODER_SHA256},
                            golden_ok=True, pipeline=SimpleNamespace(detector=None))

    monkeypatch.setattr(e1, "require_environment", lambda: None)
    monkeypatch.setattr(e1, "_domains", lambda version, commit, cache: (
        public, private, {"repo": "composite/public", "version": "v2", "commit": "a" * 40,
                          "fingerprint": "public-fingerprint", "sources": []}))
    monkeypatch.setattr(e1, "source_identity", lambda: {"source_sha256": "source"})
    monkeypatch.setattr("pettrace.evaluation.core.load_model", lambda uri: model)

    def evaluate(ds, pipeline, zero_shot):
        evaluations.append((ds.version, zero_shot))
        return SimpleNamespace(dataset=ds, protocol="zero-shot", result={"test": {}})

    monkeypatch.setattr("pettrace.evaluation.core.evaluate", evaluate)
    monkeypatch.setattr("pettrace.evaluation.core.report", lambda result: "metrics")
    monkeypatch.setattr("pettrace.evaluation.figures.save",
                        lambda ev, model, folder: saved.append((ev.dataset.version, folder)))
    monkeypatch.setattr("pettrace.evaluation.research.save_benchmark_comparison",
                        lambda *args: ("| comparison |", None))

    def publish(*args, **kwargs):
        published.append((args, kwargs))
        return {"published": True, "run_id": "private", "public_run_id": "public", "url": "run-url"}

    monkeypatch.setattr("pettrace.experiments.zero_shot.publish_results", publish)
    report = tmp_path / "e1"
    result = e1.run(report, private_version="v5", private_commit="b" * 40,
                    cache_dir=tmp_path / "cache")

    assert evaluations == [("v2", True), ("v5", True)]
    assert {version for version, _ in saved} == {"v2", "v5"}
    assert len(published) == 1
    assert published[0][1]["private_dataset_version"] == "v5"
    assert published[0][1]["extra_design"]["private_dataset_commit"] == "b" * 40
    assert (report / "benchmark-comparison.md").is_file()
    assert "| comparison |" not in capsys.readouterr().out
    assert json.loads((report / "dataset-selection.json").read_text())["dataset"]["version"] == "v5"
    assert result["published"] is True


def test_resume_republishes_saved_metrics_without_evaluation(tmp_path, monkeypatch):
    report, public = tmp_path / "e1", tmp_path / "e1" / "public_test"
    public.mkdir(parents=True)
    for folder, version, commit in ((public, "v1", "a" * 40), (report, "v5", "b" * 40)):
        (folder / "evaluation-input.json").write_text(json.dumps({
            "dataset": {"repo": "owner/data", "version": version, "commit": commit,
                        "fingerprint": f"fingerprint-{version}"},
            "protocol": "zero-shot", "result": {"test": {}},
        }))
    (report / "runtime.json").write_text(json.dumps({
        "source_sha256": "source", "model_sha256": {"encoder.onnx": e1.ENCODER_SHA256}}))
    model = SimpleNamespace(uri=e1.MODEL, sha256={"encoder.onnx": e1.ENCODER_SHA256}, golden_ok=True)
    calls = []
    monkeypatch.setattr(e1, "require_environment", lambda: None)
    monkeypatch.setattr(e1, "source_identity", lambda: {"source_sha256": "source"})
    monkeypatch.setattr("pettrace.evaluation.core.load_model", lambda uri: model)
    monkeypatch.setattr("pettrace.experiments.zero_shot.publish_results",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or {"published": True})

    assert e1.resume_publication(report)["published"] is True
    assert calls[0][1]["private_dataset_version"] == "v5"
    assert calls[0][0][1].dataset.version == "v5"
