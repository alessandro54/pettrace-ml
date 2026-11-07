"""Source acquisition percentages must not be confused with effective training volume."""
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_expanded_training_rejects_test_leakage_and_changed_validation():
    from pettrace.experiments.e3 import _validate_benchmark_separation

    val = {"animal": "v", "species": "cat", "sha256": "v", "split": "val"}
    test = {"animal": "t", "species": "cat", "sha256": "t", "split": "test"}
    train = {"animal": "a", "species": "cat", "sha256": "a", "split": "train"}
    _validate_benchmark_separation([train, val], [val, test], set())
    for changed in ({**train, "animal": "t"}, {**train, "sha256": "t"}):
        with pytest.raises(ValueError, match="overlaps"):
            _validate_benchmark_separation([changed, val], [val, test], set())
    with pytest.raises(ValueError, match="validation"):
        _validate_benchmark_separation([train, {**val, "sha256": "new"}], [val, test], set())


def test_public_guard_rejects_images_shared_with_private_benchmark(tmp_path):
    from pettrace.data.verification import sha256
    from pettrace.experiments.e3 import _validate_benchmark_separation

    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"same photo")
    row = {"animal": "a", "species": "cat", "sha256": sha256(photo), "split": "test"}
    with pytest.raises(ValueError, match="shared with the PetTrace benchmark"):
        _validate_benchmark_separation([], [row], {sha256(photo)})


def test_e3_changed_private_tag_stops_before_download(tmp_path, monkeypatch):
    import huggingface_hub
    from pettrace.experiments.e3 import load_benchmarks

    downloads = []
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        dataset_info=lambda *a, **k: SimpleNamespace(sha="changed")))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: downloads.append(k))
    with pytest.raises(ValueError, match="tag changed"):
        load_benchmarks([], tmp_path, private_version="v4", private_commit="pinned")
    assert downloads == []


def test_load_benchmarks_writes_verified_public_selection(tmp_path, monkeypatch, capsys):
    import huggingface_hub
    from pettrace.data import benchmark
    from pettrace.data import snapshots
    from pettrace.evaluation import research
    from pettrace.experiments.e3 import load_benchmarks

    private_photo = tmp_path / "private.jpg"
    private_photo.write_bytes(b"private")
    private = SimpleNamespace(
        repo="private/repo", version="v4", commit="pinned", name="private-v4",
        photos=[SimpleNamespace(path=private_photo)])
    val = {"animal": "val", "species": "cat", "sha256": "v", "split": "val"}
    test = {"animal": "test", "species": "cat", "sha256": "t", "split": "test"}
    prepared = SimpleNamespace(rows=[val, test], fingerprint="prepared")
    public = SimpleNamespace(
        repo="public/repo", version="v1", commit="public-commit", fingerprint="public-test",
        photos=[object()], animals={"test": {}})
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        dataset_info=lambda *a, **k: SimpleNamespace(sha="pinned")))
    sessions = []
    monkeypatch.setattr(huggingface_hub, "configure_http_backend",
                        lambda *, backend_factory: sessions.append(backend_factory))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: tmp_path)
    monkeypatch.setattr(benchmark, "from_dir", lambda *a, **k: private)
    monkeypatch.setattr(snapshots, "load_training", lambda **k: (prepared, {"commit": "public-commit"}))
    monkeypatch.setattr(research, "public_test_dataset", lambda *a: public)

    result = load_benchmarks(
        [{"animal": "train", "species": "dog", "sha256": "a", "split": "train"}, val],
        tmp_path, private_version="v4", private_commit="pinned")
    assert result.private is private and result.public is public
    assert len(sessions) == 1
    assert "Benchmarks ready" in capsys.readouterr().out
    selection = json.loads((result.public_report / "dataset-selection.json").read_text())
    assert selection["scope"] == "published_test_only"
    assert selection["rows"] == [{"animal": "test", "species": "cat", "sha256": "t"}]


def test_both_domains_receive_training_volume_and_final_receipts(tmp_path, monkeypatch):
    import mlflow
    from contextlib import nullcontext
    from pettrace import tracking
    from pettrace.experiments import e3
    from pettrace.experiments.e3 import publish_results

    @dataclass
    class Config:
        seed: int = 42

    batches, tags, artifacts = {}, {}, {}

    def log_batch(run_id, metrics, params):
        batches[run_id] = ({m.key: m.value for m in metrics}, {p.key: p.value for p in params})

    client = SimpleNamespace(log_batch=log_batch,
        set_tag=lambda run, key, value: tags.update({(run, key): value}),
        get_run=lambda run: SimpleNamespace(info=SimpleNamespace(status="FINISHED")),
        log_artifact=lambda run, path, **kwargs: artifacts.update({run: json.loads(Path(path).read_text())}))
    monkeypatch.setattr(mlflow, "MlflowClient", lambda: client)
    monkeypatch.setattr(e3, "compact_json_uploads", nullcontext)
    monkeypatch.setattr(tracking, "save_provenance", lambda *a: None)
    monkeypatch.setattr(tracking, "publish_evaluation", lambda evaluation, *a, **k: evaluation.domain)
    public = tmp_path / "public_test"
    public.mkdir()
    for name in ("dataset-coverage.json", "training-config.json", "training-summary.json", "sources.json"):
        (tmp_path / name).write_text("{}")
    dataset = SimpleNamespace(repo="benchmark", version="v1", commit="pinned", fingerprint="digest")
    public_evaluation = SimpleNamespace(domain="public", dataset=dataset, protocol="zero-shot")
    private_evaluation = SimpleNamespace(domain="private", dataset=dataset, protocol="zero-shot")
    coverage = {"dataset.sample_requested_pct": 20, "dataset.sampled_pct": 19.999,
                "dataset.train_pct": 12, "dataset.train_photos": 100, "dataset.train_identities": 10}
    result = publish_results(
        public_evaluation, private_evaluation, SimpleNamespace(sha256={"encoder.onnx": "hash"}),
        tmp_path, public, "source", experiment="E3", source_sample_pct=20, config=Config(),
        training_summary={"history": [{"epoch": 1, "train.loss": .1, "train.active_triplets": .2,
            "val.macro_species_mAP": .3, "training.optimizer_steps": 1,
            "training.examples_seen": 32, "training.unique_photos_seen": 32}],
            "best_epoch": 1, "best_val_mAP": .3},
        prepared_fingerprint="prepared", training_source={"commit": "pinned"},
        coverage_metrics=coverage,
        coverage_params={"dataset.reference.scope": "original_archives_before_filtering"},
        private_dataset_version="v1")
    assert result == {"run_id": "private", "public_run_id": "public"}
    assert batches["public"] == batches["private"]
    assert batches["public"][0]["dataset.train_photos"] == 100
    assert batches["public"][0]["dataset.sample_requested_pct"] == 20
    assert tags["public", "source_sample_pct"] == tags["private", "source_sample_pct"] == "20"
    assert artifacts["public"]["published"] and artifacts["private"]["published"]
    assert artifacts["private"]["public_run_id"] == "public"


def test_publication_cells_can_retry_after_second_domain_fails(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import mlflow
    from pettrace import tracking
    from pettrace.experiments import e3
    from pettrace.experiments.e3 import publish_results

    @dataclass
    class Config:
        seed: int = 42

    monkeypatch.setattr(e3, "compact_json_uploads", nullcontext)
    uploads, calls = {}, []
    client = SimpleNamespace(
        log_batch=lambda *a, **kw: None, set_tag=lambda *a: None,
        get_run=lambda run: SimpleNamespace(info=SimpleNamespace(status="FINISHED")),
        log_artifact=lambda run, path, **kw: uploads.update({run: json.loads(Path(path).read_text())}))
    monkeypatch.setattr(mlflow, "MlflowClient", lambda: client)

    fail_private = True

    def publish(evaluation, model, report, source, **kwargs):
        calls.append(evaluation)
        assert (report / "experiment-design.json").is_file()
        assert kwargs["expected_protocol"] == "zero-shot"
        if evaluation.domain == "private" and fail_private:
            raise ConnectionError("Upload interrupted")
        return evaluation.domain

    monkeypatch.setattr(tracking, "publish_evaluation", publish)
    monkeypatch.setattr(tracking, "save_provenance", lambda *a: None)
    public_report = tmp_path / "public_test"
    public_report.mkdir()
    for name in ("dataset-coverage.json", "training-config.json", "training-summary.json", "sources.json"):
        (tmp_path / name).write_text("{}")
    dataset = SimpleNamespace(repo="benchmark", version="v1", commit="pinned", fingerprint="digest")
    public_evaluation = SimpleNamespace(domain="public", dataset=dataset, protocol="zero-shot")
    private_evaluation = SimpleNamespace(domain="private", dataset=dataset, protocol="zero-shot")
    kwargs = dict(experiment="E3", source_sample_pct=50, config=Config(),
        training_summary={"history": [], "best_epoch": 1, "best_val_mAP": .3},
        prepared_fingerprint="prepared", training_source={"commit": "pinned"},
        coverage_metrics={"dataset.train_photos": 100}, coverage_params={}, private_dataset_version="v1")
    with pytest.raises(ConnectionError, match="Upload interrupted"):
        publish_results(public_evaluation, private_evaluation, SimpleNamespace(sha256={"encoder.onnx": "hash"}),
                        tmp_path, public_report, "source", **kwargs)
    assert not json.loads((tmp_path / "publication.json").read_text())["published"]
    fail_private = False
    publish_results(public_evaluation, private_evaluation, SimpleNamespace(sha256={"encoder.onnx": "hash"}),
                    tmp_path, public_report, "source", **kwargs)
    assert [evaluation.domain for evaluation in calls] == ["public", "private", "public", "private"]
    assert uploads == {"public": {"published": True, "run_id": "public"},
                       "private": {"published": True, "run_id": "private", "public_run_id": "public"}}
