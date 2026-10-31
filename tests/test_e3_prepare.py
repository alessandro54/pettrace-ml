import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from pettrace.data.preparation import PrepareConfig, load_prepared, prepare
from pettrace.data.curation import sha256


class FakeDetector:
    calls = 0

    def __init__(self, root, step):
        self.classes, self.size, self.input_name = [15, 16], 640, "images"
        self.session = SimpleNamespace(run=self.run)

    def letterbox(self, image):
        return np.asarray(image)[None], 1, 0, 0

    def run(self, outputs, inputs):
        type(self).calls += 1
        marker = inputs["images"][0, 0, 0, 0]
        raw = np.zeros((1, 84, 2), dtype=np.float32)
        raw[0, :4, 0] = (8, 8, 6, 6)
        raw[0, 4 + (15 if marker == 40 else 16), 0] = .95
        if marker == 99:
            raw[0, :4, 1] = (25, 25, 6, 6)
            raw[0, 20, 1] = .9
        return [raw]


def setup_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr("pettrace.data.preparation.Detector", FakeDetector)
    FakeDetector.calls = 0
    roots = {"dogs-world": tmp_path / "raw-dogs", "lcw": tmp_path / "raw-cats"}
    rng = np.random.default_rng(42)
    for source, root in roots.items():
        if source == "dogs-world":
            (root / "metadata").mkdir(parents=True)
            (root / "images").mkdir()
        for identity in range(16):
            for photo in range(5):
                pixels = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
                pixels[0, 0, 0] = 40 if source == "lcw" else 80
                if source == "dogs-world" and identity == 0 and photo < 2:
                    pixels[0, 0, 0] = 99
                if source == "dogs-world":
                    path = root / f"images/{identity}-{photo}.png"
                    (root / f"metadata/{identity}-{photo}.json").write_text(json.dumps(
                        {"path": str(path.relative_to(root)), "identities": [{"identity": str(identity)}]}))
                else:
                    path = root / f"CatFullDataset/{identity}/{photo}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(path)
    (roots["lcw"] / "CatFullDataset/0/0.png").write_bytes(b"unreadable source image")
    model = tmp_path / "model"
    model.mkdir()
    (model / "detector.onnx").write_bytes(b"fake model; runtime replaced by fixture")
    (model / "pipeline.yaml").write_text(yaml.safe_dump(
        {"steps": [{"op": "detect_crop", "enabled": True, "file": "detector.onnx", "input_size": 640}]}))
    return PrepareConfig(data_root=str(tmp_path / "data"), detector_dir=str(model),
                         source_roots={k: str(v) for k, v in roots.items()},
                         identities_per_species=16, photos_per_identity=5, visual_sample_per_species=1,
                         expected_detector_sha256=sha256(model / "detector.onnx"))


def test_pipeline_creates_portable_snapshot_reuses_it_and_preserves_pending_decisions(tmp_path, monkeypatch):
    config = setup_pipeline(tmp_path, monkeypatch)

    def no_network(*args, **kwargs):
        raise AssertionError("Manual source roots must not use network")

    monkeypatch.setattr("requests.sessions.Session.request", no_network)
    dataset = prepare(config)
    assert dataset.summary["candidate_photos"] == 160
    assert dataset.summary["retained_photos"] == 154  # two detections reject an entire five-photo ID, plus one corrupt cat.
    assert dataset.summary["retained_identities"] == 31
    assert dataset.summary["exclusion_reasons"] == {"multiple_animals": 2, "insufficient_distinct_photos": 3, "unreadable": 1}
    audit = json.loads((dataset.root / "audit.json").read_text())
    assert audit["review_decisions"] == {"": 160}
    assert audit["retained_with_unverified_source_label"] == 154
    assert all(Path(row["path"]).is_relative_to(dataset.root) for row in dataset.rows)
    assert all(len({r["split"] for r in dataset.rows if r["animal"] == row["animal"]}) == 1 for row in dataset.rows)
    # Exercise the actual training-notebook input cell without private access or model fitting.
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/e3_mlp_head_triplet.ipynb").read_text())
    cell = next(c for c in notebook["cells"] if c["id"] == "e3-load-prepared")
    report = tmp_path / "training-report"
    report.mkdir()
    namespace = {"load_prepared": load_prepared, "PREPARED_DATASET_ROOT": dataset.root,
                 "REPORT": report, "json": json, "shutil": shutil, "sha256": sha256, "TRAINING_DATASET_REPO": ""}
    exec(compile("".join(cell["source"]), "training-input", "exec"), namespace)
    assert namespace["manifest_rows"] == dataset.rows
    assert json.loads((report / "data-preparation.json").read_text())["prepared_dataset_fingerprint"] == dataset.fingerprint
    calls = FakeDetector.calls
    repeated = prepare(config)
    assert repeated.fingerprint == dataset.fingerprint and FakeDetector.calls == calls
    pointer = Path(config.data_root) / "prepared/latest.json"
    assert load_prepared(pointer).fingerprint == dataset.fingerprint
    relocated = tmp_path / "relocated"
    shutil.copytree(dataset.root, relocated)
    for root in config.source_roots.values():
        shutil.rmtree(root)
    assert load_prepared(relocated).fingerprint == dataset.fingerprint
    assert all(Path(row["path"]).is_relative_to(relocated) for row in load_prepared(relocated).rows)


def test_snapshot_rejects_tampered_images_and_manifest(tmp_path, monkeypatch):
    config = setup_pipeline(tmp_path, monkeypatch)
    dataset = prepare(config)
    image = Path(dataset.rows[0]["path"])
    original = image.read_bytes()
    image.write_bytes(b"modified")
    with pytest.raises(ValueError, match="artifact changed"):
        load_prepared(dataset.root)
    image.write_bytes(original)
    with (dataset.root / "manifest.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="artifact changed"):
        prepare(config)


def test_failed_preparation_never_updates_latest_or_exposes_partial_snapshot(tmp_path, monkeypatch):
    config = setup_pipeline(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Detector hash"):
        prepare(replace(config, expected_detector_sha256="0" * 64))
    assert not (Path(config.data_root) / "prepared/latest.json").exists()
    with pytest.raises(ValueError, match="keep or drop"):
        prepare(replace(config, review_policy="strict"))
    output = Path(config.data_root) / "prepared"
    assert not (output / "latest.json").exists()
    assert not list(output.glob(".preparing-*"))
    assert not list(output.glob("*/prepared.json"))

    successful = prepare(config)
    before = (output / "latest.json").read_bytes()
    with pytest.raises(ValueError, match="keep or drop"):
        prepare(replace(config, review_policy="strict"))
    assert (output / "latest.json").read_bytes() == before
    assert load_prepared(output / "latest.json").fingerprint == successful.fingerprint


def test_detector_acquisition_reads_only_pinned_detector_and_spec_and_checks_cache(tmp_path, monkeypatch):
    import mlflow
    from pettrace.data.preparation import acquire_detector

    source = tmp_path / "upstream"
    source.mkdir()
    (source / "detector.onnx").write_bytes(b"test detector")
    (source / "pipeline.yaml").write_text("steps: []\n")
    calls = []
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda value: None)
    monkeypatch.setattr(mlflow, "set_registry_uri", lambda value: None)
    monkeypatch.setattr(mlflow, "get_registry_uri", lambda: "https://ml.pettrace.app")
    monkeypatch.setattr(mlflow, "MlflowClient", lambda: SimpleNamespace(
        get_model_version=lambda name, version: SimpleNamespace(source="fake://artifacts/e2")))

    def download(*, artifact_uri, dst_path):
        filename = artifact_uri.rsplit("/", 1)[-1]
        calls.append(filename)
        shutil.copy2(source / filename, Path(dst_path) / filename)

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download)
    expected = sha256(source / "detector.onnx")
    directory = acquire_detector("models:/pettrace-embedder/2", tmp_path / "models", expected_sha256=expected)
    assert calls == ["detector.onnx", "pipeline.yaml"]
    assert acquire_detector("models:/pettrace-embedder/2", tmp_path / "models", expected_sha256=expected) == directory
    assert len(calls) == 2
    (directory / "pipeline.yaml").write_text("modified spec")
    with pytest.raises(ValueError, match="artifacts changed"):
        acquire_detector("models:/pettrace-embedder/2", tmp_path / "models", expected_sha256=expected)
    with pytest.raises(ValueError, match="pinned"):
        acquire_detector("models:/pettrace-embedder@production", tmp_path / "models", expected_sha256=expected)


def test_detector_401_is_recoverable_but_permission_errors_are_preserved(tmp_path, monkeypatch):
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import UNAUTHENTICATED, PERMISSION_DENIED
    from pettrace.data.preparation import DetectorAuthenticationError, acquire_detector

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda value: None)
    monkeypatch.setattr(mlflow, "set_registry_uri", lambda value: None)
    monkeypatch.setattr(mlflow, "get_registry_uri", lambda: "https://ml.pettrace.app")

    def reject(*args, **kwargs):
        raise MlflowException("fixture rejection", error_code=UNAUTHENTICATED)

    monkeypatch.setattr(mlflow, "MlflowClient", lambda: SimpleNamespace(get_model_version=reject))
    with pytest.raises(DetectorAuthenticationError, match="401") as exc:
        acquire_detector("models:/pettrace-embedder/2", tmp_path / "models")
    assert "force=True" in str(exc.value)
    assert exc.value._render_traceback_() == [str(exc.value)]

    def forbidden(*args, **kwargs):
        raise MlflowException("permission denied", error_code=PERMISSION_DENIED)

    monkeypatch.setattr(mlflow, "MlflowClient", lambda: SimpleNamespace(get_model_version=forbidden))
    with pytest.raises(MlflowException) as permission:
        acquire_detector("models:/pettrace-embedder/2", tmp_path / "models")
    assert permission.value.get_http_status_code() == 403


def test_notebook_force_refresh_ignores_stale_environment_and_colab_secrets(monkeypatch):
    import getpass
    import sys

    notebook = json.loads((Path(__file__).parents[1] / "datasets/prepare_training.ipynb").read_text())
    source = "".join(next(c for c in notebook["cells"] if c["id"] == "training-data-access")["source"])
    requests = []
    answers = iter(["new-fixture-user", "new-fixture-password"])
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "old-fixture-user")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "old-fixture-password")
    monkeypatch.setenv("HF_TOKEN", "unrelated-fixture-token")
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(userdata=SimpleNamespace(
        get=lambda name: pytest.fail("Forced refresh must bypass stored Colab Secrets"))))

    def prompt(label):
        requests.append(label)
        return next(answers)

    monkeypatch.setattr(getpass, "getpass", prompt)
    namespace = {}
    exec(compile(source, "credential-access", "exec"), namespace)
    namespace["configure_access"](names=("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"), force=True)
    import os
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "new-fixture-user"
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "new-fixture-password"
    assert os.environ["HF_TOKEN"] == "unrelated-fixture-token"
    assert requests == ["MLFLOW_TRACKING_USERNAME: ", "MLFLOW_TRACKING_PASSWORD: "]
