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
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/e3_mlp_head_triplet_20.ipynb").read_text())
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


def test_preparation_force_refresh_ignores_stale_environment_and_colab_secrets(monkeypatch):
    import getpass
    import sys

    from pettrace.data.workflow import configure_access
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
    configure_access(names=("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"), force=True)
    import os
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "new-fixture-user"
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "new-fixture-password"
    assert os.environ["HF_TOKEN"] == "unrelated-fixture-token"
    assert requests == ["MLFLOW_TRACKING_USERNAME: ", "MLFLOW_TRACKING_PASSWORD: "]


@pytest.mark.parametrize('indexed', [False, True])
def test_volume_reuses_cleaning_and_preserves_fixed_v1_splits(tmp_path, monkeypatch, indexed):
    from dataclasses import replace
    from pettrace.data import preparation, volume
    from pettrace.data.sources import discover_body_ids

    config = setup_pipeline(tmp_path, monkeypatch)
    reference = prepare(config)
    monkeypatch.setattr("pettrace.evaluation.research.PUBLIC_FINGERPRINT", reference.fingerprint)
    rows, sources = [], []
    for source, root in config.source_roots.items():
        found, _ = discover_body_ids(root, source)
        rows.extend(found)
        sources.append({"ref": volume.CAPACITY[source]["ref"], "sampled_photos": len(found)})
    monkeypatch.setattr(preparation, "acquire_candidates", lambda config, **kw: (rows, {"sources": sources, "skipped_paths": []}))
    if indexed:
        from pettrace.data.lookup import lookup_duplicates
        from pettrace.data.sources import write_review
        review = write_review(rows, Path(config.data_root) / 'source-20-seed42/review.csv')
        config = replace(config, candidate_index=str(lookup_duplicates(review, tmp_path / 'indices')))
    previous_inferences = FakeDetector.calls
    expanded = prepare(replace(config, source_percent=20, reference_dataset=str(reference.root)))
    assert FakeDetector.calls == previous_inferences
    def signature(items):
        return {(r["animal"], r["sha256"], r["split"]) for r in items if r["split"] in {"val", "test"}}
    assert signature(expanded.rows) == signature(reference.rows)
    reserved = {r["animal"] for r in reference.rows if r["split"] in {"val", "test"}}
    assert not reserved & {r["animal"] for r in expanded.rows if r["split"] == "train"}
    for counts in expanded.summary["processing"].values():
        assert counts["train_photos"] + counts["excluded_photos"] == counts["processed_photos"]
        assert sum(counts["exclusion_reasons"].values()) == counts["excluded_photos"]


def test_cuda_preparation_retains_cpu_detections_and_filter_outcomes(tmp_path, monkeypatch):
    from itertools import cycle
    from pettrace.data import screening_gpu

    config = setup_pipeline(tmp_path, monkeypatch)
    baseline = prepare(config)
    cache = Path(config.data_root) / 'animal-screening-cache'
    original = {p.name: p.read_bytes() for p in cache.glob('*.json')}
    times = cycle((0, .02, 1, 1.005))
    monkeypatch.setattr(screening_gpu, 'perf_counter', lambda: next(times))
    monkeypatch.setattr(screening_gpu, 'cuda_detector', lambda *a: (FakeDetector('', {}), {'provider': 'CUDAExecutionProvider'}))
    accelerated = prepare(replace(config, screening_provider='cuda'))
    assert accelerated.fingerprint != baseline.fingerprint
    fields = ('sha256', 'animal', 'species', 'split')
    assert sorted(tuple(r[f] for f in fields) for r in accelerated.rows) == sorted(tuple(r[f] for f in fields) for r in baseline.rows)
    report = json.loads((accelerated.root / 'animal-screening.json').read_text())
    assert report['gpu_validation']['passed']
    assert {r['execution_provider'] for r in report['records']} == {'CPUExecutionProvider'}
    assert original == {p.name: p.read_bytes() for p in cache.glob('*.json')}
    inputs = json.loads((accelerated.root / 'prepared.json').read_text())['inputs']
    assert 'data/screening_gpu.py' in inputs['implementation']


def test_indexed_preparation_screens_only_candidates_and_preserves_audit(tmp_path, monkeypatch):
    import csv
    from pettrace.data import preparation
    from pettrace.data.lookup import lookup_duplicates, load_candidates
    from pettrace.data.sources import REVIEW_FIELDS, write_review

    config = setup_pipeline(tmp_path, monkeypatch)
    cats = Path(config.source_roots['lcw']) / 'CatFullDataset'
    shutil.copyfile(cats / '1/0.png', cats / '1/1.png')  # within-identity copy
    shutil.copyfile(cats / '2/0.png', cats / '3/0.png')  # conflicting identities
    selected, _ = preparation.acquire_candidates(config)
    review = write_review(selected, Path(config.data_root) / 'pilot-seed42/review.csv')
    with review.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row['animal'] == 'dogs-world:dog:2' and Path(row['path']).name in {'2-0.png', '2-1.png'}:
            row['decision'] = 'drop'
    with review.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = lookup_duplicates(review, tmp_path / 'indices')
    candidates, excluded = load_candidates(report, review)
    seen = []
    original_screen = preparation.screen_animals
    def screen(rows, *args, **kwargs):
        seen.extend(rows)
        return original_screen(rows, *args, **kwargs)
    monkeypatch.setattr(preparation, 'screen_animals', screen)
    dataset = prepare(replace(config, candidate_index=str(report)))
    assert {r['path'] for r in seen} == {r['path'] for r in candidates}
    assert not set(excluded) & {r['path'] for r in seen}
    assert len(seen) < len(rows)
    assert len({r['sha256'] for r in seen}) == len(seen)
    audit = json.loads((dataset.root / 'audit.json').read_text())
    assert audit['conflicting_identities_quarantined'] == ['lcw:cat:2', 'lcw:cat:3']
    receipt = json.loads((dataset.root / 'prepared.json').read_text())
    assert receipt['inputs']['candidate_index_sha256'] == sha256(report / 'lookup.json')
    assert (dataset.root / 'candidates.csv').read_bytes() == (report / 'candidates.csv').read_bytes()
    assert not {'lcw:cat:2', 'lcw:cat:3', 'dogs-world:dog:2'} & {r['animal'] for r in dataset.rows}
    assert sum(dataset.summary['exclusion_reasons'].values()) + len(dataset.rows) == len(rows)
    # A stale index must fail before a detector is loaded or inference starts.
    (report / 'candidates.csv').write_text('changed')
    monkeypatch.setattr(preparation, 'Detector', lambda *a: pytest.fail('No detector with invalid index'))
    with pytest.raises(ValueError, match='Candidate index'):
        prepare(replace(config, candidate_index=str(report)))
