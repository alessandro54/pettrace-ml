"""Production notebook checks use mocked clients; never contact the production server."""

import json
from types import SimpleNamespace
from unittest.mock import Mock
from functools import partial

import pytest

from pettrace.evaluation import core as evaluation
from pettrace import tracking


ENCODER_SHA256 = "4773e649274ca8774c4a63d45f0438fe598ee7ba67e65def97537aa0eca9a576"
publish = partial(tracking.publish_evaluation, experiment="E1",
                  expected_model_uri="models:/pettrace-embedder/1",
                  expected_model_sha256={"encoder.onnx": ENCODER_SHA256},
                  expected_dataset_version="v4", expected_protocol="zero-shot")

@pytest.fixture
def publication(tmp_path, monkeypatch):
    import mlflow

    experiment = SimpleNamespace(experiment_id="3", name="pettrace-reid-eval", lifecycle_stage="active")
    client = Mock()
    client.get_experiment_by_name.return_value = experiment
    client.search_runs.return_value = []
    client.create_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="new-run", status="RUNNING"))
    monkeypatch.setattr(mlflow, "MlflowClient", lambda: client)
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: tracking.PRODUCTION_URI)
    monkeypatch.setattr(mlflow, "get_registry_uri", lambda: tracking.PRODUCTION_URI)
    monkeypatch.setattr(mlflow, "active_run", lambda: None)
    log = Mock(return_value="new-run")
    monkeypatch.setattr(evaluation, "log_run", log)
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "runtime.json").write_text("{}")
    ds = SimpleNamespace(version="v4", repo="private/pettrace-reid", name="pettrace-reid@v4",
                         commit="fixed-commit", fingerprint="fixed-digest")
    ev = SimpleNamespace(dataset=ds, protocol="zero-shot", result={"rank1": 0.5})
    model = SimpleNamespace(uri="models:/pettrace-embedder/1", model_version="clip-vit-b32@v1",
                            golden_ok=True, sha256={"encoder.onnx": ENCODER_SHA256})
    return client, ev, model, tmp_path, log


def test_preflight_is_read_only(publication):
    client, *_ = publication
    tracking.production_context()
    assert [call[0] for call in client.method_calls] == ["get_experiment_by_name"]


@pytest.mark.parametrize("setting", ["get_tracking_uri", "get_registry_uri"])
def test_wrong_endpoint_fails_before_client_access(publication, monkeypatch, setting):
    import mlflow

    client, ev, model, report, log = publication
    monkeypatch.setattr(mlflow, setting, lambda: "file:///tmp/mlruns")
    with pytest.raises(ValueError):
        publish(ev, model, report, "sources")
    assert not client.method_calls
    log.assert_not_called()


def test_missing_experiment_is_not_created(publication):
    client, ev, model, report, log = publication
    client.get_experiment_by_name.return_value = None
    with pytest.raises(ValueError, match="existing production experiment"):
        publish(ev, model, report, "sources")
    client.create_experiment.assert_not_called()
    client.create_run.assert_not_called()
    log.assert_not_called()


def test_manifest_is_saved_before_upload_and_run_is_reused(publication):
    client, ev, model, report, log = publication

    def upload(*args, **kwargs):
        assert json.loads((report / "run.json").read_text())["run_id"] == "new-run"
        assert kwargs["tracking_experiment_id"] == "3"
        assert kwargs["run_id"] == "new-run"
        return "new-run"

    log.side_effect = upload
    assert publish(ev, model, report, "sources") == "new-run"
    manifest = json.loads((report / "run.json").read_text())
    client.search_runs.return_value = [SimpleNamespace(info=SimpleNamespace(run_id="new-run", status="FINISHED"))]
    assert publish(ev, model, report, "sources") == "new-run"
    client.create_run.assert_called_once()
    log.assert_called_once()
    assert json.loads((report / "run.json").read_text())["publication_key"] == manifest["publication_key"]


def test_failed_upload_resumes_the_existing_run(publication):
    client, ev, model, report, log = publication
    log.side_effect = ConnectionError("upload interrupted")
    with pytest.raises(ConnectionError):
        publish(ev, model, report, "sources")
    assert (report / "run.json").is_file()
    client.search_runs.return_value = [SimpleNamespace(info=SimpleNamespace(run_id="new-run", status="FAILED"))]
    log.side_effect = None
    assert publish(ev, model, report, "sources") == "new-run"
    client.create_run.assert_called_once()
    assert log.call_args.kwargs["run_id"] == "new-run"


def test_independent_executions_have_separate_publication_keys(publication):
    _, ev, model, report, _ = publication
    keys = []
    for execution_id in ("first", "first", "second"):
        (report / "runtime.json").write_text(json.dumps({"execution_id": execution_id}))
        publish(ev, model, report, "sources")
        keys.append(json.loads((report / "run.json").read_text())["publication_key"])
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]


def test_rejects_unverified_model_and_active_run(publication, monkeypatch):
    import mlflow

    client, ev, model, report, log = publication
    model.sha256["encoder.onnx"] = "unexpected"
    with pytest.raises(ValueError, match="verified expected model"):
        publish(ev, model, report, "sources")
    model.sha256["encoder.onnx"] = ENCODER_SHA256
    monkeypatch.setattr(mlflow, "active_run", lambda: object())
    with pytest.raises(RuntimeError, match="active MLflow run"):
        publish(ev, model, report, "sources")
    client.create_run.assert_not_called()
    log.assert_not_called()


def test_ambiguous_runs_are_not_overwritten(publication):
    client, ev, model, report, log = publication
    client.search_runs.return_value = [object(), object()]
    with pytest.raises(RuntimeError, match="Multiple evaluation runs"):
        publish(ev, model, report, "sources")
    client.create_run.assert_not_called()
    log.assert_not_called()


def test_shared_logger_resumes_without_creating_experiment(tmp_path, monkeypatch):
    import mlflow
    from unittest.mock import MagicMock
    from pettrace.data.benchmark import Dataset

    start = MagicMock()
    start.return_value.__enter__.return_value.info.run_id = "existing-run"
    monkeypatch.setattr(mlflow, "start_run", start)
    set_experiment = Mock()
    monkeypatch.setattr(mlflow, "set_experiment", set_experiment)
    for method in ("log_input", "log_params", "set_tags", "log_metrics", "log_artifact", "log_artifacts"):
        monkeypatch.setattr(mlflow, method, Mock())
    ds = Dataset("private/pettrace-reid", "v4", "commit", "a" * 64, str(tmp_path))
    ev = SimpleNamespace(dataset=ds, protocol="zero-shot",
                         result={"test": {"global": {"queries": 2}}, "test_threshold": {}})
    model = SimpleNamespace(uri="models:/pettrace-embedder/1", model_version="clip-vit-b32@v1",
                            registry_version="1", sha256={}, golden_fingerprint="golden", golden_ok=True)
    assert evaluation.log_run(ev, model, tracking_experiment_id="3", run_id="existing-run") == "existing-run"
    set_experiment.assert_not_called()
    assert start.call_args.kwargs["experiment_id"] == "3"
    assert start.call_args.kwargs["run_id"] == "existing-run"
    assert mlflow.log_input.call_args.args[0].digest == "a" * 32
    assert mlflow.log_params.call_args.args[0]["dataset.fingerprint"] == "a" * 64


def test_attachment_skips_old_run_when_publication_disabled(tmp_path, monkeypatch):
    import mlflow
    from tools import attach_notebook

    report = tmp_path / "report"
    report.mkdir()
    receipt = report / "publication.json"
    receipt.write_text(json.dumps({"published": False, "run_id": None}))
    (report / "run.json").write_text(json.dumps({"run_id": "old-run"}))
    notebook = tmp_path / "e1.ipynb"
    notebook.write_text(json.dumps({"metadata": {"pettrace": {
        "report": str(report), "publication_receipt": str(receipt),
    }}}))
    monkeypatch.setattr("sys.argv", ["attach_notebook.py", str(notebook)])
    start = Mock()
    monkeypatch.setattr(mlflow, "start_run", start)
    assert attach_notebook.main() == 0
    start.assert_not_called()
    receipt.write_text(json.dumps({"published": True, "run_id": "new-run"}))
    assert attach_notebook.main() == 1
    start.assert_not_called()


def test_shared_publication_accepts_another_experiment(publication):
    client, ev, model, report, log = publication
    model.uri = "models:/pettrace-embedder/2"
    model.model_version = "clip-vit-b32-crop@v1"
    ev.dataset.version = "v5"
    assert tracking.publish_evaluation(ev, model, report, "sources", experiment="E2") == "new-run"
    assert client.create_run.call_args.kwargs["tags"]["experiment"] == "E2"
    assert log.call_args.kwargs["experiment"] == "E2"
    assert json.loads((report / "run.json").read_text())["experiment"] == "E2"


def test_source_identity_does_not_require_a_repository_in_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    identity = tracking.source_identity()
    assert len(identity["source_sha256"]) == 64
    assert identity["package_version"]


def test_source_identity_includes_nested_modules_and_embedded_release_provenance(tmp_path, monkeypatch):
    package = tmp_path / "installed" / "pettrace"
    nested = package / "training"
    nested.mkdir(parents=True)
    module = nested / "triplet.py"
    module.write_text("seed = 42\n")
    build = {"git_sha": "a" * 40, "git_dirty": False, "uv_lock_sha256": "b" * 64}
    (package / "_build.json").write_text(json.dumps(build))
    monkeypatch.setattr(tracking, "__file__", str(package / "tracking.py"))
    first = tracking.source_identity()
    assert first["git_sha"] == build["git_sha"] and first["git_dirty"] is False
    assert first["build"]["uv_lock_sha256"] == build["uv_lock_sha256"]
    module.write_text("seed = 43\n")
    assert tracking.source_identity()["source_sha256"] != first["source_sha256"]
