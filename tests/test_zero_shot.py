"""Shared E1/E2 presentation and publication mechanics."""

from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from pettrace.experiments import zero_shot


def evaluation(version="v4", protocol="zero-shot"):
    dataset = SimpleNamespace(repo=f"repo/{version}", version=version, commit=f"commit-{version}",
                              fingerprint=f"fingerprint-{version}")
    return SimpleNamespace(dataset=dataset, protocol=protocol)


def test_publish_results_records_both_domains_and_final_receipts(tmp_path, monkeypatch):
    report, public = tmp_path / "run", tmp_path / "run" / "public_test"
    public.mkdir(parents=True)
    calls, provenance, inputs, artifacts = [], [], [], []

    def publish(ev, model, folder, source_sha256, **kwargs):
        calls.append((ev.dataset.version, folder, source_sha256, kwargs))
        return "public-run" if ev.dataset.version == "public-v1" else "private-run"

    client = SimpleNamespace(
        get_run=lambda run_id: SimpleNamespace(info=SimpleNamespace(status="FINISHED")),
        log_artifact=lambda run_id, filename, artifact_path: artifacts.append(
            (run_id, filename, artifact_path)),
    )
    monkeypatch.setattr(zero_shot, "publish_evaluation", publish)
    monkeypatch.setattr(zero_shot, "save_provenance", lambda folder, model: provenance.append(folder))
    monkeypatch.setattr(zero_shot, "production_context", lambda: (
        client, SimpleNamespace(experiment_id="3")))
    monkeypatch.setattr(zero_shot, "compact_json_uploads", nullcontext)
    monkeypatch.setattr("pettrace.experiments.e1.save_publication_input",
                        lambda ev, folder: inputs.append((ev.dataset.version, folder)))

    result = zero_shot.publish_results(
        evaluation("public-v1"), evaluation(), SimpleNamespace(sha256={"encoder.onnx": "abc"}),
        report, public, "source", experiment="E2", expected_model_uri="models:/encoder/1",
        expected_model_sha256={"encoder.onnx": "abc"}, private_dataset_version="v4",
        extra_design={"frozen_e1_reference_runs": {"public": "one", "private": "two"}},
    )

    assert [call[0] for call in calls] == ["public-v1", "v4"]
    assert provenance == [report, public]
    assert inputs == [("public-v1", public), ("v4", report)]
    assert json.loads((report / "publication.json").read_text()) == {
        "published": True, "run_id": "private-run", "public_run_id": "public-run"}
    assert json.loads((public / "publication.json").read_text()) == {
        "published": True, "run_id": "public-run"}
    design = json.loads((report / "experiment-design.json").read_text())
    assert design["mlflow_publication_required"] is True
    assert design["frozen_e1_reference_runs"]["public"] == "one"
    assert [artifact[0] for artifact in artifacts] == ["public-run", "private-run"]
    assert result == {"published": True, "run_id": "private-run", "public_run_id": "public-run",
                      "url": "https://ml.pettrace.app/#/experiments/3/runs/private-run"}


@pytest.mark.parametrize("public_protocol,private_protocol", [
    ("other", "zero-shot"), ("zero-shot", "other"), ("other", "other")])
def test_publish_results_rejects_non_zero_shot_protocols(tmp_path, public_protocol, private_protocol):
    with pytest.raises(ValueError, match="zero-shot"):
        zero_shot.publish_results(
            evaluation("public-v1", public_protocol), evaluation(protocol=private_protocol),
            SimpleNamespace(sha256={}), tmp_path / "run", tmp_path / "public", "source",
            experiment="E1", expected_model_uri="model", expected_model_sha256={},
            private_dataset_version="v4",
        )
