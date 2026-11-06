"""Recover published E3 evidence without a training or tracking write API."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pettrace.experiments.e3 import load_published


def test_e3_published_summary_is_read_only():
    path = Path(__file__).parents[1] / "notebooks/e3_mlp_head_triplet.ipynb"
    cell = next(c for c in json.loads(path.read_text())["cells"]
                if c.get("id") == "e3-published-seed-summary")
    source = "".join(cell["source"])
    assert cell["cell_type"] == "code"
    assert "client.get_run" in source and "client.get_model_version" in source
    assert not any(call in source for call in
                   ("publish_results", "register_candidate", "start_run", "log_batch", "log_artifact"))


@pytest.mark.parametrize("failure", [None, "receipt", "benchmark", "evaluation", "model", "golden", "registry", "source"])
def test_recover_published_e3_and_reuse_local_artifacts(tmp_path, monkeypatch, failure):
    import mlflow
    from pettrace.evaluation import core
    from pettrace.experiments import e3

    values = {d: json.dumps({"dataset": d, "test": {}}).encode() for d in ("public", "private")}
    expected = {d: {"run_id": d, "commit": d + "-commit",
                    "evaluation_sha256": hashlib.sha256(values[d]).hexdigest()} for d in values}
    model_hashes = {"encoder.onnx": "encoder-digest", "detector.onnx": "detector-digest"}
    calls = []

    class Client:
        # No create/log/register methods: reading evidence cannot publish a new experiment.
        def get_model_version(self, name, version):
            return SimpleNamespace(status="READY", run_id="other" if failure == "registry" else "private",
                                   source="changed" if failure == "source" else "mlflow-artifacts:/private/report/model")

        def get_run(self, run_id):
            params = {"dataset.commit": "changed" if failure == "benchmark" else expected[run_id]["commit"],
                      "protocol": "zero-shot: same-species gallery"}
            params.update({f"model.sha256.{name}": digest for name, digest in model_hashes.items()})
            return SimpleNamespace(info=SimpleNamespace(status="FINISHED", artifact_uri="mlflow-artifacts:/private"),
                                   data=SimpleNamespace(tags={"experiment": "E3"}, params=params))

        def download_artifacts(self, run_id, artifact, folder):
            calls.append((run_id, artifact))
            target = Path(folder) / Path(artifact).name
            if artifact == "report/model":
                target.mkdir()
                for name in (*model_hashes, "pipeline.yaml", "golden.npy"):
                    (target / name).write_bytes(b"fixture")
            elif artifact == "evaluation.json":
                target.write_bytes(b"changed" if failure == "evaluation" else values[run_id])
            else:
                target.write_text(json.dumps({"published": failure != "receipt", "run_id": run_id}))
            return str(target)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _: None)
    monkeypatch.setattr(mlflow, "set_registry_uri", lambda _: None)
    monkeypatch.setattr(e3, "production_context", lambda: (Client(), None))
    monkeypatch.setattr(core, "file_hashes", lambda _: {} if failure == "model" else model_hashes)
    monkeypatch.setattr(core, "load_model", lambda _: SimpleNamespace(golden_ok=failure != "golden", pipeline=object()))
    monkeypatch.setattr(core, "report", lambda result: result["dataset"])
    if failure:
        with pytest.raises(ValueError):
            load_published(expected, model_hashes, "3")
    else:
        model, results = load_published(expected, model_hashes, "3")
        model, results = load_published(expected, model_hashes, "3")
        assert model.golden_ok
        assert set(results) == {"public", "private"}
        assert calls.count(("private", "report/model")) == 1
        assert all(calls.count((d, "evaluation.json")) == 1 for d in expected)
