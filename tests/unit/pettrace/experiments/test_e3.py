"""E3 comparisons read published controls and never evaluate or publish them again."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pettrace.experiments.e3 import FROZEN_RUNS, frozen_controls


def test_frozen_controls_validate_data_models_and_queries(tmp_path, monkeypatch):
    rows = [{"query": 0, "photo": "runtime/images/cat/one.jpg", "animal": "cat:one", "species": "cat", "gallery_animals": 3}]
    frozen_rows = [{**rows[0], "query": 91, "photo": "cache/images/cat/one.jpg"}]
    evaluation = SimpleNamespace(dataset=SimpleNamespace(commit="fixed", fingerprint="data"),
                                 result={"test": {"per_query": rows}})
    bad = {"commit": False, "query": False}
    calls = []

    class Client:
        def get_run(self, run_id):
            label = next(k for k, v in FROZEN_RUNS["public"].items() if v == run_id)
            return SimpleNamespace(info=SimpleNamespace(status="FINISHED"), data=SimpleNamespace(tags={"experiment": label}, params={
                "dataset.commit": "wrong" if bad["commit"] else "fixed", "dataset.fingerprint": "data", "protocol": "zero-shot: fixed",
                "model.sha256.encoder.onnx": "4773e649274ca8774c4a63d45f0438fe598ee7ba67e65def97537aa0eca9a576",
                "model.sha256.detector.onnx": "da8f9b9ebea5157cdcefa7bea9fd782249ebdf119f917c07d64e5b41942b9c9d"}))

        def download_artifacts(self, run_id, artifact, dest):
            calls.append((run_id, artifact))
            path = Path(dest) / Path(artifact).name
            result = {"test": {"per_query": [] if bad["query"] else frozen_rows}} if artifact == "evaluation.json" else {"global": {"mAP": .5}}
            path.write_text(json.dumps(result))
            return str(path)

    monkeypatch.setattr("pettrace.experiments.e3.production_context", lambda: (Client(), None))
    monkeypatch.setattr("pettrace.evaluation.core.evaluate", lambda *a, **k: pytest.fail("Frozen controls must not rerun"))
    controls, summaries = frozen_controls(evaluation, "public", tmp_path)
    assert set(controls) == set(summaries) == {"E1", "E2"}
    assert len(calls) == 4
    assert json.loads((tmp_path / "frozen-controls.json").read_text())["E2"]["run_id"] == FROZEN_RUNS["public"]["E2"]
    bad["commit"] = True
    with pytest.raises(ValueError, match="benchmark"):
        frozen_controls(evaluation, "public", tmp_path)
    assert len(calls) == 4
    bad["commit"], bad["query"] = False, True
    with pytest.raises(ValueError, match="queries"):
        frozen_controls(evaluation, "public", tmp_path)
