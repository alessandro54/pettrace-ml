from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from pettrace.experiments import e5


def test_e5_loads_only_the_frozen_e4_winner(tmp_path, monkeypatch):
    model = SimpleNamespace(golden_ok=True, registry_version="",
                            model_version=e5.CONTROL_MODEL_LABEL, sha256=e5.CONTROL_MODEL_SHA256)
    monkeypatch.setattr(e5, "load_model", lambda uri: model)

    loaded, receipt = e5.load_frozen_e4("/models/e4", tmp_path)
    assert loaded is model
    assert receipt["selected_epoch"] == 27
    assert receipt["model_uri"] == "/models/e4"
    assert receipt["inference"] == {"representation": "single", "identity_aggregation": "max"}
    assert (tmp_path / "e4-control.json").is_file()

    model.sha256 = {"encoder.onnx": "different"}
    with pytest.raises(ValueError, match="identity or golden"):
        e5.load_frozen_e4("/models/e4", tmp_path)


def test_reranking_audit_stays_outside_standard_test_metrics():
    @dataclass
    class Evaluation:
        result: dict

    visual = Evaluation(result={"test": {"global": {"rank1": 0.5}}, "val": {}})
    reranked = {"global": {"rank1": 0.75}, "per_query": [], "audit": {"rescued_queries": 1}}

    updated = e5.as_evaluation(visual, reranked)

    assert updated.result["test"] == {"global": {"rank1": 0.75}, "per_query": []}
    assert updated.result["reranking_audit"] == {"rescued_queries": 1}
    assert "audit" not in updated.result["test"]
