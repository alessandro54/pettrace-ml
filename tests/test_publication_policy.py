"""Experiment entry points cannot silently omit required MLflow publication."""
import ast
import inspect
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("filename", ["e1_clip_zero_shot.ipynb", "e2_yolo_crop_zero_shot.ipynb",
                                      "e3_mlp_head_triplet.ipynb",
                                      "e4_clip_lora.ipynb"])
def test_notebook_publication_is_required_before_side_effects(filename):
    notebook = json.loads((Path(__file__).parents[1] / "notebooks" / filename).read_text())
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    config = next(s for s in code if "LOG_TO_MLFLOW =" in s)
    assignment = next(n for n in ast.parse(config).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "LOG_TO_MLFLOW" for t in n.targets))
    assert ast.literal_eval(assignment.value) is True
    if filename.startswith(("e3_", "e4_")):
        from pettrace.experiments.e3 import publish_results
    else:
        from pettrace.experiments.zero_shot import publish_results

    publication = next(s for s in code if "publication = publish_results(" in s)
    if not filename.startswith(("e3_", "e4_")):
        with pytest.raises(ValueError, match="mandatory"):
            exec(publication, {"LOG_TO_MLFLOW": False})
    assert "log_to_mlflow" not in inspect.signature(publish_results).parameters
    calls = [node for node in ast.walk(ast.parse(inspect.getsource(publish_results)))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "publish_evaluation"]
    assert len(calls) == 2


def test_cli_rejects_tracking_opt_out_before_loading_models(monkeypatch):
    from pettrace.cli import main

    monkeypatch.setattr("sys.argv", ["pettrace-evaluate", "--dataset", "v4", "--no-mlflow"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_training_runner_rejects_tracking_opt_out_before_work(tmp_path):
    from pettrace.experiments.legacy import run

    with pytest.raises(ValueError, match="mandatory"):
        run("nonexistent.csv", "unused", "v4", tmp_path / "run", log_mlflow=False)
    assert not (tmp_path / "run").exists()
