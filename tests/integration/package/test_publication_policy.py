"""Experiment entry points cannot silently omit required MLflow publication."""
import ast
import inspect
import json
import tomllib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests import ROOT


def test_mlflow_is_an_optional_package_extra():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert not any(requirement.startswith("mlflow") for requirement in config["project"]["dependencies"])
    assert config["project"]["optional-dependencies"]["tracking"] == ["mlflow-skinny==3.17.0"]


@pytest.mark.parametrize("filename", ["e1_clip_zero_shot.ipynb", "e2_yolo_crop_zero_shot.ipynb",
                                      "e3_mlp_head_triplet.ipynb",
                                      "e4_clip_lora.ipynb"])
def test_notebook_publication_is_required_before_side_effects(filename):
    notebook = json.loads((ROOT / "notebooks" / filename).read_text())
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    config = next(s for s in code if "LOG_TO_MLFLOW =" in s)
    assignment = next(n for n in ast.parse(config).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "LOG_TO_MLFLOW" for t in n.targets))
    assert ast.literal_eval(assignment.value) is True
    if filename in {"e1_clip_zero_shot.ipynb", "e2_yolo_crop_zero_shot.ipynb"}:
        publication = next(s for s in code if "publication = run(REPORT)" in s)
        assert "LOG_TO_MLFLOW" in publication
        return
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


def test_cli_rejects_tracking_opt_out_before_loading_models(tmp_path, monkeypatch):
    from pettrace.cli import main

    monkeypatch.setattr("sys.argv", ["pettrace", "evaluate", "--model", str(tmp_path / "model"),
                                     "--dataset", "v4", "--no-mlflow"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_cli_evaluates_without_publishing_when_tracking_is_not_configured(tmp_path, monkeypatch):
    from pettrace import cli

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr("sys.argv", ["pettrace", "evaluate", "--model", str(tmp_path / "model"),
                                     "--dataset", str(tmp_path)])
    monkeypatch.setattr(cli.data, "from_dir", lambda _: "dataset")
    monkeypatch.setattr(cli, "load_model", lambda _: SimpleNamespace(golden_ok=True, pipeline="pipeline"))
    monkeypatch.setattr(cli, "evaluate", lambda *args, **kwargs: SimpleNamespace(result={}))
    monkeypatch.setattr(cli, "report", lambda _: "local evaluation")
    publish = Mock()
    monkeypatch.setattr(cli, "log_run", publish)

    assert cli.main() == 0
    publish.assert_not_called()


def test_cli_rejects_untracked_experiment_before_loading_models(tmp_path, monkeypatch):
    from pettrace import cli

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr("sys.argv", ["pettrace", "evaluate", "--model", str(tmp_path / "model"),
                                     "--dataset", str(tmp_path), "--experiment", "E5"])
    load = Mock()
    monkeypatch.setattr(cli, "load_model", load)

    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    load.assert_not_called()


