"""Verify the notebooks' research protocol without remote access or retraining."""
import ast
import json
import re
from types import SimpleNamespace

import pytest

from tests import ROOT

NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
PRIVATE_COMMITS = {
    "e4_clip_lora.ipynb": "52bfe26384c4230008fe4707aa0cb145988766a8",
    "e5_language_reranking.ipynb": "52bfe26384c4230008fe4707aa0cb145988766a8",
}
DEFAULT_PRIVATE_COMMIT = "1777734dbffda0df0bf43305233809ee73157d94"


def cells(path):
    return json.loads(path.read_text())["cells"]


def python_source(cell):
    """Return executable Python while ignoring supported IPython magic lines."""
    return "".join(line for line in cell["source"] if not line.lstrip().startswith("%"))


@pytest.mark.parametrize("path", [p for p in NOTEBOOKS if not p.name.startswith("e1_")], ids=lambda p: p.stem)
@pytest.mark.parametrize("version", [(3, 10), (3, 11), (3, 12), (3, 13), (3, 14)])
def test_notebook_python_support_matches_dependency_profile(path, version):
    source = next("".join(c["source"]) for c in cells(path)
                  if c["cell_type"] == "code" and "sys.version_info" in "".join(c["source"]))
    source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
    guard = next(n for n in ast.parse(source).body
                 if isinstance(n, ast.If) and "version_info" in ast.unparse(n.test))
    code = compile(ast.Module(body=[guard], type_ignores=[]), str(path), "exec")
    namespace = {"sys": SimpleNamespace(version_info=version, version=".".join(map(str, version)))}
    supported = (3, 11) <= version < (3, 14)
    namespace["sys"].platform = "linux"
    if supported:
        exec(code, namespace)
    else:
        with pytest.raises(RuntimeError, match="Python"):
            exec(code, namespace)


def test_current_e3_notebook_installs_the_published_package_directly():
    contents = cells(ROOT / "notebooks/e3_mlp_head_triplet.ipynb")
    install = next("".join(c["source"]) for c in contents if c["id"] == "e3-setup")
    notebook = "\n".join("".join(c["source"]) for c in contents)
    assert '"pettrace-ml[training,notebooks,tracking]==0.2.37"' in install
    assert "requirements-e3-runtime" not in notebook
    assert "e3-runtime.txt" not in notebook
    assert "drive_root" not in notebook
    assert "SOURCE_SAMPLE_PCT, SEEDS = 100, (45, 46, 47, 48, 49)" in notebook


def test_e1_notebook_installs_the_complete_headless_toolkit():
    contents = cells(ROOT / "notebooks/e1_clip_zero_shot.ipynb")
    install = next("".join(c["source"]) for c in contents if c["id"].endswith("setup"))
    notebook = "\n".join("".join(c["source"]) for c in contents)
    assert '"pettrace-ml[notebooks,tracking]==0.2.33"' in install
    assert "import mlflow" in install
    assert "requirements-e1-runtime" not in notebook
    assert "publication = run(REPORT)" in notebook
    assert "snapshot_download" not in notebook
    assert "publish_results" not in notebook


def test_e2_notebook_installs_the_published_package_directly():
    contents = cells(ROOT / "notebooks/e2_yolo_crop_zero_shot.ipynb")
    install = next("".join(c["source"]) for c in contents if c["id"].endswith("setup"))
    notebook = "\n".join("".join(c["source"]) for c in contents)
    assert '"pettrace-ml[notebooks,tracking]==0.2.35"' in install
    assert "publication = run(REPORT)" in notebook
    assert "snapshot_download" not in notebook
    assert "publish_results" not in notebook


def test_e4_installs_complete_published_training_package():
    contents = cells(ROOT / "notebooks/e4_clip_lora.ipynb")
    install = next("".join(c["source"]) for c in contents if c["id"] == "e4-setup")
    notebook = "\n".join("".join(c["source"]) for c in contents)
    assert '"pettrace-ml[training,notebooks,tracking]==0.2.39"' in install
    assert "requirements" not in install
    assert "LOG_TO_MLFLOW = True" in notebook
    assert "training(version=TRAINING_DATASET_VERSION" in notebook
    assert "drive_root" not in notebook
    assert 'TRAINING_DATASET_VERSION, SOURCE_SAMPLE_PCT, SEED = "v2", 100, 46' in notebook


def test_e5_installs_the_vlm_profile_and_freezes_before_test():
    contents = cells(ROOT / "notebooks/e5_language_reranking.ipynb")
    install = next("".join(c["source"]) for c in contents if c["id"] == "e5-setup")
    calibration_index = next(i for i, c in enumerate(contents) if c["id"] == "e5-calibrate")
    evaluation_index = next(i for i, c in enumerate(contents) if c["id"] == "e5-evaluate")
    calibration = "".join(contents[calibration_index]["source"])
    evaluation = "".join(contents[evaluation_index]["source"])
    notebook = "\n".join("".join(c["source"]) for c in contents)

    assert '"pettrace-ml[vlm,notebooks,tracking]==0.2.41"' in install
    assert calibration_index < evaluation_index
    assert "select_calibration(" in calibration
    assert 'public_split_dataset(prepared, public_source, "val")' in notebook
    assert "load_benchmarks(" in notebook
    assert "evaluate(benchmarks.public" in evaluation
    assert "PetTrace v6" in notebook
    assert "publication = publish_results(" in notebook


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_experiment_notebooks_do_not_use_google_drive(path):
    source = "\n".join("".join(cell["source"]) for cell in cells(path))
    assert "DriveBackup" not in source
    assert "/content/drive" not in source
    assert "google.colab import drive" not in source


def test_e4_loads_colab_secrets_before_dataset_access():
    contents = cells(ROOT / "notebooks/e4_clip_lora.ipynb")
    config = next("".join(c["source"]) for c in contents if c["id"] == "e4-config")
    load_index = next(i for i, c in enumerate(contents) if c["id"] == "e4-load-data")
    config_index = next(i for i, c in enumerate(contents) if c["id"] == "e4-config")
    assert config_index < load_index
    assert "from google.colab import userdata" in config
    assert 'userdata.get(name)' in config
    assert "isinstance(value, str)" in config
    assert "configure_access" not in config


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_shared_research_structure_and_fixed_benchmark(path):
    contents = cells(path)
    headings = [line for c in contents if c["cell_type"] == "markdown"
                for line in "".join(c["source"]).splitlines() if line.startswith("## ")]
    assert [int(re.match(r"## Fase (\d+)\.", h).group(1)) for h in headings] == [1, 2, 3]
    assignments = [node for c in contents if c["cell_type"] == "code"
                   and "DATASET_COMMIT =" in "".join(c["source"])
                   for node in ast.parse("".join(c["source"])).body if isinstance(node, ast.Assign)]
    commit = next(n for n in assignments if any(isinstance(t, ast.Name) and t.id == "DATASET_COMMIT"
                                                for t in n.targets))
    expected_commit = PRIVATE_COMMITS.get(path.name, DEFAULT_PRIVATE_COMMIT)
    assert ast.literal_eval(commit.value) == expected_commit
    publication = "\n".join("".join(c["source"]) for c in contents if c["cell_type"] == "code")
    if path.name in {"e1_clip_zero_shot.ipynb", "e2_yolo_crop_zero_shot.ipynb"}:
        assert "publication = run(REPORT)" in publication
        experiment = path.stem.split("_")[0]
        helper = (ROOT / f"src/pettrace/experiments/{experiment}.py").read_text()
        assert helper.index("_dataset_selection(") < helper.index("publication = publish_results(")
    elif path.name.startswith(("e3_", "e4_")):
        assert "publication = publish_results(" in publication
        helper = (ROOT / "src/pettrace/experiments/e3.py").read_text()
        assert helper.index('"experiment-design.json"') < helper.index("public_run_id = publish_evaluation(")
    elif path.name.startswith("e5_"):
        assert "publication = publish_results(" in publication
        helper = (ROOT / "src/pettrace/experiments/e5.py").read_text()
        assert helper.index('"experiment-design.json"') < helper.index("public_run_id = publish_evaluation(")
    else:
        assert "publication = publish_results(" in publication
        helper = (ROOT / "src/pettrace/experiments/zero_shot.py").read_text()
        assert helper.index('"experiment-design.json"') < helper.index("public_run_id = publish_evaluation(")

def test_e3_selects_by_validation_before_test_and_publishes_every_seed():
    contents = cells(ROOT / "notebooks/e3_mlp_head_triplet.ipynb")
    train_index = next(i for i, c in enumerate(contents) if c.get("id") == "e3-train")
    evaluation_index = next(i for i, c in enumerate(contents) if c.get("id") == "e3-evaluate")
    train = "".join(contents[train_index]["source"])
    evaluation = "".join(contents[evaluation_index]["source"])
    registration = next("".join(c["source"]) for c in contents if c.get("id") == "e3-register")

    assert train_index < evaluation_index
    assert "selected = select_validation_candidate(candidates)" in train
    assert "for candidate in candidates:" in evaluation
    assert "publication = publish_results(" in evaluation
    assert 'selected["publication"]' in registration
