"""Verify the notebooks' research protocol without remote access or retraining."""
import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
COMMIT = "8b12db0dce41040a821f9792bea5b352342263ce"


def cells(path):
    return json.loads(path.read_text())["cells"]


def python_source(cell):
    """Return executable Python while ignoring supported IPython magic lines."""
    return "".join(line for line in cell["source"] if not line.lstrip().startswith("%"))


def test_preparation_download_is_quiet_and_waits_out_hub_quota(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.utils import are_progress_bars_disabled, enable_progress_bars
    from urllib3.response import HTTPResponse
    from pettrace.data import workflow

    job = workflow.TrainingPreparation(drive_root=tmp_path, work_dir=tmp_path / "work")
    monkeypatch.setattr("pettrace.data.screening_gpu.check_cuda_environment", lambda: None)
    monkeypatch.setattr(workflow, "stage_archive", lambda source, target: target)
    factories = []
    monkeypatch.setattr(huggingface_hub, "configure_http_backend",
                        lambda *, backend_factory: factories.append(backend_factory))

    def stop_at_download(**kwargs):
        assert are_progress_bars_disabled()
        raise RuntimeError("Download configured")

    monkeypatch.setattr(workflow, "load_training", stop_at_download)
    disabled = are_progress_bars_disabled()
    try:
        with pytest.raises(RuntimeError, match="Download configured"):
            job.prepare()
        assert len(factories) == 1
        with factories[0]() as session:
            retry = session.get_adapter("https://huggingface.co").max_retries
            assert retry.get_retry_after(HTTPResponse(status=429, headers={"Retry-After": "370"})) == 370
            assert not retry.is_retry("GET", 401)
            waits = []
            for _ in range(3):
                retry = retry.increment(method="GET", response=HTTPResponse(status=429))
                waits.append(retry.get_backoff_time())
            assert waits == [0, 120, 240]
    finally:
        if not disabled:
            enable_progress_bars()


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
    assert '"pettrace-ml[training,notebooks]==0.2.26"' in install
    assert "requirements-e3-runtime" not in notebook
    assert "e3-runtime.txt" not in notebook
    assert "drive_root" not in notebook
    assert "SOURCE_SAMPLE_PCT, SEEDS = 100, (45, 46, 47, 48, 49)" in notebook


@pytest.mark.parametrize("name", ["e1_clip_zero_shot.ipynb", "e2_yolo_crop_zero_shot.ipynb"])
def test_current_zero_shot_notebooks_install_the_published_package_directly(name):
    contents = cells(ROOT / "notebooks" / name)
    install = next("".join(c["source"]) for c in contents if c["id"].endswith("setup"))
    notebook = "\n".join("".join(c["source"]) for c in contents)
    assert '"pettrace-ml[notebooks]==0.2.17"' in install
    assert "requirements-e1-runtime" not in notebook


def test_e4_installs_complete_published_training_package():
    contents = cells(ROOT / "notebooks/e4_clip_lora.ipynb")
    install = next("".join(c["source"]) for c in contents if c["id"] == "e4-setup")
    notebook = "\n".join("".join(c["source"]) for c in contents)
    assert '"pettrace-ml[training,notebooks]==0.2.25"' in install
    assert "requirements" not in install
    assert "LOG_TO_MLFLOW = True" in notebook
    assert "training(version=TRAINING_DATASET_VERSION" in notebook
    assert "drive_root" not in notebook


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
    assert ast.literal_eval(commit.value) == COMMIT
    publication = "\n".join("".join(c["source"]) for c in contents if c["cell_type"] == "code")
    if path.name.startswith(("e3_", "e4_")):
        assert "publication = publish_results(" in publication
        helper = (ROOT / "src/pettrace/experiments/e3.py").read_text()
        assert helper.index('"experiment-design.json"') < helper.index("public_run_id = publish_evaluation(")
    else:
        assert "publication = publish_results(" in publication
        helper = (ROOT / "src/pettrace/experiments/zero_shot.py").read_text()
        assert helper.index('"experiment-design.json"') < helper.index("public_run_id = publish_evaluation(")


@pytest.mark.parametrize("name", ["e1_clip_zero_shot.ipynb", "e2_yolo_crop_zero_shot.ipynb"])
def test_mutated_tag_stops_before_any_download(name):
    code = next("".join(c["source"]) for c in cells(ROOT / "notebooks" / name)
                if c["cell_type"] == "code" and "resolved_commit =" in "".join(c["source"]))
    downloads = []
    namespace = {"HfApi": lambda: SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(sha="changed")),
                 "data": SimpleNamespace(REPO="test/benchmark"), "DATASET": "v4", "DATASET_COMMIT": COMMIT,
                 "snapshot_download": lambda *a, **k: downloads.append(k)}
    with pytest.raises(AssertionError, match="pinned"):
        exec(code, namespace)
    assert downloads == []


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
