"""Verify the notebooks' research protocol without remote access or retraining."""
import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    monkeypatch.setattr(workflow, "configure_access", lambda **kw: None)
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


@pytest.mark.parametrize("path", [p for p in NOTEBOOKS if not p.name.startswith("e1_")]
                         + sorted((ROOT / "datasets").glob("prepare_training*.ipynb")), ids=lambda p: p.stem)
@pytest.mark.parametrize("version", [(3, 10), (3, 11), (3, 12), (3, 13), (3, 14)])
def test_notebook_python_support_matches_dependency_profile(path, version):
    source = next("".join(c["source"]) for c in cells(path)
                  if c["cell_type"] == "code" and "sys.version_info" in "".join(c["source"]))
    source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
    guard = next(n for n in ast.parse(source).body
                 if isinstance(n, ast.If) and "version_info" in ast.unparse(n.test))
    code = compile(ast.Module(body=[guard], type_ignores=[]), str(path), "exec")
    namespace = {"sys": SimpleNamespace(version_info=version, version=".".join(map(str, version)))}
    supported = version == (3, 13) if "E3 Colab profile" in source else (3, 11) <= version < (3, 14)
    namespace["sys"].platform = "linux"
    if supported:
        exec(code, namespace)
    else:
        with pytest.raises(RuntimeError, match="Python"):
            exec(code, namespace)


@pytest.mark.parametrize("percent", [20, 50, 100])
def test_current_e3_notebooks_install_the_published_package_directly(percent):
    contents = cells(ROOT / f"notebooks/e3_mlp_head_triplet_{percent}.ipynb")
    install = next("".join(c["source"]) for c in contents if c["id"] == "e3-02")
    notebook = "\n".join("".join(c["source"]) for c in contents)
    assert '"pettrace-ml[training,notebooks]==0.2.16"' in install
    assert "requirements-e3-runtime" not in notebook
    assert "e3-runtime.txt" not in notebook


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
    assert '"pettrace-ml[training,notebooks]==0.2.20"' in install
    assert "requirements" not in install
    assert "LOG_TO_MLFLOW = True" in notebook


def test_e4_loads_colab_secrets_before_dataset_access():
    contents = cells(ROOT / "notebooks/e4_clip_lora.ipynb")
    config = next("".join(c["source"]) for c in contents if c["id"] == "e4-config")
    load_index = next(i for i, c in enumerate(contents) if c["id"] == "e4-load-data")
    config_index = next(i for i, c in enumerate(contents) if c["id"] == "e4-config")
    assert config_index < load_index
    assert "configure_access" in config
    assert "HF_TOKEN" in config


def test_current_preparation_notebook_installs_the_published_package_directly():
    contents = cells(ROOT / "datasets/prepare_training.ipynb")
    install = next("".join(c["source"]) for c in contents if c["id"] == "training-data-dependencies")
    notebook = "\n".join("".join(c["source"]) for c in contents)
    assert '"pettrace-ml[training,notebooks]==0.2.16"' in install
    assert "e3-runtime.txt" not in notebook


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


def test_e3_identity_macro_does_not_weight_by_photo_count(tmp_path):
    code = next("".join(c["source"]) for c in cells(ROOT / "notebooks/e3_mlp_head_triplet_20.ipynb")
                if c["id"] == "e3-identity-findings")

    def evaluation(ranks):
        rows = [{"animal": animal, "rank": rank, "ap": 1.0 / rank}
                for animal, rank in zip(["a", "a", "b"], ranks)]
        return SimpleNamespace(result={"test": {"per_query": rows}})

    namespace = {"np": np, "json": json, "REPORT": tmp_path,
                 "baseline_evaluations": {"E1": evaluation([1, 2, 1]), "E2": evaluation([2, 2, 1])},
                 "ev": evaluation([1, 1, 2]), "ds": SimpleNamespace(name="benchmark", commit=COMMIT,
                      fingerprint="digest", animals={"a": {"species": "cat"}, "b": {"species": "dog"}}),
                 "PROTOCOL": "zero-shot", "comparison": {}, "train_summary": {},
                 "prepared": SimpleNamespace(fingerprint="public"), "coverage_metrics": {},
                 "display": lambda *a: None, "Markdown": lambda s: s}
    exec(code, namespace)
    analysis = json.loads((tmp_path / "analysis.json").read_text())
    assert analysis["macro_rank1"]["E3"] == 0.5
    assert analysis["per_animal"][0]["rank1_delta_e3_e2"] == 1.0
    assert analysis["per_animal"][1]["rank1_delta_e3_e2"] == -1.0


def test_single_preparation_notebook_delegates_to_package():
    path = ROOT / "datasets/prepare_training.ipynb"
    assert list(path.parent.glob("prepare_training*.ipynb")) == [path]
    contents = cells(path)
    for cell in contents:
        if cell['cell_type'] == 'code':
            ast.parse(python_source(cell))
    source = next(''.join(c['source']) for c in contents if 'prepared = preparation.prepare(' in ''.join(c['source']))
    prepared = SimpleNamespace(fingerprint='verified')
    workflow = SimpleNamespace(prepare=lambda **kw: prepared, version_path='v2.50.json')
    namespace = {'preparation': workflow, 'perf_counter': lambda: 0, 'os': SimpleNamespace(environ={})}
    exec(source, namespace)
    assert namespace['prepared'] is prepared


def test_preparation_notebook_exports_hf_version_setting(capsys):
    source = next(''.join(c['source']) for c in cells(ROOT / "datasets/prepare_training.ipynb")
                  if c['id'] == 'training-data-use')
    namespace = {'json': json, 'preparation': SimpleNamespace(version_path='v2.50.json', version='v2.50'),
                 'prepared': SimpleNamespace(fingerprint='verified'),
                 'release': {'repo': 'fixture/training', 'version': 'v2.50', 'commit': 'a' * 40,
                             'fingerprint': 'verified'}}
    exec(source, namespace)
    output = capsys.readouterr().out
    settings = json.loads(output[output.index('{'):output.rindex('}') + 1])
    assert settings['HF_TRAINING_DATASET_VERSION'] == 'v2.50'


@pytest.mark.parametrize("percent", [20, 50, 100])
@pytest.mark.parametrize("case, error", [
    ("missing", "not ready"), ("other_volume", "include v4"),
    ("invalid_hash", "Invalid prepared dataset fingerprint"),
    ("missing_index", "candidate index checksum"), ("changed_data", "fingerprint differs"),
    ("changed_index", "different automatic candidate index"), ("ready", None),
])
def test_e3_requires_completed_volume_receipt(tmp_path, monkeypatch, percent, case, error):
    import pettrace.data

    source = next(''.join(c['source']) for c in cells(ROOT / f'notebooks/e3_mlp_head_triplet_{percent}.ipynb')
                  if c.get('id') == 'e3-load-prepared').split('manifest_rows =')[0]
    drive = tmp_path / 'content/drive/MyDrive'
    receipt = drive / 'PetTrace/data-lookup/inventory100/volume-series.json'
    receipt.parent.mkdir(parents=True)
    version = f'v4.{percent}'
    entry = {'version': version, 'fingerprint': 'a' * 64, 'candidate_index_sha256': 'b' * 64}
    if case == 'invalid_hash':
        entry['fingerprint'] = 'unfinished'
    if case == 'missing_index':
        del entry['candidate_index_sha256']
    if case != 'missing':
        other_percent = 100 if percent != 100 else 50
        receipt.write_text(json.dumps({str(other_percent if case == 'other_volume' else percent): entry}))
    prepared_root = tmp_path / 'refined'
    prepared_root.mkdir()
    (prepared_root / 'prepared.json').write_text(json.dumps({
        'inputs': {'candidate_index_sha256': ('c' if case == 'changed_index' else 'b') * 64}}))
    calls = []

    def load(version, **kwargs):
        calls.append((version, kwargs))
        return SimpleNamespace(root=prepared_root,
                               fingerprint=('d' if case == 'changed_data' else 'a') * 64), {}

    monkeypatch.setattr(pettrace.data, 'load', load)
    report = tmp_path / 'report'
    report.mkdir()
    namespace = {'Path': lambda p: tmp_path / str(p).lstrip('/'), 'json': json, 'REPORT': report,
                 'SOURCE_SAMPLE_PCT': percent, 'TRAINING_DATASET_VERSION': version}
    if error:
        with pytest.raises((RuntimeError, ValueError), match=error):
            exec(source, namespace)
        assert not (report / 'volume-series.json').exists()
    else:
        exec(source, namespace)
        assert (report / 'volume-series.json').read_bytes() == receipt.read_bytes()
        assert namespace['TRAINING_DATASET_FINGERPRINT'] == 'a' * 64
    if case in {'missing', 'other_volume', 'invalid_hash', 'missing_index'}:
        assert calls == []
    else:
        assert calls == [(version, {'cache_dir': '/content/datasets/build/pettrace-training',
                                    'drive_root': '/content/drive/MyDrive/PetTrace'})]


@pytest.mark.parametrize('outcome', ['prepared', 'failed', 'preparing'])
def test_e3_100_waits_for_active_preparation_without_starting_it(tmp_path, monkeypatch, outcome):
    import time

    source = next(''.join(c['source']) for c in cells(ROOT / 'notebooks/e3_mlp_head_triplet_100.ipynb')
                  if 'so preparation keeps the available RAM' in ''.join(c['source']))
    source = source.split('if globals().get("_E3_RESTART_REQUIRED"):')[0]
    assert 'import torch' not in source and 'import pettrace' not in source
    source = source.replace('"/content/', f'"{tmp_path}/content/').replace('"/proc/', f'"{tmp_path}/proc/')
    drive = tmp_path / 'content/drive/MyDrive/PetTrace/data-lookup/inventory100'
    drive.mkdir(parents=True)
    state = drive / 'preparation-status.json'
    state.write_text(json.dumps({'state': 'preparing'}))
    (tmp_path / 'content/volume-preparation.pid').write_text('123')
    process = tmp_path / 'proc/123/cmdline'
    process.parent.mkdir(parents=True)
    process.write_bytes(b'python\0continue_volume_preparation.py\0')
    waits = []

    def finish(seconds):
        waits.append(seconds)
        process.unlink()
        state.write_text(json.dumps({'state': outcome}))

    monkeypatch.setattr(time, 'sleep', finish)
    namespace = {}
    if outcome == 'prepared':
        exec(source, namespace)
    else:
        with pytest.raises(RuntimeError, match='without a successful receipt'):
            exec(source, namespace)
    assert waits == [30]


def test_e3_100_keeps_newer_local_feature_cache(tmp_path):
    source = next(''.join(c['source']) for c in cells(ROOT / 'notebooks/e3_mlp_head_triplet_100.ipynb')
                  if 'missing_cache_files =' in ''.join(c['source']))
    source = 'missing_cache_files =' + source.split('missing_cache_files =', 1)[1].split('with feature_backup:')[0]
    (tmp_path / 'current.sqlite3').write_bytes(b'newer local embeddings')
    restored = []
    backup = SimpleNamespace(inventory=lambda: {'features/current.sqlite3': {}, 'features/missing.sqlite3': {}},
                             restore=lambda **kwargs: restored.append(kwargs))
    exec(source, {'FEATURE_CACHE_ROOT': tmp_path, 'feature_backup': backup})
    assert restored == [{'files': {'features/missing.sqlite3'}}]
    assert (tmp_path / 'current.sqlite3').read_bytes() == b'newer local embeddings'
