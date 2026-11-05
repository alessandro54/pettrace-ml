import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pettrace.data.preparation import prepare
from pettrace.data.versions import load_local_training, load_training, record_version
from test_e3_prepare import setup_pipeline


class Hub:
    def __init__(self, root):
        self.root, self.tags = root, {}
        self.root.mkdir()
        self.commit = "c" * 40
        self.downloads = []

    def dataset_info(self, repo_id, revision=None):
        return SimpleNamespace(sha=self.tags.get(revision, self.commit))

    def download_snapshot(self, repo_id, **kwargs):
        self.downloads.append(kwargs)
        assert kwargs["revision"] == "c" * 40
        assert kwargs["allow_patterns"] == ["versions/v1/**"]
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.root / "versions/v1", target / "versions/v1", dirs_exist_ok=True)
        return str(target)


def install_hub(tmp_path, monkeypatch):
    hub = Hub(tmp_path / "hub")
    monkeypatch.setattr("pettrace.data.versions.HfApi", lambda: hub)
    monkeypatch.setattr("pettrace.data.versions.snapshot_download", hub.download_snapshot)
    return hub


def publish_fixture(dataset, hub):
    shutil.copytree(dataset.root, hub.root / "versions/v1")
    hub.tags["v1"] = hub.commit
    return {"repo": "alessandro54/pettrace-training", "version": "v1", "commit": hub.commit,
            "fingerprint": dataset.fingerprint, "path_in_repo": "versions/v1"}


def test_named_versions_are_stable_reusable_and_do_not_move_to_other_snapshots(tmp_path, monkeypatch):
    config = setup_pipeline(tmp_path, monkeypatch)
    dataset = prepare(config)
    alias = record_version(dataset.root, dataset.root.parent)
    assert record_version(dataset.root, dataset.root.parent) == alias
    loaded, source = load_local_training(alias)
    assert loaded.fingerprint == dataset.fingerprint and source["version"] == "v1"
    other = prepare(replace(config, seed=43))
    with pytest.raises(ValueError, match="another snapshot"):
        record_version(other.root, other.root.parent)
    assert load_local_training(alias)[0].fingerprint == dataset.fingerprint
    assert record_version(other.root, other.root.parent, version="v2").exists()
    content = json.loads(alias.read_text())
    content["path"] = "../outside"
    alias.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="Unsafe"):
        load_local_training(alias)


def test_hub_snapshot_loads_by_commit(tmp_path, monkeypatch):
    config = setup_pipeline(tmp_path, monkeypatch)
    dataset = prepare(config)
    hub = install_hub(tmp_path, monkeypatch)
    release = publish_fixture(dataset, hub)
    loaded, source = load_training(expected_fingerprint=dataset.fingerprint, cache_dir=tmp_path / "cache")
    assert loaded.fingerprint == dataset.fingerprint and source == release
    assert all(Path(row["path"]).is_relative_to(tmp_path / "cache") for row in loaded.rows)
    with pytest.raises(ValueError, match="fingerprint"):
        load_training(expected_fingerprint="0" * 64, cache_dir=tmp_path / "cache")
    # A moved external tag cannot change the commit between resolution and download.
    def resolve_and_move(repo, revision=None):
        result = SimpleNamespace(sha="c" * 40)
        hub.tags["v1"] = "d" * 40
        return result
    monkeypatch.setattr(hub, "dataset_info", resolve_and_move)
    assert load_training(cache_dir=tmp_path / "cache")[1]["commit"] == "c" * 40


def test_preparation_notebook_has_no_publication_controls():
    notebook = json.loads((Path(__file__).parents[1] / "datasets/prepare_training.ipynb").read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "PUBLISH_TO_HF" not in source and ".publish(" not in source


def test_pinned_local_snapshot_needs_no_hub_queries_and_still_checks_contents(tmp_path, monkeypatch):
    from pettrace.data.versions import REPO
    from pettrace.data.verification import verification_session
    dataset = prepare(setup_pipeline(tmp_path, monkeypatch))
    commit = 'c' * 40
    cache = tmp_path / 'cache'
    target = cache / REPO.replace('/', '--') / commit / 'versions/v1'
    shutil.copytree(dataset.root, target)
    monkeypatch.setattr('pettrace.data.versions.HfApi', lambda: pytest.fail('Pinned local reference must not query HF'))
    monkeypatch.setattr('pettrace.data.versions.snapshot_download', lambda *a, **kw: pytest.fail('No repeated download'))
    kwargs = dict(revision=commit, expected_fingerprint=dataset.fingerprint, cache_dir=cache)
    with verification_session() as index:
        loaded, receipt = load_training(**kwargs)
        reads = index.reads
        assert load_training(**kwargs)[0].fingerprint == dataset.fingerprint
        assert index.reads == reads and index.hits > 0
        assert receipt['commit'] == commit
        with pytest.raises(ValueError, match='fingerprint'):
            load_training(**{**kwargs, 'expected_fingerprint': '0' * 64})
        Path(loaded.rows[0]['path']).write_bytes(b'corrupt')
        with pytest.raises(ValueError, match='artifact changed'):
            load_training(**kwargs)
