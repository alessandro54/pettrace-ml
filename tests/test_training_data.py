import fnmatch
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pettrace.data.preparation import prepare
from pettrace.data.versions import load_local_training, load_training, publish_training, record_version
from test_e3_prepare import setup_pipeline


class Hub:
    def __init__(self, root):
        self.root, self.private, self.tags = root, True, {}
        self.root.mkdir()
        self.commit = "a" * 40
        self.uploads = []
        self.downloads = []

    def create_repo(self, repo_id, **kwargs):
        assert kwargs == {"repo_type": "dataset", "private": True, "exist_ok": True}

    def dataset_info(self, repo_id, revision=None):
        return SimpleNamespace(private=self.private, sha=self.tags.get(revision, self.commit))

    def list_repo_refs(self, repo_id, **kwargs):
        return SimpleNamespace(tags=[SimpleNamespace(name=k, target_commit=v) for k, v in self.tags.items()],
                               branches=[SimpleNamespace(name="main", target_commit=self.commit)])

    def file_exists(self, repo_id, filename, **kwargs):
        return (self.root / filename).exists()

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)
        for path in Path(kwargs["folder_path"]).rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(kwargs["folder_path"]).as_posix()
            if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in kwargs["allow_patterns"]):
                continue
            target = self.root / kwargs["path_in_repo"] / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        self.commit = "b" * 40
        return SimpleNamespace(oid=self.commit)

    def upload_file(self, **kwargs):
        assert kwargs["parent_commit"] == self.commit
        (self.root / kwargs["path_in_repo"]).write_bytes(kwargs["path_or_fileobj"])
        self.commit = "c" * 40
        return SimpleNamespace(oid=self.commit)

    def list_repo_files(self, repo_id, **kwargs):
        return [p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()]

    def create_tag(self, repo_id, **kwargs):
        assert not kwargs["exist_ok"] and kwargs["tag"] not in self.tags
        self.tags[kwargs["tag"]] = kwargs["revision"]

    def download_file(self, repo_id, *, filename, revision, **kwargs):
        return str(self.root / filename)

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
    monkeypatch.setattr("pettrace.data.versions.hf_hub_download", hub.download_file)
    monkeypatch.setattr("pettrace.data.versions.snapshot_download", hub.download_snapshot)
    return hub


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


def test_private_hub_release_is_idempotent_inventory_only_and_loads_by_commit(tmp_path, monkeypatch):
    config = setup_pipeline(tmp_path, monkeypatch)
    dataset = prepare(config)
    (dataset.root / ".env").write_text("DO_NOT_UPLOAD=test fixture")
    hub = install_hub(tmp_path, monkeypatch)
    release = publish_training(dataset.root)
    assert release["commit"] == "c" * 40 and release["fingerprint"] == dataset.fingerprint
    assert not (hub.root / "versions/v1/.env").exists()
    assert "pettrace-reid" in (hub.root / "README.md").read_text()
    assert publish_training(dataset.root) == release and len(hub.uploads) == 1
    loaded, source = load_training(expected_fingerprint=dataset.fingerprint, cache_dir=tmp_path / "cache")
    assert loaded.fingerprint == dataset.fingerprint and source == release
    assert all(Path(row["path"]).is_relative_to(tmp_path / "cache") for row in loaded.rows)
    with pytest.raises(ValueError, match="fingerprint"):
        load_training(expected_fingerprint="0" * 64, cache_dir=tmp_path / "cache")
    # Exercise the experiment's real Hub input branch, without credentials or fitting a model.
    from pettrace.data.curation import sha256
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/e3_mlp_head_triplet.ipynb").read_text())
    cell = next(c for c in notebook["cells"] if c["id"] == "e3-load-prepared")
    report = tmp_path / "report"
    report.mkdir()
    accesses = []
    monkeypatch.setenv("PETTRACE_TRAINING_CACHE", str(tmp_path / "experiment-cache"))
    namespace = {"TRAINING_DATASET_REPO": release["repo"], "TRAINING_DATASET_VERSION": "v1",
                 "TRAINING_DATASET_REVISION": release["commit"], "TRAINING_DATASET_FINGERPRINT": dataset.fingerprint,
                 "configure_private_access": lambda **kwargs: accesses.append(kwargs), "REPORT": report,
                 "json": json, "shutil": shutil, "sha256": sha256}
    exec(compile("".join(cell["source"]), "hub-experiment-input", "exec"), namespace)
    assert accesses == [{"names": ("HF_TOKEN",)}]
    assert json.loads((report / "data-preparation.json").read_text())["training_source"] == release
    # A moved external tag cannot change the commit between resolution and download.
    def resolve_and_move(repo, revision=None):
        result = SimpleNamespace(private=True, sha="c" * 40)
        hub.tags["v1"] = "d" * 40
        return result
    monkeypatch.setattr(hub, "dataset_info", resolve_and_move)
    assert load_training(cache_dir=tmp_path / "cache")[1]["commit"] == "c" * 40


def test_public_repository_benchmark_and_release_overwrite_are_rejected(tmp_path, monkeypatch):
    config = setup_pipeline(tmp_path, monkeypatch)
    dataset = prepare(config)
    hub = install_hub(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="benchmark"):
        publish_training(dataset.root, repo_id="alessandro54/pettrace-reid")
    with pytest.raises(ValueError, match="version"):
        publish_training(dataset.root, version="../v1")
    hub.private = False
    with pytest.raises(ValueError, match="private"):
        publish_training(dataset.root)
    assert not hub.uploads
    hub.private = True
    publish_training(dataset.root)
    remote = hub.root / "versions/v1/prepared.json"
    content = json.loads(remote.read_text())
    content["fingerprint"] = "0" * 64
    remote.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="another dataset"):
        publish_training(dataset.root)
    assert len(hub.uploads) == 1


def test_notebook_defaults_do_not_publish_or_prompt_for_hf_access():
    notebook = json.loads((Path(__file__).parents[1] / "datasets/prepare_training.ipynb").read_text())
    cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    assert "PUBLISH_TO_HF = False" in cells["training-data-config"]
    calls = []
    namespace = {"PUBLISH_TO_HF": False, "configure_access": calls.append, "publish_training": calls.append}
    exec(compile(cells["training-data-publish"], "publication-default", "exec"), namespace)
    assert namespace["release"] is None and not calls


def test_incomplete_hub_upload_does_not_create_a_version_tag(tmp_path, monkeypatch):
    config = setup_pipeline(tmp_path, monkeypatch)
    dataset = prepare(config)
    hub = install_hub(tmp_path, monkeypatch)
    real_inventory = hub.list_repo_files
    monkeypatch.setattr(hub, "list_repo_files", lambda *args, **kwargs: ["README.md"])
    with pytest.raises(ValueError, match="incomplete"):
        publish_training(dataset.root)
    assert not hub.tags
    monkeypatch.setattr(hub, "list_repo_files", real_inventory)
    assert publish_training(dataset.root)["version"] == "v1"
    assert "v1" in hub.tags
