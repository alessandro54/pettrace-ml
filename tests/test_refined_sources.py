import csv
import json
from types import SimpleNamespace

import pytest

from pettrace import data
from pettrace.data import refined_sources
from pettrace.data.verification import sha256


def release(tmp_path, source="dogs-world"):
    root = tmp_path / "snapshot"
    version = root / "versions/v1"
    shard = version / "data/train/source-00000.tar"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"tar fixture")
    metadata = version / "metadata.csv"
    with metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["sha256", "identity_id", "species", "source", "split", "shard"])
        writer.writeheader()
        writer.writerow({"sha256": "a" * 64, "identity_id": "dog:1", "species": "dog",
                         "source": source, "split": "train",
                         "shard": "versions/v1/data/train/source-00000.tar"})
    manifest = {
        "source": source, "release_version": "v1",
        "file_sha256": {"versions/v1/metadata.csv": sha256(metadata)},
        "shards": [{"path": "versions/v1/data/train/source-00000.tar", "samples": 1}],
    }
    (version / "dataset.json").write_text(json.dumps(manifest))
    return root


def test_dogs_loader_resolves_and_downloads_audited_commit(monkeypatch, tmp_path):
    root = release(tmp_path)
    spec = refined_sources.RELEASES["dogs-world"]
    calls = []
    monkeypatch.setattr(refined_sources, "HfApi", lambda token=None: SimpleNamespace(
        dataset_info=lambda repo, revision: SimpleNamespace(sha=spec["commit"])))
    monkeypatch.setattr(refined_sources, "snapshot_download",
                        lambda repo, **kwargs: calls.append((repo, kwargs)) or root)

    loaded = data.dogs_of_the_world(cache_dir=tmp_path / "cache")

    assert loaded.repo_id == spec["repo_id"] and loaded.commit == spec["commit"]
    assert loaded.shards("train") == [root / "versions/v1/data/train/source-00000.tar"]
    assert next(loaded.metadata())["identity_id"] == "dog:1"
    assert calls == [(spec["repo_id"], {
        "repo_type": "dataset", "revision": spec["commit"], "token": None,
        "allow_patterns": ["versions/v1/**"], "cache_dir": str(tmp_path / "cache"),
    })]


def test_loader_rejects_a_moved_v1_tag(monkeypatch):
    monkeypatch.setattr(refined_sources, "HfApi", lambda token=None: SimpleNamespace(
        dataset_info=lambda repo, revision: SimpleNamespace(sha="f" * 40)))
    with pytest.raises(ValueError, match="audited commit"):
        data.labeled_cats_in_the_wild()
