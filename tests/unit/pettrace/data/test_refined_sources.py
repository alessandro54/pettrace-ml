import csv
import gzip
import json
from types import SimpleNamespace

import pytest

from pettrace import data
from pettrace.data import refined_sources
from pettrace.data.verification import sha256
from pettrace.data.volume import training_volume


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


def test_training_combines_pinned_v2_sources_with_v4_100_equivalence(monkeypatch, tmp_path):
    sources = {}
    for name, species, digest in (("lcw", "cat", "a" * 64), ("dogs-world", "dog", "b" * 64)):
        screening = tmp_path / f"{name}.csv.gz"
        with gzip.open(screening, "wt", newline="") as stream:
            writer = csv.DictWriter(stream, [
                "sha256", "identity_id", "species", "source", "split", "score", "class_id",
                "x1", "y1", "x2", "y2",
            ])
            writer.writeheader()
            writer.writerow({"sha256": digest, "identity_id": f"{name}:1", "species": species,
                             "source": name, "split": "train", "score": 0.9, "class_id": 1,
                             "x1": 1, "y1": 2, "x2": 10, "y2": 11})
        spec = refined_sources.RELEASES[name]
        sources[name] = SimpleNamespace(
            source=name, species=species, repo_id=spec["repo_id"], version="v2",
            commit=spec["v2_commit"], root=tmp_path,
            base_manifest={"source": name, "prepared_version": "v4.100",
                           "prepared_fingerprint": "frozen", "photos": 1},
            manifest={"photos": 1, "localization": {"detector_sha256": "d" * 64}},
            localization_path=screening,
        )

    monkeypatch.setattr(refined_sources, "load_refined_source",
                        lambda name, **kwargs: sources[name])

    def extracted(root, evidence, destination):
        name = evidence["source"]
        species, digest = ("cat", "a" * 64) if name == "lcw" else ("dog", "b" * 64)
        row = {"path": str(tmp_path / f"{digest}.jpg"), "source": name, "species": species,
               "animal": f"{name}:1", "sha256": digest, "split": "train"}
        label = {"identity_id": row["animal"], "species": species, "source": name, "split": "train"}
        return [row], {digest: label}

    monkeypatch.setattr(refined_sources, "_extract_rows", extracted)

    loaded = data.training(cache_dir=tmp_path / "cache")

    assert loaded.fingerprint == "frozen"
    assert loaded.summary == {"retained_photos": 2, "retained_identities": 2}
    assert loaded.rows[0]["box"] == [1.0, 2.0, 10.0, 11.0]
    assert loaded.provenance["prepared_version"] == "v4.100"

    metrics, params = training_volume(loaded, 100, tmp_path)
    assert metrics["dataset.retained_photos"] == 2
    assert metrics["dataset.sampled_pct"] == 100.0
    assert params["dataset.transport"] == "huggingface_refined_v2"
    assert json.loads((tmp_path / "sources.json").read_text())["prepared_fingerprint"] == "frozen"
