"""Dataset downloads must use the same immutable revision as their provenance."""

import csv
from types import SimpleNamespace
from unittest.mock import Mock

from pettrace.data import benchmark as data


def test_load_resolves_tag_before_downloading_exact_commit(monkeypatch, tmp_path):
    import huggingface_hub

    commit = "a" * 40
    info = Mock(return_value=SimpleNamespace(sha=commit))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(dataset_info=info))

    def download(repo, **kwargs):
        info.assert_called_once_with("test/pets", revision="v4")
        assert repo == "test/pets"
        assert kwargs == {"repo_type": "dataset", "revision": commit, "cache_dir": str(tmp_path)}
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    loaded = object()
    from_dir = Mock(return_value=loaded)
    monkeypatch.setattr(data, "from_dir", from_dir)
    assert data.load("v4", repo="test/pets", cache_dir=str(tmp_path)) is loaded
    from_dir.assert_called_once_with(str(tmp_path / "snapshot"), repo="test/pets", version="v4", commit=commit)


def test_complete_benchmark_without_split_is_loaded_as_test(tmp_path):
    with (tmp_path / "animals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["animal_id", "species"])
        writer.writeheader()
        writer.writerow({"animal_id": "cat-1", "species": "cat"})
    with (tmp_path / "metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["file_name", "animal_id", "species", "light"])
        writer.writeheader()
        writer.writerow({"file_name": "cat.jpg", "animal_id": "cat-1", "species": "cat", "light": "0.5"})

    dataset = data.from_dir(str(tmp_path))

    assert dataset.photos[0].split == "test"
    assert dataset.split("test") == dataset.photos
