import json
from types import SimpleNamespace

import pytest

from pettrace.data import volume


def prepared():
    rows = [
        {"source": "lcw", "species": "cat", "animal": "cat-1", "split": "train"},
        {"source": "lcw", "species": "cat", "animal": "cat-2", "split": "val"},
        {"source": "dogs-world", "species": "dog", "animal": "dog-1", "split": "train"},
        {"source": "dogs-world", "species": "dog", "animal": "dog-2", "split": "test"},
    ]
    return SimpleNamespace(rows=rows, fingerprint="frozen", provenance={"version": "v2"})


def test_training_volume_reports_public_source_coverage(tmp_path, monkeypatch):
    capacities = {
        "lcw": {"ref": "cats", "version": 1, "species": "cat", "photos": 10,
                "archive_index_sha256": "c" * 64},
        "dogs-world": {"ref": "dogs", "version": 1, "species": "dog", "photos": 20,
                       "archive_index_sha256": "d" * 64},
    }
    monkeypatch.setattr(volume, "CAPACITY", capacities)
    metrics, params = volume.training_volume(prepared(), 100, tmp_path)
    assert metrics["dataset.source_photos"] == 30
    assert metrics["dataset.train_photos"] == 2
    assert metrics["dataset.excluded_photos"] == 26
    assert params["dataset.transport"] == "huggingface_refined_v2"
    assert json.loads((tmp_path / "sources.json").read_text()) == {"version": "v2"}


def test_training_volume_rejects_partial_or_unverified_inputs(tmp_path):
    with pytest.raises(ValueError, match="complete 100%"):
        volume.training_volume(prepared(), 50, tmp_path)
    with pytest.raises(ValueError, match="complete 100%"):
        volume.training_volume(SimpleNamespace(rows=[]), 100, tmp_path)
