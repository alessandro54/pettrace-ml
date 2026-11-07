import csv
import json
from types import SimpleNamespace

import pytest

from pettrace.data import snapshots
from pettrace.data.verification import sha256


def snapshot(tmp_path):
    root = tmp_path / "versions/v1"
    root.mkdir(parents=True)
    inputs = {"config": {"minimum_photos": 1}}
    rows = []
    counts = {}
    for species, source in (("cat", "lcw"), ("dog", "dogs-world")):
        for split, identities in (("train", 8), ("val", 2), ("test", 2)):
            counts[f"{species}.{split}"] = {"identities": identities, "photos": identities}
            for index in range(identities):
                relative = f"images/{species}-{split}-{index}.jpg"
                image = root / relative
                image.parent.mkdir(exist_ok=True)
                image.write_bytes(f"{species}-{split}-{index}".encode())
                rows.append({"path": relative, "source": source,
                             "animal": f"{source}:{species}:{split}-{index}",
                             "species": species, "split": split, "sha256": sha256(image),
                             "original_test": "False"})
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, snapshots.MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps({"counts": counts}))
    for name in ("audit.json", "sources.json", "animal-screening.json"):
        (root / name).write_text("{}")
    (root / "review.csv").write_text("path\n")
    files = {row["path"]: row["sha256"] for row in rows}
    for name in ("manifest.csv", "summary.json", "audit.json", "sources.json",
                 "animal-screening.json", "review.csv"):
        files[name] = sha256(root / name)
    (root / "prepared.json").write_text(json.dumps({
        "schema": 1, "status": "complete", "inputs": inputs,
        "fingerprint": snapshots._digest(inputs), "files": files,
    }))
    return root


def test_load_prepared_verifies_snapshot_and_rejects_changed_images(tmp_path):
    root = snapshot(tmp_path)
    loaded = snapshots.load_prepared(root)
    assert len(loaded.rows) == 24
    assert loaded.fingerprint == snapshots._digest({"config": {"minimum_photos": 1}})
    next((root / "images").iterdir()).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        snapshots.load_prepared(root)


def test_load_training_pins_commit_and_fingerprint(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    commit = "c" * 40
    monkeypatch.setattr(snapshots, "HfApi", lambda: SimpleNamespace(
        dataset_info=lambda repo, revision: SimpleNamespace(sha=commit)))
    monkeypatch.setattr(snapshots, "snapshot_download", lambda *args, **kwargs: tmp_path)
    loaded, receipt = snapshots.load_training(
        repo_id="example/public", version="v1", revision=commit,
        expected_fingerprint=snapshots._digest({"config": {"minimum_photos": 1}}),
        cache_dir=tmp_path / "cache",
    )
    assert loaded.root == root
    assert receipt["commit"] == commit
    with pytest.raises(ValueError, match="fingerprint"):
        snapshots.load_training(repo_id="example/public", version="v1", revision=commit,
                                expected_fingerprint="0" * 64, cache_dir=tmp_path / "cache")
