"""Repeats must not silently use reordered or corrupt cached embeddings."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("repeat_e3", Path(__file__).parents[1] / "tools/repeat_e3.py")
repeat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repeat)


def test_cached_manifest_and_vector_integrity(tmp_path):
    rows = [{"animal": a, "species": "cat", "split": s, "sha256": h}
            for a, s, h in (("a", "train", "a"), ("v", "val", "v"), ("t", "test", "t"))]
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    config = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "cache_sha256": hashlib.sha256(json.dumps({"model": repeat.MODEL_KEY, "rows": rows[:2]},
                                                        sort_keys=True).encode()).hexdigest()}
    db = tmp_path / "features.sqlite3"
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1
    blob = vector.tobytes()
    checksum = hashlib.sha256(blob).hexdigest()
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE embeddings (image TEXT, vector BLOB, checksum TEXT)")
        conn.executemany("INSERT INTO embeddings VALUES (?, ?, ?)", [(h, blob, checksum) for h in ("a", "v")])
    all_rows, training, values = repeat.cached_vectors(manifest, config, db)
    assert all_rows == rows and training == rows[:2]
    np.testing.assert_array_equal(values, np.stack([vector, vector]))
    with pytest.raises(ValueError, match="identity/order"):
        repeat.cached_vectors(manifest, {**config, "cache_sha256": "wrong"}, db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE embeddings SET checksum='bad' WHERE image='a'")
    with pytest.raises(ValueError, match="Corrupt"):
        repeat.cached_vectors(manifest, config, db)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM embeddings WHERE image='a'")
    with pytest.raises(ValueError, match="Missing"):
        repeat.cached_vectors(manifest, config, db)
    manifest.write_text("changed")
    with pytest.raises(ValueError, match="Manifest"):
        repeat.cached_vectors(manifest, config, db)


def test_pixel_cache_keys_include_shape_and_mode():
    from PIL import Image
    images = [Image.new("RGB", (3, 3), color=(i, 0, 0)) for i in range(4)]
    assert len({repeat.image_key(im) for im in images}) == 4
    # Identical pixels copied to a new object retain their key.
    assert repeat.image_key(images[0]) == repeat.image_key(images[0].copy())
    assert repeat.image_key(Image.new("RGB", (3, 3))) != repeat.image_key(Image.new("RGB", (1, 9)))


def test_preparation_evidence_cannot_overwrite_run_results(tmp_path):
    source, report = tmp_path / "input", tmp_path / "report"
    source.mkdir()
    report.mkdir()
    (source / "summary.json").write_text("preparation")
    (source / "dataset-coverage.json").write_text("coverage")
    (source / "training-summary.json").write_text("stale training")
    (source / "registration.json").write_text("stale registration")
    (report / "summary.json").write_text("evaluation")

    repeat.copy_preparation_evidence(source, report)

    assert (report / "summary.json").read_text() == "evaluation"
    assert (report / "preparation/summary.json").read_text() == "preparation"
    assert (report / "preparation/dataset-coverage.json").read_text() == "coverage"
    assert not (report / "preparation/training-summary.json").exists()
    assert not (report / "preparation/registration.json").exists()
