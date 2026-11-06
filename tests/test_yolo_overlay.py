import csv
import gzip
import io
import json
import tarfile

from PIL import Image

from pettrace.data.verification import sha256
from pettrace.data.yolo_overlay import _extract_rows, _write_overlay, update_dataset_card


def release_fixture(tmp_path):
    root = tmp_path / "release"
    version = root / "versions/v1"
    shard = version / "data/train/lcw-00000.tar"
    shard.parent.mkdir(parents=True)
    image = tmp_path / "cat.jpg"
    Image.new("RGB", (16, 12), "orange").save(image)
    digest = sha256(image)
    metadata = {
        "key": digest, "identity_id": "lcw:cat:1", "species": "cat", "source": "lcw",
        "split": "train", "sha256": digest, "original_test": "False",
        "shard": "versions/v1/data/train/lcw-00000.tar",
    }
    with tarfile.open(shard, "w") as archive:
        archive.add(image, arcname=f"{digest}.jpg")
        payload = json.dumps(metadata).encode()
        info = tarfile.TarInfo(f"{digest}.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    metadata_path = version / "metadata.csv"
    with metadata_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, metadata.keys())
        writer.writeheader()
        writer.writerow(metadata)
    evidence = {
        "source": "lcw", "release_version": "v1", "prepared_version": "v4.100",
        "prepared_fingerprint": "fixture", "photos": 1,
        "file_sha256": {"versions/v1/metadata.csv": sha256(metadata_path)},
        "shards": [{"path": metadata["shard"], "sha256": sha256(shard), "samples": 1}],
    }
    return root, evidence, metadata, digest


def test_extracts_verified_v1_and_writes_compact_overlay(tmp_path):
    root, evidence, metadata, digest = release_fixture(tmp_path)
    rows, indexed = _extract_rows(root, evidence, tmp_path / "images")
    assert rows == [{"path": str(tmp_path / "images" / f"{digest}.jpg"), "source": "lcw",
                     "species": "cat", "animal": "lcw:cat:1", "sha256": digest}]
    report = {
        "records": [{"image_sha256": digest, "reason": "single_expected_animal",
                     "boxes": [{"score": 0.9, "class_id": 15, "box": [1, 2, 10, 11]}]}],
        "config": {"confidence": 0.25, "nms_iou": 0.5}, "execution": {"provider": "CUDA"},
        "gpu_validation": {"passed": True}, "limitation": "fixture",
    }
    spec = {"repo_id": "owner/cats", "commit": "a" * 40}
    output = _write_overlay(tmp_path / "v2", spec, evidence, indexed, report)
    manifest = json.loads((output / "dataset.json").read_text())
    assert manifest["base"]["commit"] == "a" * 40
    assert manifest["photos"] == manifest["identities"] == 1
    with gzip.open(output / "screening.csv.gz", "rt") as stream:
        record = next(csv.DictReader(stream))
    assert record["sha256"] == digest
    assert record["identity_id"] == metadata["identity_id"]
    assert "path" not in record


def test_dataset_card_update_is_idempotent():
    manifest = {
        "base": {"commit": "b" * 40}, "photos": 2, "identities": 1,
        "localization": {"detector_model_uri": "models:/pettrace-embedder/2",
                         "detector_sha256": "c" * 64,
                         "config": {"confidence": 0.25, "nms_iou": 0.5}},
    }
    updated = update_dataset_card("---\nlicense: apache-2.0\n---\n\n# Cats\n", manifest)
    assert update_dataset_card(updated, manifest) == updated
    assert "reuses every full-frame image" in updated
