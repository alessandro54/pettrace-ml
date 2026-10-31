"""Nested acquisition reuses cached photos; all volumes share the existing filtering."""
import hashlib
import io
import json
from types import SimpleNamespace
import zipfile

import pytest

from pettrace.data import volume
from pettrace.data.pilot import RangeFile
from test_pilot_download import Session


def archive_fixture(source):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for i in range(10):
            path = f"images/{i}.jpg" if source == "dogs-world" else f"CatFullDataset/1/{i}.jpg"
            archive.writestr(path, f"photo-{i}".encode())
            if source == "dogs-world":
                archive.writestr(f"metadata/{i}.json", json.dumps({"path": path, "identities": [{"identity": "one"}]}))
    with zipfile.ZipFile(io.BytesIO(output.getvalue())) as archive:
        digest = hashlib.sha256()
        for info in archive.infolist():
            digest.update(json.dumps([info.filename, info.CRC, info.file_size, info.compress_size,
                                     info.header_offset], separators=(",", ":")).encode() + b"\n")
    return output.getvalue(), digest.hexdigest()


@pytest.mark.parametrize("source", ["dogs-world", "lcw"])
def test_acquisition_is_nested_and_only_downloads_added_photos(tmp_path, monkeypatch, source):
    blob, signature = archive_fixture(source)
    capacity = {**volume.CAPACITY[source], "photos": 10, "archive_index_sha256": signature}
    monkeypatch.setitem(volume.CAPACITY, source, capacity)
    monkeypatch.setattr(volume, "RangeFile", lambda *a, **k: RangeFile("https://fixture", session=Session(blob)))
    fetched = []
    windows = volume._photo_windows

    def track(remote, infos, workers):
        fetched.append({i.filename for i in infos})
        return windows(remote, infos, 1)

    monkeypatch.setattr(volume, "_photo_windows", track)
    receipts = []
    for percent in (20, 50, 100):
        _, receipt = volume.acquire_volume(tmp_path, source, percent=percent, allow_download=True, workers=1)
        receipts.append(receipt)
    assert [len(items) for items in fetched] == [2, 3, 5]
    assert len(set.union(*fetched)) == 10
    assert set(receipts[0]["file_sha256"]) <= set(receipts[1]["file_sha256"]) <= set(receipts[2]["file_sha256"])
    volume.acquire_volume(tmp_path, source, percent=20, allow_download=False)
    assert len(fetched) == 3
    first = next(name for name in receipts[0]["file_sha256"] if name.endswith('.jpg'))
    (tmp_path / "source-cache" / source / first).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        volume.acquire_volume(tmp_path, source, percent=20)


def test_selection_ignores_archive_order_and_rejects_changed_inventory(monkeypatch):
    monkeypatch.setitem(volume.CAPACITY, "lcw", {"photos": 10})
    infos = [SimpleNamespace(filename=f"CatFullDataset/1/{i}.jpg") for i in range(10)]
    a = volume.photo_sample(infos, "lcw", 20, 42)
    b = volume.photo_sample(list(reversed(infos)), "lcw", 50, 42)
    assert a == b[:2]
    with pytest.raises(ValueError, match="inventory"):
        volume.photo_sample(infos[:-1], "lcw", 20, 42)


def test_processing_must_be_complete_before_reporting_training(tmp_path, monkeypatch):
    capacities = {source: {"ref": source, "version": 1, "species": sp, "photos": 20, "archive_index_sha256": source}
                  for source, sp in (("dogs-world", "dog"), ("lcw", "cat"))}
    monkeypatch.setattr(volume, "CAPACITY", capacities)
    sources, rows, processing = [], [], {}
    for source, cap in capacities.items():
        prefix = "images" if source == "dogs-world" else "CatFullDataset/1"
        sources.append({**cap, "extraction_pending": False, "file_sha256": {f"{prefix}/{i}.jpg": f"{i:064x}" for i in range(4)}})
        rows.append({"source": source, "animal": source, "species": cap["species"], "split": "train", "sha256": f"{0:064x}"})
        processing[source] = {"requested_pct": 20, "processed_photos": 4, "train_photos": 1,
                              "excluded_photos": 3, "exclusion_reasons": {"filtered": 3}}
    (tmp_path / "sources.json").write_text(json.dumps({"sources": sources}))
    prepared = SimpleNamespace(root=tmp_path, rows=rows, summary={"processing": processing})
    metrics, _ = volume.training_volume(prepared, 20, tmp_path)
    assert metrics["dataset.processed_pct"] == 20 and metrics["dataset.train_pct"] == 5
    processing["lcw"]["processed_photos"] = 3
    with pytest.raises(ValueError, match="processing"):
        volume.training_volume(prepared, 20, tmp_path)
