import json

import pytest
from PIL import Image, ImageDraw

from pettrace.data.lookup import load_candidates, lookup_duplicates
from pettrace.data.near_copies import find_near_copies, pixel_error, signature
from pettrace.data.review import quarantine_gallery, quarantine_groups
from pettrace.data.sources import write_review
from pettrace.data.verification import sha256


def images(tmp_path):
    base = Image.new("RGB", (256, 256), "#c2a896")
    draw = ImageDraw.Draw(base)
    for i in range(10):
        draw.ellipse((10 + i * 18, 20 + i * 9, 60 + i * 18, 130 + i * 9), fill=(30 + i * 13, 70 + i * 8, 90))
    paths = [tmp_path / name for name in ("a.png", "b.jpg", "c.png", "unrelated.png", "wide.png")]
    base.save(paths[0])
    base.save(paths[1], quality=95)
    base.resize((128, 128)).save(paths[2])
    Image.new("RGB", base.size, "blue").save(paths[3])
    base.resize((512, 256)).save(paths[4])
    return paths


def test_copy_detection_precedes_minimum_and_quarantines_without_merging(tmp_path):
    paths = images(tmp_path)
    rows = [{"path": str(p), "animal": f"lcw:cat:{i}", "source": "lcw", "species": "cat"}
            for i, p in enumerate(paths)]
    review = write_review(rows, tmp_path / "review.csv")
    exact = lookup_duplicates(review, tmp_path / "reports")
    before = {p.name: p.read_bytes() for p in exact.iterdir()}
    report = lookup_duplicates(review, tmp_path / "reports", near_cache=tmp_path / "cache")
    summary = json.loads((report / "summary.json").read_text())
    assert summary["near_copy_images"] == 3 and summary["near_copy_identity_labels"] == 3
    assert summary["pre_yolo_exclusions"] == {"near_copy_review_pending": 3, "insufficient_distinct_photos": 2}
    candidates, reasons = load_candidates(report, review)
    assert not candidates and len(reasons) == 5
    assert {p.name: p.read_bytes() for p in exact.iterdir()} == before
    groups = quarantine_groups(report)
    assert len(groups) == 1 and groups[0]["animals"] == ["lcw:cat:0", "lcw:cat:1", "lcw:cat:2"]
    assert len(groups[0]["near_hashes"]) == 3 and groups[0]["shared_hashes"] == {}
    gallery = quarantine_gallery(report, tmp_path / "gallery.html")
    assert gallery.read_text().count("data:image/jpeg;base64,") == 3
    assert "MAE RGB" in gallery.read_text() and "pendientes" in gallery.read_text()
    # Reuse signatures in a new report without reopening source photos.
    for path in paths:
        path.unlink()
    again = lookup_duplicates(review, tmp_path / "other-reports", near_cache=tmp_path / "cache")
    assert (again / "audit.json").read_bytes() == (report / "audit.json").read_bytes()
    chunk = next((tmp_path / "cache").rglob("*.db"))
    chunk.write_bytes(b"bad cache")
    with pytest.raises(ValueError, match="cache checksum"):
        lookup_duplicates(review, tmp_path / "third-reports", near_cache=tmp_path / "cache")


def test_same_label_copies_corrupt_and_changed_files_are_not_training_candidates(tmp_path):
    paths = images(tmp_path)
    paths[-1].write_bytes(b"not a photo")
    rows = [{"path": str(p), "animal": "lcw:cat:a", "source": "lcw", "species": "cat"} for p in paths[:4]]
    rows.append({**rows[-1], "path": str(paths[-1]), "animal": "lcw:cat:bad"})
    review = write_review(rows, tmp_path / "review.csv")
    report = lookup_duplicates(review, tmp_path / "reports", near_cache=tmp_path / "cache")
    summary = json.loads((report / "summary.json").read_text())
    assert summary["pre_yolo_exclusions"] == {"near_copy_review_pending": 4, "decode_error": 1}
    assert len(quarantine_groups(report)) == 1
    paths[0].write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Source photo changed"):
        lookup_duplicates(review, tmp_path / "new-reports", near_cache=tmp_path / "empty-cache")


def test_banded_search_matches_bruteforce_and_shared_representation(tmp_path):
    paths = images(tmp_path)
    rows = {sha256(p): [{"path": str(p)}] for p in paths}
    actual, corrupt = find_near_copies(rows, tmp_path / "cache")
    assert not corrupt
    expected, features = [], {}
    for digest, row in sorted(rows.items()):
        with Image.open(row[0]["path"]) as image:
            current = signature(image)
        for old, previous in features.items():
            distance = (current[0] ^ previous[0]).bit_count()
            if distance <= 2 and abs(current[1] - previous[1]) < .01:
                error = pixel_error(current[2], previous[2])
                if error <= 2:
                    expected.append({"left": old, "right": digest, "hash_distance": distance, "rgb_mae": error})
        features[digest] = current
    assert actual == expected


def test_drive_restores_uncached_representatives_and_survives_new_kernel(tmp_path):
    from pettrace.data.checkpoints import DriveBackup
    data = tmp_path / "data"
    folder = data / "source-cache/lcw/images"
    folder.mkdir(parents=True)
    paths = images(folder)
    rows = {sha256(p): [{"path": str(p), "source": "lcw"}] for p in paths}
    drive = tmp_path / "drive"
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "unrelated.json").write_text("{}")
    DriveBackup({"data": data, "prepared": prepared}, drive / "checkpoints/preparation")(force=True)
    for path in paths:
        path.unlink()
    first = find_near_copies(rows, tmp_path / "cache", drive_root=drive, source_root=data)
    assert all(p.is_file() for p in paths)
    for path in paths:
        path.unlink()
    second = find_near_copies(rows, tmp_path / "new-cache", drive_root=drive, source_root=data)
    assert first == second and not any(p.exists() for p in paths)
    assert list((drive / "checkpoints/lookup-near").rglob("*.tar"))


def test_gallery_can_verify_one_original_without_requiring_exact_copies(tmp_path):
    path, *_ = images(tmp_path)
    duplicate = tmp_path / "duplicate.png"
    duplicate.write_bytes(path.read_bytes())
    rows = [{"path": str(p), "animal": f"lcw:cat:{i}", "source": "lcw", "species": "cat"}
            for i, p in enumerate((path, duplicate))]
    review = write_review(rows, tmp_path / "review.csv")
    report = lookup_duplicates(review, tmp_path / "reports")
    duplicate.unlink()
    with pytest.raises(FileNotFoundError):
        quarantine_gallery(report, tmp_path / "all.html")
    output = quarantine_gallery(report, tmp_path / "one.html", representatives_only=True)
    assert "un original por SHA256" in output.read_text()


def test_three_bands_do_not_miss_matches_at_partition_boundaries(tmp_path, monkeypatch):
    import pettrace.data.near_copies as module
    hashes = [0] + [1 << n for n in (0, 21, 22, 42, 43, 63)]
    hashes += [(1 << a) | (1 << b) for a, b in ((0, 22), (22, 43), (21, 63), (42, 43))]
    hashes += [(1 << 0) | (1 << 22) | (1 << 43)]
    rows, assigned = {}, {}
    for n, value in enumerate(hashes):
        path = tmp_path / f"{n}.png"
        Image.new("RGB", (12, 12), (n, 0, 0)).save(path)
        digest = sha256(path)
        rows[digest] = [{"path": str(path)}]
        assigned[digest] = value
    monkeypatch.setattr(module, "signature", lambda image:
                        (hashes[image.getpixel((0, 0))[0]], 1.0, Image.new("RGB", (64, 64), "gray")))
    pairs, _ = find_near_copies(rows, tmp_path / "cache")
    ordered = sorted(rows)
    expected = {(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1:]
                if (assigned[a] ^ assigned[b]).bit_count() <= 2}
    assert {(p["left"], p["right"]) for p in pairs} == expected


def test_failed_scan_keeps_completed_signatures_for_retry(tmp_path):
    paths = images(tmp_path)
    rows = {sha256(p): [{"path": str(p)}] for p in paths}
    ordered = sorted(rows)
    changed = rows[ordered[-1]][0]["path"]
    from pathlib import Path
    original = Path(changed).read_bytes()
    Path(changed).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source photo changed"):
        find_near_copies(rows, tmp_path / "cache")
    assert list((tmp_path / "cache").rglob("*.db"))
    Path(changed).write_bytes(original)
    # All other originals are now absent, so success proves reuse after a failed pass.
    for digest in ordered[:-1]:
        Path(rows[digest][0]["path"]).unlink()
    pairs, _ = find_near_copies(rows, tmp_path / "cache")
    assert pairs
