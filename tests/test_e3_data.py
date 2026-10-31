import csv
import json

import numpy as np
import pytest
from PIL import Image

from pettrace.data.sources import acquire, curate_review, discover_body_ids, select_pilot, write_review


def test_dog_sidecars_use_individual_id_and_reject_ambiguous_and_escaping_paths(tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "images").mkdir()
    Image.new("RGB", (16, 16)).save(tmp_path / "images/a.png")
    sidecar = tmp_path / "metadata/a.json"
    sidecar.write_text(json.dumps({"path": "images/a.png", "identities": [{"identity": "dog-7", "breeds": ["Husky"]}]}))
    rows, skipped = discover_body_ids(tmp_path, "dogs-world")
    assert rows[0]["animal"] == "dogs-world:dog:dog-7" and not skipped
    sidecar.write_text(json.dumps({"path": "images/a.png", "identities": [{"identity": "1"}, {"identity": "2"}]}))
    with pytest.raises(ValueError, match="No usable"):
        discover_body_ids(tmp_path, "dogs-world")
    sidecar.write_text(json.dumps({"path": "../escape.png", "identities": [{"identity": "1"}]}))
    with pytest.raises(ValueError, match="escapes"):
        discover_body_ids(tmp_path, "dogs-world")


def test_lcw_reads_identity_folders_not_breeds(tmp_path):
    directory = tmp_path / "CatFullDataset/12"
    directory.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(directory / "1.jpg")
    rows, skipped = discover_body_ids(tmp_path, "lcw")
    assert rows[0]["animal"] == "lcw:cat:12" and not skipped


def make_rows(root):
    rng = np.random.default_rng(42)
    rows = []
    for species, source in [("cat", "lcw"), ("dog", "dogs-world")]:
        for identity in range(12):
            folder = root / source / str(identity)
            folder.mkdir(parents=True)
            for image in range(4):
                path = folder / f"{image}.png"
                Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(path)
                rows.append({"path": str(path), "source": source, "species": species,
                             "animal": f"{source}:{species}:{identity}", "original_test": False})
    return rows


def approve(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row["decision"] = "keep"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def test_review_is_required_persistent_and_bound_to_photos_and_labels(tmp_path):
    rows = select_pilot(make_rows(tmp_path), identities_per_species=12, photos_per_identity=4)
    path = write_review(rows, tmp_path / "review.csv")
    with pytest.raises(ValueError, match="keep or drop"):
        curate_review(path, tmp_path / "curated")
    approve(path)
    approved = path.read_bytes()
    write_review(rows, path)
    assert path.read_bytes() == approved
    retained, report = curate_review(path, tmp_path / "curated")
    assert len(retained) == 96
    assert report["counts"]["cat.train"]["identities"] == 8
    assert report["counts"]["dog.val"]["identities"] == 2
    identities = {}
    for row in retained:
        identities.setdefault(row["animal"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in identities.values())
    changed = [dict(row) for row in rows]
    changed[0]["animal"] += "-changed"
    with pytest.raises(ValueError, match="input changed"):
        write_review(changed, path)
    Image.new("RGB", (16, 16), "red").save(rows[0]["path"])
    with pytest.raises(ValueError, match="changed after review"):
        curate_review(path, tmp_path / "changed")


def test_duplicate_removal_enforces_four_actual_photos(tmp_path):
    rows = make_rows(tmp_path)
    path = write_review(rows, tmp_path / "review.csv")
    # Review the repeated bytes as well, then require four distinct images during the audit.
    first = next(row for row in rows if row["animal"] == "lcw:cat:0")
    others = [row for row in rows if row["animal"] == first["animal"] and row != first]
    from pathlib import Path
    Path(others[0]["path"]).write_bytes(Path(first["path"]).read_bytes())
    path.unlink()
    write_review(rows, path)
    approve(path)
    retained, report = curate_review(path, tmp_path / "curated")
    assert all(row["animal"] != "lcw:cat:0" for row in retained)
    assert "lcw:cat:0" in report["insufficient_photo_identities_excluded"]


def test_acquisition_does_not_download_implicitly_and_blocks_incomplete_extraction(tmp_path, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("network is forbidden")
    monkeypatch.setattr("pettrace.data.sources.requests.get", forbid)
    with pytest.raises(FileNotFoundError, match="ALLOW_DATA_DOWNLOAD"):
        acquire(tmp_path, "lcw")
    folder = tmp_path / "lcw"
    folder.mkdir()
    (folder / "partial.jpg").write_bytes(b"incomplete")
    (tmp_path / "lcw-source.json").write_text(json.dumps({"extraction_pending": True, "version": 1}))
    with pytest.raises(ValueError, match="Incomplete extraction"):
        acquire(tmp_path, "lcw")


def test_exploratory_filters_technical_problems_without_approving_pending_labels(tmp_path):
    from pathlib import Path

    rows = make_rows(tmp_path / "source")
    def group(animal):
        return [r for r in rows if r["animal"] == animal]
    Path(group("dogs-world:dog:0")[0]["path"]).write_bytes(b"invalid source image")
    repeated = group("dogs-world:dog:3")
    Path(repeated[1]["path"]).write_bytes(Path(repeated[0]["path"]).read_bytes())
    Path(group("lcw:cat:1")[0]["path"]).write_bytes(Path(group("lcw:cat:0")[0]["path"]).read_bytes())
    path = write_review(rows, tmp_path / "review.csv")
    with path.open(newline="") as stream:
        decisions = list(csv.DictReader(stream))
    dropped = group("dogs-world:dog:1")[0]["path"]
    verified = group("dogs-world:dog:4")[0]["path"]
    for row in decisions:
        if row["path"] == dropped:
            row["decision"] = "drop"
        if row["path"] == verified:
            row["decision"] = "keep"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, decisions[0].keys())
        writer.writeheader()
        writer.writerows(decisions)
    before = path.read_bytes()
    retained, audit = curate_review(path, tmp_path / "curated", review_policy="exploratory",
                                    excluded_animals={"dogs-world:dog:2": "known ambiguous target"})
    assert path.read_bytes() == before
    assert audit["review_policy"] == "exploratory"
    assert len(audit["corrupt"]) == 1 and len(audit["removed_duplicates"]) == 1
    assert audit["excluded_animal_photos"] == 4
    assert audit["conflicting_identities_quarantined"] == ["lcw:cat:0", "lcw:cat:1"]
    assert audit["retained_with_unverified_source_label"] == len(retained) - 1
    assert audit["review_decisions"][""] == 94
    forbidden = {"dogs-world:dog:0", "dogs-world:dog:1", "dogs-world:dog:2", "dogs-world:dog:3",
                 "lcw:cat:0", "lcw:cat:1"}
    assert not forbidden.intersection(row["animal"] for row in retained)
    assert {row["species"] for row in retained} == {"cat", "dog"}
    assert {row["split"] for row in retained} == {"train", "val", "test"}
    assert all(len({r["split"] for r in retained if r["animal"] == row["animal"]}) == 1 for row in retained)


def test_exploratory_keeps_integrity_checks_and_strict_review_default(tmp_path):
    rows = make_rows(tmp_path / "source")
    path = write_review(rows, tmp_path / "review.csv")
    with pytest.raises(ValueError, match="keep or drop"):
        curate_review(path, tmp_path / "strict")
    with pytest.raises(ValueError, match="review_policy"):
        curate_review(path, tmp_path / "bad-policy", review_policy="approve-all")
    from pathlib import Path
    Path(rows[0]["path"]).write_bytes(b"changed after selection")
    with pytest.raises(ValueError, match="changed after review"):
        curate_review(path, tmp_path / "changed", review_policy="exploratory")


def test_detector_exclusions_precede_minimum_photo_counts_and_split_assignment(tmp_path):
    rows = make_rows(tmp_path / "source")
    path = write_review(rows, tmp_path / "review.csv")
    excluded = next(row for row in rows if row["animal"] == "dogs-world:dog:0")
    original = path.read_bytes()
    retained, audit = curate_review(path, tmp_path / "curated", review_policy="exploratory",
                                    excluded_paths={excluded["path"]: "multiple_animals"})
    assert audit["excluded_path_photos"] == 1
    assert excluded["animal"] in audit["insufficient_photo_identities_excluded"]
    assert all(row["animal"] != excluded["animal"] for row in retained)
    assert path.read_bytes() == original
