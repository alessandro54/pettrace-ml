import csv
import json
from pathlib import Path

import pytest

from pettrace.data.checkpoints import DriveBackup
from pettrace.data.lookup import load_candidates, lookup_duplicates, restore_lookup_review
from pettrace.data.review import quarantine_groups
from pettrace.data.sources import REVIEW_FIELDS


def inventory(tmp_path):
    rows = []
    for index, (animal, digest) in enumerate((("a", "a"), ("a", "a"), ("b", "a"), ("b", "b"),
                                               ("c", "b"), ("d", "c"), ("a", "d"))):
        source, species = ("dogs-world", "dog") if animal == "d" else ("lcw", "cat")
        rows.append({"path": f"/not-mounted/{index}.jpg", "source": source, "species": species,
                     "animal": f"{source}:{species}:{animal}", "sha256": digest * 64,
                     "decision": "drop" if index == 0 else "", "note": "original decision"})
    path = tmp_path / "review.csv"
    write(path, rows)
    return path, rows


def write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_lookup_counts_full_inventory_without_images_and_keeps_ambiguous_labels(tmp_path):
    path, rows = inventory(tmp_path)
    before = path.read_bytes()
    report = lookup_duplicates(path, tmp_path / "lookup")
    summary = json.loads((report / "summary.json").read_text())
    assert (summary["files"], summary["unique_images"], summary["duplicate_copies"]) == (7, 4, 3)
    assert summary["conflicting_hashes"] == 2 and summary["conflicting_identity_labels"] == 3
    assert summary["identities_by_distinct_photos"] == {"1": 2, "2": 2}
    assert summary["review_decisions_recorded"] == {"drop": 1, "pending": 6}
    assert summary["by_source"]["lcw"] == {"files": 6, "unique_images": 3, "within_source_duplicate_copies": 3}
    unique, duplicate = read(report / "unique-images.csv"), read(report / "duplicate-copies.csv")
    assert len(unique) == 4 and len(duplicate) == 3
    assert json.loads(unique[0]["animal_labels"]) == ["lcw:cat:a", "lcw:cat:b"]
    assert unique[0]["status"] == "identity_review_pending"
    representatives = {row["sha256"]: row["path"] for row in unique}
    assert all(row["representative_path"] == representatives[row["sha256"]] for row in duplicate)
    assert {r["path"] for r in unique + duplicate} == {r["path"] for r in rows}
    assert len(quarantine_groups(report)) == 1
    assert path.read_bytes() == (report / "review.csv").read_bytes() == before
    assert lookup_duplicates(path, tmp_path / "lookup") == report
    assert not (report / "prepared.json").exists()
    (report / "unique-images.csv").write_text("corrupted")
    with pytest.raises(ValueError, match="report changed"):
        lookup_duplicates(path, tmp_path / "lookup")


@pytest.mark.parametrize("change", [
    {"sha256": "bad-hash"}, {"animal": "breed-name"}, {"source": "unknown"},
    {"species": "dog"}, {"decision": "approve"}, {"path": ""},
])
def test_lookup_rejects_invalid_inventory_before_publishing(tmp_path, change):
    path, rows = inventory(tmp_path)
    rows[0].update(change)
    write(path, rows)
    with pytest.raises(ValueError):
        lookup_duplicates(path, tmp_path / "lookup")
    assert not list((tmp_path / "lookup").glob("*/lookup.json"))


def test_lookup_new_input_keeps_previous_report_and_cross_source_conflict(tmp_path):
    path, rows = inventory(tmp_path)
    first = lookup_duplicates(path, tmp_path / "lookup")
    old = {p.name: p.read_bytes() for p in first.iterdir()}
    rows[5]["sha256"] = "a" * 64
    write(path, rows)
    second = lookup_duplicates(path, tmp_path / "lookup")
    assert first != second and {p.name: p.read_bytes() for p in first.iterdir()} == old
    unique = read(second / "unique-images.csv")
    assert json.loads(unique[0]["species_labels"]) == ["cat", "dog"]
    assert unique[0]["status"] == "identity_review_pending"


def test_data_lookup_notebook_runs_on_inventory_without_photos(tmp_path, monkeypatch):
    import IPython.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(IPython.display, "display", lambda *args: None)
    path, _ = inventory(tmp_path)
    data = tmp_path / "preparation"
    review = data / "source-100-seed42/review.csv"
    review.parent.mkdir(parents=True)
    review.write_bytes(path.read_bytes())
    drive = tmp_path / "drive"
    DriveBackup({"data": data}, drive / "checkpoints/preparation")(force=True)
    notebook = json.loads((Path(__file__).parents[1] / "datasets/data_lookup.ipynb").read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or cell["id"] == "lookup-install":
            continue
        if cell["id"] == "lookup-inputs":
            namespace.update(WORK=tmp_path / "new-runtime", DRIVE_ROOT=drive,
                             SOURCE_SAMPLE_PCT=100, OUTPUT_ROOT=drive / "data-lookup", CHECK_NEAR_COPIES=False)
        else:
            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    assert namespace["summary"]["duplicate_copies"] == 3
    assert namespace["REVIEW_CSV"].read_bytes() == path.read_bytes()
    # A different runtime recovers only the CSV and reuses the same persistent report.
    again = restore_lookup_review(drive, tmp_path / "another-runtime", percent=100)
    assert lookup_duplicates(again, drive / "data-lookup") == namespace["REPORT"]
    headings = [line for cell in notebook["cells"] if cell["cell_type"] == "markdown"
                for line in "".join(cell["source"]).splitlines() if line.startswith("## ")]
    assert len(headings) == 3 and all(f"Fase {i}." in line for i, line in enumerate(headings, 1))


def test_lookup_restore_requires_mount_and_matching_volume(tmp_path):
    with pytest.raises(FileNotFoundError, match="Mount Drive"):
        restore_lookup_review(tmp_path / "missing-drive", tmp_path / "work")
    with pytest.raises(ValueError, match="20, 50 or 100"):
        restore_lookup_review(tmp_path, tmp_path / "work", percent=10)
    with pytest.raises(FileNotFoundError, match="source-50"):
        restore_lookup_review(tmp_path, tmp_path / "work", percent=50)


def test_candidate_index_keeps_distinct_images_and_all_rejection_reasons(tmp_path):
    path, rows = inventory(tmp_path)
    for i in range(4):
        rows.append({"path": f"/not-mounted/eligible-{i}.jpg", "source": "lcw", "species": "cat",
                     "animal": "lcw:cat:eligible", "sha256": f"{10 + i:064x}", "decision": "", "note": ""})
    rows.append({**rows[-1], "path": "/not-mounted/z-keep.jpg", "decision": "keep"})
    rows.append({**rows[-2], "path": "/not-mounted/a-drop.jpg", "decision": "drop"})
    write(path, rows)
    report = lookup_duplicates(path, tmp_path / "lookup")
    candidates, excluded = load_candidates(report, path)
    assert len(candidates) == len({r['sha256'] for r in candidates}) == 4
    assert {r['animal'] for r in candidates} == {'lcw:cat:eligible'}
    assert any(r['path'].endswith('z-keep.jpg') for r in candidates)
    assert excluded['/not-mounted/a-drop.jpg'] == 'manual_drop'
    assert excluded['/not-mounted/eligible-3.jpg'] == 'exact_duplicate'
    assert excluded['/not-mounted/6.jpg'] == 'identity_review_pending'  # whole connected identity, even unique photo
    assert excluded['/not-mounted/5.jpg'] == 'insufficient_distinct_photos'
    assert len(candidates) + len(excluded) == len(rows)
    summary = json.loads((report / 'summary.json').read_text())
    assert summary['candidate_photos'] == 4 and summary['candidate_identities'] == 1
    assert summary['candidate_unverified_labels'] == 3
    with pytest.raises(ValueError, match='review/minimum'):
        load_candidates(report, path, minimum_photos=5)
    # A changed human decision invalidates the old selection rather than silently ignoring it.
    rows[-1]['decision'] = 'keep'
    write(path, rows)
    with pytest.raises(ValueError, match='review/minimum'):
        load_candidates(report, path)


def test_empty_candidate_index_is_valid_but_tampered_selection_is_rejected(tmp_path):
    path, _ = inventory(tmp_path)
    report = lookup_duplicates(path, tmp_path / 'lookup')
    candidates, excluded = load_candidates(report, path)
    assert not candidates and len(excluded) == 7
    (report / 'candidates.csv').write_text('changed')
    with pytest.raises(ValueError, match='index changed'):
        load_candidates(report, path)
