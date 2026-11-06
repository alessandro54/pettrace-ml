import csv
import json

import pytest
from PIL import Image

from pettrace.data.sources import write_review
from pettrace.data.review import ReviewPending, photo_reviewer, require_review_complete, review_status, save_decisions
from pettrace.data.curation import sha256


def fixture_review(tmp_path):
    rows = []
    for identity in (1, 2):
        for photo in (1, 2):
            path = tmp_path / f"{identity}-{photo}.jpg"
            Image.new("RGB", (20, 30), "brown").save(path)
            rows.append({"path": str(path), "source": "lcw", "species": "cat", "animal": f"lcw:cat:{identity}"})
    return rows, write_review(rows, tmp_path / "review.csv")


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def descendants(widget):
    yield widget
    for child in getattr(widget, "children", ()):
        yield from descendants(child)


def test_review_saves_only_explicit_choices_and_rejects_stale_or_changed_inputs(tmp_path):
    rows, path = fixture_review(tmp_path)
    before = read_rows(path)
    digest = sha256(path)
    updates = {rows[0]["path"]: {"decision": "drop", "note": "dos gatos, objetivo ambiguo"}}
    save_decisions(path, updates, expected_sha256=digest)
    after = read_rows(path)
    assert {k: after[0][k] for k in before[0] if k not in {"decision", "note"}} == {
        k: before[0][k] for k in before[0] if k not in {"decision", "note"}}
    assert after[0]["decision"] == "drop" and after[1:] == before[1:]
    saved = path.read_bytes()
    with pytest.raises(ValueError, match="another window"):
        save_decisions(path, updates, expected_sha256=digest)
    Image.new("RGB", (20, 30), "red").save(rows[0]["path"])
    with pytest.raises(ValueError, match="photo changed"):
        save_decisions(path, updates, expected_sha256=sha256(path))
    assert path.read_bytes() == saved


def test_review_never_accepts_label_updates_and_gate_reports_remaining_count(tmp_path):
    rows, path = fixture_review(tmp_path)
    with pytest.raises(ValueError, match="decision/note"):
        save_decisions(path, {rows[0]["path"]: {"animal": "lcw:cat:9"}}, expected_sha256=sha256(path))
    with pytest.raises(ReviewPending, match="4/4") as exc:
        require_review_complete(path)
    assert exc.value._render_traceback_() == [str(exc.value)]
    assert "viewer" in str(exc.value)
    save_decisions(path, {r["path"]: {"decision": "drop", "note": "explicit test decision"} for r in rows},
                   expected_sha256=sha256(path))
    assert require_review_complete(path) == {"total": 4, "keep": 0, "drop": 4, "pending": 0}


def test_widget_displays_bad_files_without_approval_saves_and_resumes(tmp_path):
    widgets = pytest.importorskip("ipywidgets")
    rows, path = fixture_review(tmp_path)
    # Bind the unreadable file to the input review, just as the source acquisition does.
    from pathlib import Path
    Path(rows[1]["path"]).write_bytes(b"invalid source image")
    path.unlink()
    write_review(rows, path)
    initial = path.read_bytes()
    viewer = photo_reviewer(path)
    controls = list(descendants(viewer))
    choices = [w for w in controls if isinstance(w, widgets.Dropdown)]
    assert len(choices) == 2 and all(w.value == "" for w in choices)
    assert any("Unreadable file" in w.value for w in controls if isinstance(w, widgets.HTML))
    assert path.read_bytes() == initial
    choices[0].value, choices[1].value = "keep", "drop"
    next(w for w in controls if isinstance(w, widgets.Button) and w.description == "Save and next").click()
    assert review_status(path) == {"total": 4, "keep": 1, "drop": 1, "pending": 2}
    assert "lcw:cat:2" in viewer.children[0].value
    resumed = photo_reviewer(path)
    assert "lcw:cat:2" in resumed.children[0].value
    assert all(w.value == "" for w in descendants(resumed) if isinstance(w, widgets.Dropdown))
    assert [r["decision"] for r in read_rows(path)] == ["keep", "drop", "", ""]


def test_prompt_review_rejects_incomplete_answers_and_resumes_saved_progress(tmp_path, monkeypatch):
    from pettrace.data.review import review_with_prompts

    rows, path = fixture_review(tmp_path)
    monkeypatch.setattr("IPython.display.display", lambda value: None)
    monkeypatch.setattr("IPython.display.clear_output", lambda **kwargs: None)
    answers = iter(["k", "k d", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert review_with_prompts(path) == {"total": 4, "keep": 1, "drop": 1, "pending": 2}
    assert [row["decision"] for row in read_rows(path)] == ["keep", "drop", "", ""]
    answers = iter(["d k"])
    assert review_with_prompts(path) == {"total": 4, "keep": 2, "drop": 2, "pending": 0}
    assert [row["decision"] for row in read_rows(path)] == ["keep", "drop", "drop", "keep"]


def test_exclusion_summary_distinguishes_singletons_and_corrupt_fallback(tmp_path):
    from pettrace.data.review import exclusion_summary

    rows = []
    for animal, size in (("one", 1), ("two", 2), ("three", 3)):
        rows.extend({"path": f"{animal}-{i}", "animal": animal, "species": "cat",
                     "reason": "insufficient_distinct_photos"} for i in range(size))
    rows.append({"path": "broken", "animal": "three", "species": "cat", "reason": "insufficient_distinct_photos"})
    with (tmp_path / "excluded.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "audit.json").write_text(json.dumps({"minimum_photos": 4, "corrupt": [{"path": "broken"}],
        "conflicting_hashes": {}, "conflicting_identities_quarantined": []}))
    result = exclusion_summary(tmp_path)
    assert result["excluded_identities_by_distinct_photos"] == {"cat": {1: 1, 2: 1, 3: 1}, "dog": {}}
    assert result["additional_identities_at_minimum_2"] == 2
    assert result["additional_photos_at_minimum_2"] == 5


def test_quarantine_gallery_shows_conflict_context_without_changing_decisions(tmp_path):
    from pettrace.data.review import quarantine_gallery

    root = tmp_path / "snapshot"
    root.mkdir()
    rows, path = fixture_review(root)
    digest = sha256(rows[0]["path"])
    (root / "audit.json").write_text(json.dumps({"conflicting_hashes": {digest: rows}}))
    with (root / "excluded.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["path", "animal", "species", "reason"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in ("path", "animal", "species")} | {"reason": "conflicting_identity"}
                         for row in rows)
    original = {p.name: p.read_bytes() for p in root.iterdir()}
    output = quarantine_gallery(root, tmp_path / "review.html", count=1)
    text = output.read_text()
    assert "lcw:cat:1" in text and "lcw:cat:2" in text and "4 files" in text
    assert text.count('class="shared"') == 1 and text.count("data:image/jpeg;base64,") == 1
    assert all(row["path"].rsplit("/", 1)[-1] in text for row in rows)
    assert "All cases remain pending" in text
    assert {p.name: p.read_bytes() for p in root.iterdir()} == original
    with pytest.raises(ValueError, match="immutable"):
        quarantine_gallery(root, root / "review.html")
    with pytest.raises(ValueError, match="No conflicts"):
        quarantine_gallery(root, tmp_path / "empty.html", start=1)
    Image.new("RGB", (20, 30), "red").save(rows[0]["path"])
    with pytest.raises(ValueError, match="Source photo changed"):
        quarantine_gallery(root, tmp_path / "changed.html")


def test_quarantine_groups_connect_indirect_labels_once_and_stably(tmp_path):
    from pettrace.data.review import quarantine_groups

    conflicts = {"h1": [{"animal": a} for a in ("a", "b", "a")],
                 "h2": [{"animal": a} for a in ("b", "c")],
                 "h3": [{"animal": a} for a in ("c", "a")],
                 "h4": [{"animal": a} for a in ("d", "e")]}
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"conflicting_hashes": conflicts}))
    expected = [{"animals": ["a", "b", "c"],
                 "shared_hashes": {"h1": ["a", "b"], "h2": ["b", "c"], "h3": ["a", "c"]}},
                {"animals": ["d", "e"], "shared_hashes": {"h4": ["d", "e"]}}]
    before = audit.read_bytes()
    assert quarantine_groups(tmp_path) == expected
    assert audit.read_bytes() == before
    audit.write_text(json.dumps({"conflicting_hashes": {k: list(reversed(v))
                                                       for k, v in reversed(list(conflicts.items()))}}))
    assert quarantine_groups(tmp_path) == expected
    audit.write_text(json.dumps({"conflicting_hashes": {}}))
    assert quarantine_groups(tmp_path) == []
    audit.write_text(json.dumps({"conflicting_hashes": {"h1": [{"animal": "a"}]}}))
    with pytest.raises(ValueError, match="two identity labels"):
        quarantine_groups(tmp_path)


def test_quarantine_gallery_shows_indirect_group_context_and_checks_each_link(tmp_path):
    from pettrace.data.review import quarantine_gallery

    root = tmp_path / "snapshot"
    root.mkdir()
    rows = []
    for animal, color in (("a", "red"), ("b", "red"), ("b", "blue"), ("c", "blue"), ("c", "green")):
        path = root / f"{animal}-{color}.jpg"
        Image.new("RGB", (20, 30), color).save(path)
        rows.append({"path": str(path), "animal": animal, "species": "cat", "source": "lcw"})
    review = write_review(rows, root / "review.csv")
    red, blue = sha256(rows[0]["path"]), sha256(rows[2]["path"])
    (root / "audit.json").write_text(json.dumps({"conflicting_hashes": {
        red: rows[:2], blue: rows[2:4]}}))
    (root / "excluded.csv").write_text("path,animal,reason\n")
    output = quarantine_gallery(root, tmp_path / "review.html", count=1)
    text = output.read_text()
    assert "Groups 1–1 of 1" in text and "3 labels · 5 files · 3 distinct images" in text
    assert text.count("data:image/jpeg;base64,") == 3
    assert text.count('class="shared"') == 2
    assert "c-green.jpg" in text and "Indirect links" in text
    # C still has the blue shared hash; missing B's blue copy must also be rejected.
    recorded = read_rows(review)
    with review.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, recorded[0].keys())
        writer.writeheader()
        writer.writerows(r for r in recorded if r["path"] != rows[2]["path"])
    with pytest.raises(ValueError, match="does not match"):
        quarantine_gallery(root, tmp_path / "incomplete.html")
