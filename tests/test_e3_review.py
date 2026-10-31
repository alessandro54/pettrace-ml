import csv

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
    with pytest.raises(ValueError, match="otra ventana"):
        save_decisions(path, updates, expected_sha256=digest)
    Image.new("RGB", (20, 30), "red").save(rows[0]["path"])
    with pytest.raises(ValueError, match="foto cambió"):
        save_decisions(path, updates, expected_sha256=sha256(path))
    assert path.read_bytes() == saved


def test_review_never_accepts_label_updates_and_gate_reports_remaining_count(tmp_path):
    rows, path = fixture_review(tmp_path)
    with pytest.raises(ValueError, match="decision/note"):
        save_decisions(path, {rows[0]["path"]: {"animal": "lcw:cat:9"}}, expected_sha256=sha256(path))
    with pytest.raises(ReviewPending, match="4/4") as exc:
        require_review_complete(path)
    assert exc.value._render_traceback_() == [str(exc.value)]
    assert "visor" in str(exc.value)
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
    assert any("Archivo ilegible" in w.value for w in controls if isinstance(w, widgets.HTML))
    assert path.read_bytes() == initial
    choices[0].value, choices[1].value = "keep", "drop"
    next(w for w in controls if isinstance(w, widgets.Button) and w.description == "Guardar y siguiente").click()
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
