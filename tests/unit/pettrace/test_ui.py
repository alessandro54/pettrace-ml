"""Compact progress and table rendering."""

import pytest

from pettrace import ui


class Bar:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.count = 0
        self.postfix = ""
        self.closed = False

    def update(self, amount=1):
        self.count += amount

    def set_postfix_str(self, value):
        self.postfix = value

    def close(self):
        self.closed = True


def test_epoch_progress_is_compact_and_tracks_best(monkeypatch):
    bars = []
    monkeypatch.setattr(ui, "tqdm", lambda **kwargs: bars.append(Bar(**kwargs)) or bars[-1])
    progress = ui.EpochProgress(3, 46)
    progress.update({"epoch": 1, "train.loss": 0.2, "val.macro_species_mAP": 0.7})
    progress.update({"epoch": 2, "train.loss": 0.1, "val.macro_species_mAP": 0.6})
    progress.close()
    assert bars[0].count == 2
    assert bars[0].kwargs["desc"] == "Training seed 46"
    assert "best=0.7000@1" in bars[0].postfix
    assert bars[0].closed


def test_markdown_table_escapes_cells_and_rejects_bad_rows():
    assert ui.markdown_table(["A", "B"], [["x|y", "line\nbreak"]]) == (
        "| A | B |\n|---|---|\n| x\\|y | line break |"
    )
    with pytest.raises(ValueError, match="match the headers"):
        ui.markdown_table(["A"], [[1, 2]])
