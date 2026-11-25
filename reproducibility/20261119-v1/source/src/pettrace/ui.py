"""Consistent notebook and terminal progress for PetTrace workflows."""

from tqdm.auto import tqdm


def task(description, total, *, unit="item"):
    """Create one normalized progress bar for a measurable operation."""
    return tqdm(total=total, desc=description, unit=unit, dynamic_ncols=True, leave=True)


class EpochProgress:
    """Render compact training progress while JSON retains the complete history."""

    def __init__(self, total, seed):
        self._bar = task(f"Training seed {seed}", total, unit="epoch")
        self._epoch = 0
        self._best = float("-inf")
        self._best_epoch = 0

    def update(self, record):
        epoch = int(record["epoch"])
        score = float(record["val.macro_species_mAP"])
        if score > self._best:
            self._best, self._best_epoch = score, epoch
        self._bar.update(epoch - self._epoch)
        self._epoch = epoch
        self._bar.set_postfix_str(
            f"loss={record['train.loss']:.4f} · val mAP={score:.4f} · best={self._best:.4f}@{self._best_epoch}"
        )

    def close(self):
        self._bar.close()


def markdown_table(headers, rows):
    """Return a dependency-free Markdown table for notebooks, reports and terminals."""
    rows = [list(row) for row in rows]
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Every table row must match the headers")

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(map(cell, headers)) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(lines)
