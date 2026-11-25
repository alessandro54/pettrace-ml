"""Read-only loader for the immutable public benchmark snapshot."""

import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from pettrace.data.verification import sha256


SOURCES = {"lcw": "cat", "dogs-world": "dog"}
MANIFEST_FIELDS = ["path", "source", "animal", "species", "split", "sha256", "original_test"]


@dataclass
class PreparedDataset:
    root: Path
    fingerprint: str
    rows: list[dict]
    summary: dict


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, default=str,
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_file(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
        raise ValueError(f"Unsafe snapshot file: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing snapshot file: {relative}")
    return path


def _validate_rows(rows, minimum_photos):
    groups = defaultdict(list)
    for row in rows:
        source = row["source"]
        if (
            source not in SOURCES
            or row["species"] != SOURCES[source]
            or row["split"] not in {"train", "val", "test"}
            or not row["animal"].startswith(f"{source}:{row['species']}:")
        ):
            raise ValueError("Invalid snapshot source, species, split or identity")
        groups[row["animal"]].append(row)
    if len({row["path"] for row in rows}) != len(rows) or len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Snapshot repeats a path or image hash")
    if any(len(photos) < minimum_photos or len({row["split"] for row in photos}) != 1
           for photos in groups.values()):
        raise ValueError("Insufficient photos or identity leakage between splits")
    counts = {}
    for species in ("cat", "dog"):
        for split, required in (("train", 8), ("val", 2), ("test", 2)):
            photos = [row for row in rows if row["species"] == species and row["split"] == split]
            count = {"identities": len({row["animal"] for row in photos}), "photos": len(photos)}
            if count["identities"] < required:
                raise ValueError(f"Too few {species}.{split} identities; need {required}")
            counts[f"{species}.{split}"] = count
    return counts


def load_prepared(path):
    """Load a completed snapshot and verify every recorded artifact and image."""
    root = Path(path).resolve()
    receipt = json.loads((root / "prepared.json").read_text())
    if receipt.get("schema") != 1 or receipt.get("status") != "complete":
        raise ValueError("Incomplete or unsupported snapshot")
    fingerprint = _digest(receipt["inputs"])
    if receipt.get("fingerprint") != fingerprint:
        raise ValueError("Snapshot fingerprint changed")
    required = {"manifest.csv", "audit.json", "sources.json", "review.csv",
                "animal-screening.json", "summary.json"}
    if not required <= receipt.get("files", {}).keys():
        raise ValueError("Incomplete snapshot inventory")
    for name, digest in receipt["files"].items():
        if sha256(_safe_file(root, name)) != digest:
            raise ValueError(f"Snapshot artifact changed: {name}")
    with (root / "manifest.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise ValueError("Invalid snapshot manifest columns")
        rows = list(reader)
    for row in rows:
        if receipt["files"].get(row["path"]) != row["sha256"]:
            raise ValueError("Snapshot manifest/image hash mismatch")
        row["path"] = str(_safe_file(root, row["path"]))
        row["original_test"] = row["original_test"] == "True"
    counts = _validate_rows(rows, receipt["inputs"]["config"]["minimum_photos"])
    summary = json.loads((root / "summary.json").read_text())
    if summary["counts"] != counts:
        raise ValueError("Snapshot counts changed")
    return PreparedDataset(root, fingerprint, rows, summary)


def load_training(*, repo_id, version, revision, expected_fingerprint, cache_dir=None):
    """Resolve one pinned Hub commit and verify its immutable prepared snapshot."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Snapshot revision must be a full commit hash")
    commit = HfApi().dataset_info(repo_id, revision=revision).sha
    if commit != revision:
        raise ValueError("Hugging Face resolved a different snapshot commit")
    destination = Path(cache_dir or Path.home() / ".cache/pettrace/snapshots").resolve()
    root = Path(snapshot_download(
        repo_id, repo_type="dataset", revision=commit, local_dir=str(destination / commit),
        allow_patterns=[f"versions/{version}/**"],
    )) / "versions" / version
    prepared = load_prepared(root)
    if prepared.fingerprint != expected_fingerprint:
        raise ValueError("Snapshot fingerprint does not match the requested release")
    return prepared, {"repo": repo_id, "version": version, "commit": commit,
                      "fingerprint": prepared.fingerprint, "path_in_repo": f"versions/{version}"}
