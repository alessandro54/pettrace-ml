"""Exact deduplication of recorded source inventories, without image inference or deletion."""

import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from pettrace.data.review import _read
from pettrace.data.sources import SOURCES
from pettrace.data.verification import sha256


REPORT_FILES = ("review.csv", "unique-images.csv", "duplicate-copies.csv", "audit.json", "summary.json")


def restore_lookup_review(drive_root, work_dir, *, percent=100):
    """Recover the raw inventory only; never restore photos or run preparation.

    Uses the latest confirmed preparation backup of the requested source volume.
    A dedicated local destination keeps an active preparation's files untouched.
    """
    from pettrace.data.checkpoints import DriveBackup

    if percent not in (20, 50, 100):
        raise ValueError("Use a source volume of 20, 50 or 100 percent")
    drive_root = Path(drive_root).resolve()
    if not drive_root.is_dir():
        raise FileNotFoundError(f"Mount Drive first: {drive_root}")
    local = Path(work_dir).resolve() / "data-lookup" / "restored"
    member = f"data/source-{percent}-seed42/review.csv"
    backup = DriveBackup({name: local / name for name in ("data", "prepared", "reference")},
                         drive_root / "checkpoints/preparation")
    print(f"Drive: buscando inventario raw del {percent}% en los recibos del respaldo…", flush=True)
    backup.restore(files={member})
    return local / member


def lookup_duplicates(review_csv, output_root):
    """Export one representative per recorded SHA256, retaining every original label.

    Reads only the pre-screening review CSV, not image bytes. The result is a source
    inventory for further review, not a prepared training dataset. No decisions,
    identity merges, species filters, photo minima or split assignments are applied.
    """
    review_csv, output_root = Path(review_csv).resolve(), Path(output_root).resolve()
    inputs = {"review_sha256": sha256(review_csv), "implementation_sha256": sha256(__file__),
              "review_implementation_sha256": sha256(_read.__code__.co_filename),
              "source_species": {name: spec["species"] for name, spec in SOURCES.items()}}
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    destination = output_root / fingerprint
    if destination.exists():
        receipt = json.loads((destination / "lookup.json").read_text())
        if (receipt.get("inputs") != inputs or receipt.get("fingerprint") != fingerprint
                or set(receipt.get("files", {})) != set(REPORT_FILES)
                or any(sha256(destination / name) != receipt["files"][name] for name in REPORT_FILES)):
            raise ValueError("Lookup report changed; preserve it and use a separate output directory")
        return destination

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lookup-", dir=output_root) as temporary:
        folder = Path(temporary) / "report"
        folder.mkdir()
        shutil.copyfile(review_csv, folder / "review.csv")
        if sha256(folder / "review.csv") != inputs["review_sha256"]:
            raise ValueError("Review changed while reading; retry after its writer finishes")
        rows = _read(folder / "review.csv")
        by_hash = defaultdict(list)
        for row in rows:
            digest, source = row["sha256"], row["source"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Review requires a valid SHA256 for every photo")
            if (not row["path"] or source not in SOURCES or row["species"] != SOURCES[source]["species"]
                    or not row["animal"].startswith(f"{source}:{row['species']}:")
                    or not row["animal"].rsplit(":", 1)[-1]):
                raise ValueError("Invalid source, species, individual label or path in review")
            if row["decision"] not in {"", "keep", "drop"}:
                raise ValueError("Invalid recorded review decision")
            by_hash[digest].append(row)

        conflicts, identity_photos = {}, Counter()
        source_files, source_unique = Counter(row["source"] for row in rows), Counter()
        with (folder / "unique-images.csv").open("w", newline="") as unique_stream, \
                (folder / "duplicate-copies.csv").open("w", newline="") as duplicate_stream:
            unique = csv.DictWriter(unique_stream, ["path", "sha256", "source_labels", "species_labels",
                                                    "animal_labels", "status"])
            duplicates = csv.DictWriter(duplicate_stream, ["path", "sha256", "representative_path",
                                                            "source", "species", "animal"])
            unique.writeheader()
            duplicates.writeheader()
            for digest, photos in sorted(by_hash.items()):
                photos.sort(key=lambda row: (row["source"], row["animal"], row["path"]))
                labels = sorted({row["animal"] for row in photos})
                sources = sorted({row["source"] for row in photos})
                identity_photos.update(labels)
                source_unique.update(sources)
                if len(labels) > 1:
                    conflicts[digest] = [{key: row[key] for key in ("path", "source", "species", "animal")}
                                         for row in photos]
                unique.writerow({"path": photos[0]["path"], "sha256": digest,
                                 "source_labels": json.dumps(sources),
                                 "species_labels": json.dumps(sorted({row["species"] for row in photos})),
                                 "animal_labels": json.dumps(labels),
                                 "status": "identity_review_pending" if len(labels) > 1 else "source_label_unverified"})
                for row in photos[1:]:
                    duplicates.writerow({key: row[key] for key in ("path", "sha256", "source", "species", "animal")}
                                        | {"representative_path": photos[0]["path"]})
        conflicted_ids = sorted({row["animal"] for photos in conflicts.values() for row in photos})
        summary = {
            "files": len(rows), "unique_images": len(by_hash), "duplicate_copies": len(rows) - len(by_hash),
            "duplicate_pct": 100 * (len(rows) - len(by_hash)) / len(rows),
            "duplicate_hashes": sum(len(photos) > 1 for photos in by_hash.values()),
            "identity_labels": len(identity_photos), "conflicting_hashes": len(conflicts),
            "conflicting_identity_labels": len(conflicted_ids),
            "identities_by_distinct_photos": dict(sorted(Counter(identity_photos.values()).items())),
            "by_source": {source: {"files": total, "unique_images": source_unique[source],
                                    "within_source_duplicate_copies": total - source_unique[source]}
                          for source, total in sorted(source_files.items())},
            "review_decisions_recorded": dict(Counter(row["decision"] or "pending" for row in rows)),
            "scope": "All rows in the supplied review CSV; no manual or automatic exclusions applied.",
            "limitation": "Recorded SHA256 only; source image bytes, visual similarity and animal identities "
                          "are not revalidated. This inventory has no train/val/test splits and is not training-ready.",
        }
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (folder / "audit.json").write_text(json.dumps({"conflicting_hashes": conflicts,
            "conflicting_identity_labels": conflicted_ids, "decision": "pending"}, indent=2) + "\n")
        receipt = {"fingerprint": fingerprint, "inputs": inputs, "source_review": str(review_csv),
                   "files": {name: sha256(folder / name) for name in REPORT_FILES}}
        (folder / "lookup.json").write_text(json.dumps(receipt, indent=2) + "\n")
        folder.rename(destination)
    return destination
