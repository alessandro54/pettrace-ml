"""Exact deduplication of recorded source inventories, without image inference or deletion."""

import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from pettrace.data.review import _read
from pettrace.data.sources import REVIEW_FIELDS, SOURCES
from pettrace.data.verification import sha256


REPORT_FILES = ("review.csv", "unique-images.csv", "duplicate-copies.csv", "audit.json", "summary.json",
                "candidates.csv", "excluded.csv")


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


def lookup_duplicates(review_csv, output_root, *, minimum_photos=4, near_cache=None,
                      drive_root=None, source_root="/content/datasets/public/e3-body"):
    """Export one representative per recorded SHA256, retaining every original label.

    By default reads only the pre-screening review CSV, not image bytes. The result is a source
    inventory for further review, not a prepared training dataset. Separately index
    pre-YOLO candidates after manual drops, identity quarantine, exact deduplication
    and a distinct-photo minimum. Never merge labels or assign splits. Supplying a
    local near_cache also checks resized/recompressed copies before the minimum;
    whole involved labels stay quarantined, including same-label matches. Optional
    Drive recovery saves signatures incrementally and reads only uncached originals.
    """
    review_csv, output_root = Path(review_csv).resolve(), Path(output_root).resolve()
    if not isinstance(minimum_photos, int) or minimum_photos < 2:
        raise ValueError("Use at least two distinct photos per identity")
    if drive_root is not None and near_cache is None:
        raise ValueError("Provide a local near_cache when using Drive for visual comparison")
    inputs = {"review_sha256": sha256(review_csv), "implementation_sha256": sha256(__file__),
              "review_implementation_sha256": sha256(_read.__code__.co_filename),
              "minimum_photos": minimum_photos,
              "source_species": {name: spec["species"] for name, spec in SOURCES.items()}}
    if near_cache is not None:
        from pettrace.data.near_copies import configuration, find_near_copies
        inputs["near_copies"] = configuration()
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
        near_pairs, corrupt = [], {}
        if near_cache is not None:
            near_pairs, corrupt = find_near_copies(by_hash, near_cache, drive_root=drive_root, source_root=source_root)
        near_hashes = {pair[side] for pair in near_pairs for side in ("left", "right")}
        near_ids = {row["animal"] for digest in near_hashes for row in by_hash[digest]}
        conflicted_ids = sorted({row["animal"] for photos in conflicts.values() for row in photos})
        quarantined = set(conflicted_ids)
        candidates, excluded = [], {}
        for photos in by_hash.values():
            eligible = []
            for row in photos:
                if row["decision"] == "drop":
                    excluded[row["path"]] = "manual_drop"
                elif row["animal"] in quarantined:
                    excluded[row["path"]] = "identity_review_pending"
                elif row["animal"] in near_ids:
                    excluded[row["path"]] = "near_copy_review_pending"
                elif row["sha256"] in corrupt:
                    excluded[row["path"]] = "decode_error"
                else:
                    eligible.append(row)
            if eligible:
                # Prefer an explicit keep; neither a hash nor a pending label proves identity.
                eligible.sort(key=lambda row: (row["decision"] != "keep", row["path"]))
                candidates.append(eligible[0])
                excluded.update({row["path"]: "exact_duplicate" for row in eligible[1:]})
        counts = Counter(row["animal"] for row in candidates)
        excluded.update({row["path"]: "insufficient_distinct_photos" for row in candidates
                         if counts[row["animal"]] < minimum_photos})
        candidates = sorted((row for row in candidates if row["path"] not in excluded),
                            key=lambda row: (row["animal"], row["sha256"]))
        with (folder / "candidates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(candidates)
        with (folder / "excluded.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, REVIEW_FIELDS + ["reason"])
            writer.writeheader()
            writer.writerows({**row, "reason": excluded[row["path"]]} for row in rows if row["path"] in excluded)
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
            "candidate_photos": len(candidates),
            "candidate_identities": len({row["animal"] for row in candidates}),
            "candidate_unverified_labels": sum(row["decision"] != "keep" for row in candidates),
            "minimum_photos": minimum_photos,
            "pre_yolo_exclusions": dict(Counter(excluded.values())),
            "scope": "Duplicate statistics cover all input rows. candidates.csv is a separate pre-YOLO selection.",
            "limitation": "Recorded SHA256 only; source image bytes, visual similarity and animal identities "
                          "are not revalidated. This inventory has no train/val/test splits and is not training-ready.",
        }
        if near_cache is not None:
            summary.update(near_copy_pairs=len(near_pairs), near_copy_images=len(near_hashes),
                           near_copy_identity_labels=len(near_ids), unreadable_unique_images=len(corrupt),
                           limitation="Conservative resized/recompressed copy proposals, not animal recognition. "
                           "Whole involved labels remain pending review, never merged. Cached signatures refer to "
                           "previously verified bytes; preparation must verify originals again. No splits assigned.")
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        audit = {"conflicting_hashes": conflicts, "conflicting_identity_labels": conflicted_ids, "decision": "pending"}
        if near_cache is not None:
            audit.update(near_copy_pairs=near_pairs, near_copy_identity_labels=sorted(near_ids),
                         near_copy_labels={digest: sorted({row["animal"] for row in by_hash[digest]})
                                           for digest in sorted(near_hashes)},
                         unreadable_images=corrupt, near_copy_method=inputs["near_copies"])
        (folder / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        receipt = {"fingerprint": fingerprint, "inputs": inputs, "source_review": str(review_csv),
                   "files": {name: sha256(folder / name) for name in REPORT_FILES}}
        (folder / "lookup.json").write_text(json.dumps(receipt, indent=2) + "\n")
        folder.rename(destination)
    return destination


def load_candidates(report, review_csv, *, minimum_photos=4):
    """Verify a pre-YOLO index against the full current review, without opening photos."""
    report = Path(report)
    receipt = json.loads((report / "lookup.json").read_text())
    if (receipt.get("fingerprint") != hashlib.sha256(json.dumps(receipt["inputs"], sort_keys=True).encode()).hexdigest()
            or receipt["inputs"].get("minimum_photos") != minimum_photos
            or receipt["inputs"]["review_sha256"] != sha256(review_csv)
            or set(receipt.get("files", {})) != set(REPORT_FILES)
            or any(sha256(report / name) != receipt["files"][name] for name in REPORT_FILES)):
        raise ValueError("Candidate index changed or does not match the review/minimum; regenerate data_lookup")
    with (report / "candidates.csv").open(newline="") as stream:
        candidates = list(csv.DictReader(stream))
    with (report / "excluded.csv").open(newline="") as stream:
        excluded = list(csv.DictReader(stream))
    original = {row["path"]: row for row in _read(review_csv)}
    combined = candidates + excluded
    if (len(combined) != len(original) or {row["path"] for row in combined} != original.keys()
            or any({key: row[key] for key in REVIEW_FIELDS} != original[row["path"]] for row in combined)
            or len({row["sha256"] for row in candidates}) != len(candidates)
            or any(row["decision"] == "drop" for row in candidates)
            or any(n < minimum_photos for n in Counter(row["animal"] for row in candidates).values())):
        raise ValueError("Candidate index does not partition the original inventory")
    return candidates, {row["path"]: row["reason"] for row in excluded}
