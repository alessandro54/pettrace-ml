"""Public body-identity sources and an explicit photo-review boundary for E3."""

import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import requests

from pettrace.data.acquisition import download, extract
from pettrace.data.curation import EXTENSIONS, audit_and_split, sha256

SOURCES = {
    "dogs-world": {"ref": "lextoumbourou/dogs-world", "version": 1, "species": "dog", "license": "CC0 (publisher)"},
    "lcw": {"ref": "dseidli/lcwlabeled-cats-in-the-wild", "version": 1, "species": "cat", "license": "Apache 2.0 (publisher)"},
}


def acquire(root, source, *, allow_download=False):
    """Use an extracted local source, or explicitly download its pinned full archive."""
    spec = SOURCES[source]
    root = Path(root).resolve()
    directory = root / source
    receipt = root / f"{source}-source.json"
    if directory.is_dir() and any(directory.iterdir()):
        # A previous automatic extraction must have completed successfully.
        if receipt.exists():
            metadata = json.loads(receipt.read_text())
            if metadata.get("extraction_pending"):
                raise ValueError(f"Incomplete extraction: {directory}; remove it and retry")
            if metadata.get("version") != spec["version"]:
                raise ValueError(f"Source version mismatch: {source}")
        else:
            metadata = {**spec, "origin": "user-provided extraction; archive version not verified"}
        return directory, metadata
    if not allow_download:
        raise FileNotFoundError(
            f"Extract {spec['ref']} v{spec['version']} under {directory}, "
            "or explicitly enable ALLOW_DATA_DOWNLOAD (full archives: about 96 GB combined publisher volume)."
        )
    root.mkdir(parents=True, exist_ok=True)
    auth = None
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        auth = (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])
    metadata_url = f"https://www.kaggle.com/api/v1/datasets/view/{spec['ref']}"
    response = requests.get(metadata_url, timeout=30, auth=auth)
    response.raise_for_status()
    publisher = response.json()
    if publisher.get("currentVersionNumber") != spec["version"]:
        raise ValueError(f"Publisher version changed for {source}; review source metadata before updating the pin")
    archive = root / f"{source}-v{spec['version']}.zip"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{spec['ref']}?datasetVersionNumber={spec['version']}"
    archive_hash = download(url, archive, auth=auth)
    metadata = {**spec, "archive_sha256": archive_hash, "publisher_license": publisher.get("licenseName"),
                "publisher_bytes": publisher.get("totalBytes"), "url": url, "extraction_pending": True}
    receipt.write_text(json.dumps(metadata, indent=2))
    extract(archive, directory)
    metadata["extraction_pending"] = False
    receipt.write_text(json.dumps(metadata, indent=2))
    return directory, metadata


def discover_body_ids(root, source):
    """Read individual IDs, never breed names; ambiguous dog sidecars are excluded."""
    root = Path(root).resolve()
    species = SOURCES[source]["species"]
    rows, skipped = [], []
    if source == "dogs-world":
        metadata_dirs = sorted(p for p in root.rglob("metadata") if p.is_dir())
        if not metadata_dirs:
            raise ValueError("Dogs World requires metadata/*.json sidecars; image filenames are not identity labels")
        for folder in metadata_dirs:
            for sidecar in sorted(folder.glob("*.json")):
                metadata = json.loads(sidecar.read_text())
                ids = metadata.get("identities", [])
                if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], dict) or not ids[0].get("identity"):
                    skipped.append({"path": str(sidecar), "reason": "ambiguous_or_missing_identity"})
                    continue
                relative = metadata.get("path")
                if not isinstance(relative, str) or not relative:
                    raise ValueError(f"Missing image path: {sidecar}")
                path = (folder.parent / relative).resolve()
                if not path.is_relative_to(folder.parent.resolve()):
                    raise ValueError(f"Image path escapes source root: {sidecar}")
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    skipped.append({"path": str(path), "reason": "missing_or_unsupported_image"})
                    continue
                identity = str(ids[0]["identity"])
                rows.append({"path": str(path), "source": source, "species": species,
                             "animal": f"{source}:{species}:{identity}", "original_test": False})
    else:
        bases = sorted(p for p in root.rglob("CatFullDataset") if p.is_dir())
        if root.name == "CatFullDataset":
            bases = [root]
        if len(bases) != 1:
            raise ValueError("LCW requires one CatFullDataset/<individual-ID>/<image> directory")
        for path in sorted(bases[0].rglob("*")):
            if path.suffix.lower() not in EXTENSIONS or not path.is_file():
                continue
            relative = path.relative_to(bases[0])
            if len(relative.parts) != 2 or not relative.parts[0].isdigit():
                skipped.append({"path": str(path), "reason": "invalid_identity_layout"})
                continue
            rows.append({"path": str(path.resolve()), "source": source, "species": species,
                         "animal": f"{source}:{species}:{int(relative.parts[0])}", "original_test": False})
    if not rows:
        raise ValueError(f"No usable individual-identity images in {root}")
    return rows, skipped


def select_pilot(rows, *, identities_per_species=300, photos_per_identity=10, minimum_photos=4, seed=42):
    """Bound the pilot before image decoding; these are candidate counts, not curated counts."""
    if identities_per_species < 12 or photos_per_identity < minimum_photos or minimum_photos < 2:
        raise ValueError("Pilot requires >=12 identities/species and photos_per_identity >= minimum_photos >=2")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["species"], row["animal"])].append(row)
    rng, selected = random.Random(seed), []
    for species in ("cat", "dog"):
        ids = sorted(k for k, photos in groups.items() if k[0] == species and len(photos) >= minimum_photos)
        rng.shuffle(ids)
        if len(ids) < 12:
            raise ValueError(f"Too few {species} candidate identities with >= {minimum_photos} files")
        for key in ids[:identities_per_species]:
            photos = sorted(groups[key], key=lambda row: row["path"])
            rng.shuffle(photos)
            selected.extend(photos[:photos_per_identity])
    return selected


REVIEW_FIELDS = ["path", "source", "species", "animal", "sha256", "decision", "note"]


def write_review(rows, path):
    """Never overwrite decisions on rerun; bind review to identity, path and image bytes."""
    path = Path(path)
    expected = [{**{k: row[k] for k in REVIEW_FIELDS[:4]}, "sha256": sha256(row["path"]),
                 "decision": "", "note": ""} for row in rows]
    if path.exists():
        with path.open(newline="") as stream:
            existing = list(csv.DictReader(stream))
        if len(existing) != len(expected) or any(
            any(a.get(k) != b[k] for k in REVIEW_FIELDS[:5]) for a, b in zip(existing, expected)):
            raise ValueError("Review input changed; use a new pilot directory, do not reuse old decisions")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(expected)
    return path


def curate_review(path, output, *, seed=42, minimum_photos=4, review_policy="strict", excluded_animals=None,
                  excluded_paths=None):
    """Strict review or explicit exploratory use of unverified publisher identity labels."""
    with Path(path).open(newline="") as stream:
        reviewed = list(csv.DictReader(stream))
    if review_policy not in {"strict", "exploratory"}:
        raise ValueError("review_policy must be strict or exploratory")
    if not reviewed or any(row.get("decision") not in {"", "keep", "drop"} for row in reviewed):
        raise ValueError("Invalid or empty review.csv; decisions must be blank, keep or drop")
    if review_policy == "strict" and any(row["decision"] == "" for row in reviewed):
        raise ValueError("Complete review.csv: every photo requires keep or drop before training")
    excluded_animals = dict(excluded_animals or {})
    excluded_paths = dict(excluded_paths or {})
    rows = []
    for row in reviewed:
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError("Photo changed after review")
        if row["source"] not in SOURCES or row["species"] != SOURCES[row["source"]]["species"]:
            raise ValueError("Invalid source/species in review")
        if not row["animal"].startswith(f"{row['source']}:{row['species']}:"):
            raise ValueError("Invalid namespaced individual ID in review")
        if row["decision"] != "drop" and row["animal"] not in excluded_animals and row["path"] not in excluded_paths:
            rows.append({k: row[k] for k in REVIEW_FIELDS[:4]} | {"original_test": False})
    # In exploratory mode, quarantine entire conflicting IDs instead of guessing a label.
    conflicting = {}
    if review_policy == "exploratory":
        by_hash = {}
        for row in rows:
            by_hash.setdefault(sha256(row["path"]), []).append(row)
        conflicting = {digest: items for digest, items in by_hash.items()
                       if len({item["animal"] for item in items}) > 1}
        conflicting_ids = {row["animal"] for items in conflicting.values() for row in items}
        rows = [row for row in rows if row["animal"] not in conflicting_ids]
    # Duplicate audit precedes the final count requirement and split assignment.
    retained, audit = audit_and_split(rows, output, seed=seed, minimum_photos=minimum_photos)
    if {row["species"] for row in retained} != {"cat", "dog"}:
        raise ValueError("E3 pilot must retain both dogs and cats")
    audit["review_sha256"] = sha256(path)
    audit["review_decisions"] = dict(Counter(row["decision"] for row in reviewed))
    audit["review_policy"] = review_policy
    audit["excluded_animals"] = excluded_animals
    audit["excluded_paths"] = excluded_paths
    audit["excluded_path_photos"] = sum(row["path"] in excluded_paths for row in reviewed)
    audit["excluded_animal_photos"] = sum(row["animal"] in excluded_animals for row in reviewed)
    audit["conflicting_hashes"] = conflicting
    audit["conflicting_identities_quarantined"] = sorted({row["animal"] for items in conflicting.values() for row in items})
    decisions = {row["path"]: row["decision"] for row in reviewed}
    audit["retained_with_unverified_source_label"] = sum(decisions[row["path"]] == "" for row in retained)
    audit["limitation"] = ("Exploratory pilot: pending decisions use publisher identity labels without individual visual "
                           "approval. Technical filtering and a visual sample do not verify all identities or sessions."
                           if review_policy == "exploratory" else "Explicit per-photo review; session diversity not verified.")
    (Path(output) / "audit.json").write_text(json.dumps(audit, indent=2))
    return retained, audit
