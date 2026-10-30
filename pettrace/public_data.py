"""Identity-only MPDD/CatIndividualImages import and conservative duplicate audit.

Readers follow WildlifeDatasets 1.0.9: MPDD identity is the filename's integer prefix;
CatIndividualImages identity is its numeric directory. Breed labels are never identities.
"""

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageChops, ImageOps, ImageStat

EXTENSIONS = {".jpg", ".jpeg", ".png"}
TEST_DIRS = {"query", "gallery", "test", "bounding_box_test"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def discover(root, source, species):
    root = Path(root).resolve()
    if source not in {"mpdd", "cat-individuals"}:
        raise ValueError("Use explicit identity metadata for other sources; do not guess labels")
    rows, skipped = [], []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS):
        parts = path.relative_to(root).parts
        if source == "mpdd":
            identity = path.stem.split("_")[0]
        else:
            numeric = [part for part in parts[:-1] if part.isdigit()]
            identity = numeric[0] if len(numeric) == 1 else ""
        if not identity.isdigit():
            skipped.append(str(path))
            continue
        rows.append({"path": str(path), "source": source, "species": species,
                     "animal": f"{source}:{species}:{int(identity)}",
                     "original_test": any(part.lower() in TEST_DIRS for part in parts[:-1])})
    if not rows:
        raise ValueError(f"No individual-ID photos found under {root}; check extracted layout")
    return rows, skipped


def audit_and_split(rows, output, seed=42, near_mae=2.0, near_hash_distance=2, minimum_photos=2):
    if minimum_photos < 2:
        raise ValueError("Identity curation requires at least 2 distinct photos")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["animal"], r["path"]))
    retained, removed, corrupt = [], [], []
    hashes, previews = {}, defaultdict(list)
    original_test_ids = {r["animal"] for r in rows if r.get("original_test")}
    # Detect ID contamination before deduplication; do not silently fix it.
    original_train_ids = {r["animal"] for r in rows if not r.get("original_test") and r["source"] == "mpdd"}
    if original_test_ids & original_train_ids:
        raise ValueError("MPDD identities occur in both original train and test directories")
    for raw in rows:
        row = dict(raw)
        try:
            row["sha256"] = sha256(row["path"])
            with Image.open(row["path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.load()
                thumb = im.resize((64, 64))
                dh = list(im.convert("L").resize((9, 8)).getdata())
                bits = [dh[y * 9 + x] > dh[y * 9 + x + 1] for y in range(8) for x in range(8)]
                dhash = sum(int(value) << bit for bit, value in enumerate(bits))
                ratio = im.width / im.height
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            corrupt.append({"path": row["path"], "error": str(exc)})
            continue
        previous = hashes.get(row["sha256"])
        if previous:
            if previous["animal"] != row["animal"]:
                raise ValueError(f"Same image has conflicting identities: {previous['path']} / {row['path']}")
            removed.append({"path": row["path"], "reason": "exact_duplicate", "retained": previous["path"]})
            continue
        # Keep every exact hash, including near-duplicate rejects, for later label-conflict checks.
        hashes[row["sha256"]] = row
        duplicate = None
        for old, old_hash, old_ratio, old_thumb in previews[row["animal"]]:
            if abs(ratio - old_ratio) < 0.01 and (dhash ^ old_hash).bit_count() <= near_hash_distance:
                error = sum(ImageStat.Stat(ImageChops.difference(thumb, old_thumb)).mean) / 3
                if error <= near_mae:
                    duplicate = old
                    break
        if duplicate:
            removed.append({"path": row["path"], "reason": "near_duplicate", "retained": duplicate["path"]})
            continue
        previews[row["animal"]].append((row, dhash, ratio, thumb))
        retained.append(row)
    counts = Counter(r["animal"] for r in retained)
    retained = [r for r in retained if counts[r["animal"]] >= minimum_photos]
    assigned = {}
    for source in sorted({r["source"] for r in retained}):
        identities = sorted({r["animal"] for r in retained if r["source"] == source})
        rng = random.Random(f"{seed}:{source}")
        fixed_test = set(identities) & original_test_ids
        available = [a for a in identities if a not in fixed_test]
        rng.shuffle(available)
        if fixed_test:
            nval = max(2, round(0.2 * len(available)))
            validation, training, testing = available[:nval], available[nval:], sorted(fixed_test)
        else:
            nval = max(2, round(0.15 * len(available)))
            ntest = max(2, round(0.15 * len(available)))
            validation, testing, training = available[:nval], available[nval:nval + ntest], available[nval + ntest:]
        if len(training) < 2 or len(validation) < 2 or len(testing) < 2:
            raise ValueError(f"Too few distinct {source} identities after curation")
        assigned.update({a: "train" for a in training})
        assigned.update({a: "val" for a in validation})
        assigned.update({a: "test" for a in testing})
    for row in retained:
        row["split"] = assigned[row["animal"]]
    fields = ["path", "source", "animal", "species", "split", "sha256", "original_test"]
    with (output / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(retained)
    summary = {"seed": seed, "raw_photos": len(rows), "retained_photos": len(retained),
               "removed_duplicates": removed, "corrupt": corrupt,
               "minimum_photos": minimum_photos,
               "insufficient_photo_identities_excluded": sorted(a for a, n in counts.items() if n < minimum_photos),
               "single_image_identities_excluded": sorted(a for a, n in counts.items() if n < 2),
               "note": "Conservative duplicate filtering does not establish multi-session diversity.",
               "counts": {f"{sp}.{split}": {"photos": sum(r["species"] == sp and r["split"] == split for r in retained),
                   "identities": len({r["animal"] for r in retained if r["species"] == sp and r["split"] == split})}
                   for sp in sorted({r["species"] for r in retained}) for split in ["train", "val", "test"]}}
    (output / "audit.json").write_text(json.dumps(summary, indent=2))
    return retained, summary


def feature_cache(base_pipeline, rows, output, base_model_dir, batch=16):
    import numpy as np

    from pettrace.evaluation import file_hashes

    # Bind the cache to every image, label/split, spec and exact frozen ONNX artifact.
    key = {"rows": [{k: r[k] for k in ["animal", "species", "split", "sha256"]} for r in rows],
           "models": file_hashes(base_model_dir), "spec": base_pipeline.spec}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    path = Path(output) / f"frozen-e2-{digest[:16]}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["fingerprint"].item()) != digest:
                raise ValueError("Feature cache fingerprint mismatch")
            return cached["features"], digest
    embeddings = []
    for i in range(0, len(rows), batch):
        images = []
        for row in rows[i:i + batch]:
            if sha256(row["path"]) != row["sha256"]:
                raise ValueError(f"Image changed since manifest: {row['path']}")
            with Image.open(row["path"]) as im:
                images.append(ImageOps.exif_transpose(im).convert("RGB"))
        embeddings.append(base_pipeline.embed(images))
        if i % (batch * 20) == 0:
            print(f"Frozen embeddings: {i + len(images)}/{len(rows)}", flush=True)
    features = np.concatenate(embeddings).astype(np.float32)
    if not np.isfinite(features).all() or features.shape != (len(rows), 512):
        raise ValueError("Invalid frozen encoder output")
    np.savez_compressed(path, features=features, fingerprint=np.asarray(digest))
    return features, digest
