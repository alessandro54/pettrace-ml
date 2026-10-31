"""Nested source-photo acquisition; filtering remains in preparation.prepare."""
import hashlib
import json
import os
from pathlib import Path
import zipfile
import zlib

from pettrace.data.curation import EXTENSIONS, sha256
from pettrace.data.pilot import RangeFile, _payload, _photo_windows, _safe_name
from pettrace.data.sources import SOURCES

CAPACITY = json.loads(Path(__file__).with_name("source_capacity_v1.json").read_text())


def photo_sample(infos, source, percent, seed):
    """One stable full-index ordering gives nested 20/50/100 samples before filtering."""
    if source not in CAPACITY or percent not in (20, 50, 100):
        raise ValueError("Use a known source and 20, 50 or 100 percent")
    prefix = "images/" if source == "dogs-world" else "CatFullDataset/"
    photos = [i for i in infos if i.filename.startswith(prefix) and Path(i.filename).suffix.lower() in EXTENSIONS]
    if len({i.filename for i in photos}) != len(photos) or len(photos) != CAPACITY[source]["photos"]:
        raise ValueError("Source photo inventory differs from the pinned capacity")
    photos.sort(key=lambda i: (hashlib.sha256(f"{seed}:{source}:{i.filename}".encode()).digest(), i.filename))
    return photos[:len(photos) * percent // 100]


def acquire_volume(root, source, *, percent, seed=42, allow_download=False, workers=4, max_network_bytes=3 << 30):
    """Reuse range/CRC download machinery; never select an identity pilot or metadata prefix."""
    root = Path(root).resolve()
    directory = root / "source-cache" / source
    receipt_path = root / f"{source}-{percent}-seed{seed}.json"
    config = {"source_percent": percent, "seed": seed, "selection": "sha256_order_full_archive_photos_v1"}
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if saved.get("config") != config:
            raise ValueError("Source sampling configuration changed")
        if not saved.get("extraction_pending", True):
            for name, digest in saved["file_sha256"].items():
                target = directory / _safe_name(name)
                if not target.is_file() or sha256(target) != digest:
                    raise ValueError("Cached source photo/metadata changed")
            return directory, saved
    if not allow_download:
        raise ValueError("Source volume is not acquired; enable downloads in the separate preparation workflow")
    if not 1 <= workers <= 16:
        raise ValueError("Use 1 to 16 download workers")
    root.mkdir(parents=True, exist_ok=True)
    spec = SOURCES[source]
    receipt = {**spec, "mode": "source-photo-volume", "config": config, "extraction_pending": True}
    receipt_path.write_text(json.dumps(receipt, indent=2))
    auth = ((os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])
            if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY") else None)
    url = f"https://www.kaggle.com/api/v1/datasets/download/{spec['ref']}?datasetVersionNumber={spec['version']}"
    with RangeFile(url, auth=auth, max_network_bytes=max_network_bytes) as remote, zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        signature = hashlib.sha256()
        for info in infos:
            signature.update(json.dumps([info.filename, info.CRC, info.file_size, info.compress_size,
                                         info.header_offset], separators=(",", ":")).encode() + b"\n")
        if signature.hexdigest() != CAPACITY[source]["archive_index_sha256"]:
            raise ValueError("Source archive index changed")
        selected = photo_sample(infos, source, percent, seed)
        names = {i.filename for i in selected}
        hashes = {}
        # Scan the complete metadata inventory. A missing/ambiguous label is a later exclusion,
        # never a reason to replace a randomly selected photograph with a different one.
        if source == "dogs-world":
            for info in sorted(infos, key=lambda i: i.header_offset):
                if Path(info.filename).parent.name != "metadata" or not info.filename.endswith(".json"):
                    continue
                target = directory / _safe_name(info.filename)
                raw = target.read_bytes() if target.is_file() else b""
                if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
                    raw = _payload(archive, info, maximum=2 << 20)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                metadata = json.loads(raw)
                relative = metadata.get("path")
                if isinstance(relative, str):
                    image = str(Path(info.filename).parent.parent / _safe_name(relative))
                    if image in names:
                        hashes[info.filename] = hashlib.sha256(raw).hexdigest()
        selected.sort(key=lambda i: i.header_offset)
        missing = [i for i in selected if not (directory / i.filename).is_file()
                   or (directory / i.filename).stat().st_size != i.file_size
                   or zlib.crc32((directory / i.filename).read_bytes()) != i.CRC]
        windows = _photo_windows(remote, missing, workers)
        try:
            for number, info in enumerate(selected, 1):
                target = directory / _safe_name(info.filename)
                if not target.is_file() or target.stat().st_size != info.file_size or zlib.crc32(target.read_bytes()) != info.CRC:
                    assert next(windows).filename == info.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_suffix(target.suffix + ".partial")
                    partial.write_bytes(_payload(archive, info))
                    partial.replace(target)
                hashes[info.filename] = sha256(target)
                if number % 1000 == 0 or number == len(selected):
                    print(f"{source}: {number}/{len(selected)} photos verified", flush=True)
        finally:
            windows.close()
        receipt.update(archive_index_sha256=signature.hexdigest(), file_sha256=hashes,
                       source_photos=CAPACITY[source]["photos"], sampled_photos=len(selected),
                       network_bytes=remote.transferred, extraction_pending=False)
    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2))
    temporary.replace(receipt_path)
    return directory, receipt


def source_sample_metrics(sources, capacities, requested_pct):
    if requested_pct not in {20, 50, 100}:
        raise ValueError("Expected a 20, 50 or 100 percent source sample")
    receipts = {source["ref"]: source for source in sources}
    if len(receipts) != len(sources) or set(receipts) != {r["ref"] for r in capacities.values()}:
        raise ValueError("Source sample requires exactly one receipt per reference source")
    metrics = {"dataset.sample_requested_pct": requested_pct}
    total_sampled = 0
    for source, reference in capacities.items():
        receipt = receipts.get(reference["ref"], {})
        if (receipt.get("version") != reference["version"] or
                receipt.get("archive_index_sha256") != reference["archive_index_sha256"] or
                receipt.get("extraction_pending", True)):
            raise ValueError(f"{source}: require a completed acquisition with the verified source index")
        files = receipt.get("file_sha256", {})
        prefix = "images/" if source == "dogs-world" else "CatFullDataset/"
        photos = {name: digest for name, digest in files.items()
                  if name.startswith(prefix) and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}}
        if any(len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest) for digest in photos.values()):
            raise ValueError(f"{source}: invalid acquisition image hash")
        target = reference["photos"] * requested_pct // 100
        if len(photos) != target:
            raise ValueError(f"{source}: {len(photos):,} source photos acquired; {requested_pct}% requires {target:,}. "
                             "Prepare and version that source sample first; the v1 pilot is not a 20/50/100% sample.")
        prefix = f"dataset.{reference['species']}"
        metrics[f"{prefix}.sampled_photos"] = len(photos)
        metrics[f"{prefix}.sampled_pct"] = 100 * len(photos) / reference["photos"]
        total_sampled += len(photos)
    metrics["dataset.sampled_photos"] = total_sampled
    metrics["dataset.sampled_pct"] = 100 * total_sampled / sum(r["photos"] for r in capacities.values())
    return metrics


def dataset_coverage(rows, sources, capacities):
    receipts = {source["ref"]: source for source in sources}
    if {row["source"] for row in rows} != set(capacities):
        raise ValueError("Coverage reference must match the dataset sources")
    metrics, params = {}, {"dataset.reference.unit": "photo_entries",
                           "dataset.reference.scope": "original_archives_before_filtering"}
    groups = [("dataset", rows, sum(c["photos"] for c in capacities.values()))]
    for source, reference in capacities.items():
        receipt = receipts.get(reference["ref"], {})
        if (receipt.get("version") != reference["version"] or
                receipt.get("archive_index_sha256") != reference["archive_index_sha256"]):
            raise ValueError(f"Unverified source capacity: {source}")
        if not isinstance(reference["photos"], int) or reference["photos"] <= 0:
            raise ValueError("Source capacity must be a positive photo count")
        source_rows = [row for row in rows if row["source"] == source]
        if any(row["species"] != reference["species"] for row in source_rows):
            raise ValueError("Source species mismatch")
        prefix = f"dataset.{reference['species']}"
        groups.append((prefix, source_rows, reference["photos"]))
        for key in ("ref", "version", "archive_index_sha256"):
            params[f"{prefix}.reference.{key}"] = str(reference[key])
    for prefix, subset, maximum in groups:
        if len(subset) > maximum or any(row["split"] not in {"train", "val", "test"} for row in subset):
            raise ValueError("Invalid coverage numerator")
        metrics[f"{prefix}.source_photos"] = maximum
        metrics[f"{prefix}.retained_identities"] = len({row["animal"] for row in subset})
        for label, splits in {"train": {"train"}, "val": {"val"}, "test_reserved": {"test"},
                              "train_val": {"train", "val"}, "retained": {"train", "val", "test"}}.items():
            selected = [row for row in subset if row["split"] in splits]
            metrics[f"{prefix}.{label}_photos"] = len(selected)
            metrics[f"{prefix}.{label}_pct"] = 100.0 * len(selected) / maximum
            metrics[f"{prefix}.{label}_identities"] = len({row["animal"] for row in selected})
    return metrics, params


def training_volume(prepared, percent, report):
    """Require completed processing evidence, report acquisition and effective train separately."""
    sources = json.loads((prepared.root / "sources.json").read_text())["sources"]
    metrics, params = dataset_coverage(prepared.rows, sources, CAPACITY)
    metrics.update(source_sample_metrics(sources, CAPACITY, percent))
    processing = prepared.summary.get("processing", {})
    if set(processing) != set(CAPACITY):
        raise ValueError("Prepared dataset lacks completed source-volume processing; rerun preparation")
    for source, reference in CAPACITY.items():
        item = processing[source]
        prefix = f"dataset.{reference['species']}"
        if (item["requested_pct"] != percent or item["processed_photos"] != metrics[prefix+".sampled_photos"]
                or item["train_photos"] != metrics[prefix+".train_photos"]
                or item["excluded_photos"] + item["train_photos"] != item["processed_photos"]
                or sum(item["exclusion_reasons"].values()) != item["excluded_photos"]):
            raise ValueError("Incomplete or inconsistent processing counts")
        metrics[prefix+".processed_photos"] = item["processed_photos"]
        metrics[prefix+".processed_pct"] = 100 * item["processed_photos"] / reference["photos"]
        metrics[prefix+".excluded_photos"] = item["excluded_photos"]
        acquired = next(r["file_sha256"] for r in sources if r["ref"] == reference["ref"])
        hashes = set(acquired.values())
        if any(r["sha256"] not in hashes for r in prepared.rows if r["source"] == source and r["split"] == "train"):
            raise ValueError("Training images are outside the acquired sample")
    for key in ("processed_photos", "excluded_photos"):
        metrics[f"dataset.{key}"] = sum(metrics[f"dataset.{sp}.{key}"] for sp in ("cat", "dog"))
    metrics["dataset.processed_pct"] = 100 * metrics["dataset.processed_photos"] / metrics["dataset.source_photos"]
    params.update({"dataset.sample_basis": "original_source_photo_entries_before_cleaning",
                   "dataset.sample_rounding": "floor_per_source", "dataset.sample_requested_pct": str(percent)})
    (Path(report) / "dataset-coverage.json").write_text(json.dumps(
        {"metrics": metrics, "params": params, "reference": CAPACITY, "processing": processing}, indent=2))
    return metrics, params
