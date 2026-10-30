"""Prepare a portable, verified public identity dataset independently of E3 training."""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image, ImageDraw, ImageOps, __version__ as pillow_version

from pettrace.e3_data import SOURCES, curate_review, discover_body_ids, select_pilot, write_review
from pettrace.e3_detection import screen_animals
from pettrace.pilot_download import acquire_pilot
from pettrace.pipeline import Detector
from pettrace.public_data import sha256

DETECTOR_SHA256 = "da8f9b9ebea5157cdcefa7bea9fd782249ebdf119f917c07d64e5b41942b9c9d"
BASE_MODEL = "models:/pettrace-embedder/2"
KNOWN_EXCLUSIONS = {"dogs-world:dog:003e295e97401ebf": "Two dogs without a verified target"}
MANIFEST_FIELDS = ["path", "source", "animal", "species", "split", "sha256", "original_test"]


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, default=str, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(_json(value))
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class PrepareConfig:
    data_root: str
    detector_dir: str
    output_root: str | None = None
    source_roots: dict = field(default_factory=dict)
    identities_per_species: int = 300
    photos_per_identity: int = 10
    minimum_photos: int = 4
    seed: int = 42
    metadata_limit: int = 100000
    max_network_bytes: int = 3 * 2**30
    allow_download: bool = False
    review_policy: str = "exploratory"
    excluded_animals: dict = field(default_factory=lambda: dict(KNOWN_EXCLUSIONS))
    confidence: float = 0.25
    nms_iou: float = 0.5
    visual_sample_per_species: int = 4
    expected_detector_sha256: str = DETECTOR_SHA256


@dataclass
class PreparedDataset:
    root: Path
    fingerprint: str
    rows: list[dict]
    summary: dict


def acquire_candidates(config):
    """Only bounded pilot acquisition; manual roots never trigger network access."""
    root = Path(config.data_root).resolve()
    rows, sources, skipped = [], [], []
    if set(config.source_roots) - set(SOURCES):
        raise ValueError("Unknown source root")
    for source in SOURCES:
        supplied = config.source_roots.get(source)
        if supplied:
            source_root = Path(supplied).resolve()
            receipt = {"source": source, "root": str(source_root),
                       "origin": "user-provided extraction; publisher archive version not verified"}
        else:
            receipt_path = root / f"{source}-source.json"
            if not config.allow_download:
                if not receipt_path.exists() or json.loads(receipt_path.read_text()).get("extraction_pending", True):
                    raise ValueError(f"{source}: incomplete/missing pilot; supply a source root or allow bounded downloads")
            source_root, receipt = acquire_pilot(
                root, source, identities=config.identities_per_species, photos_per_identity=config.photos_per_identity,
                minimum_photos=config.minimum_photos, seed=config.seed, metadata_limit=config.metadata_limit,
                max_network_bytes=config.max_network_bytes)
        found, ignored = discover_body_ids(source_root, source)
        rows.extend(found)
        skipped.extend(ignored)
        sources.append(receipt)
    selected = select_pilot(rows, identities_per_species=config.identities_per_species,
                            photos_per_identity=config.photos_per_identity, minimum_photos=config.minimum_photos,
                            seed=config.seed)
    return selected, {"sources": sources, "skipped_paths": skipped, "candidate_files": len(rows)}


def acquire_detector(uri, cache, *, expected_sha256=DETECTOR_SHA256):
    """Read only detector + spec from a pinned MLflow version; no encoder, HF data or publication."""
    import mlflow
    from pettrace.tracking import PRODUCTION_URI

    if not uri.startswith("models:/") or not uri.rsplit("/", 1)[-1].isdigit():
        raise ValueError("Use a pinned models:/name/version detector URI")
    tracking = os.environ.get("MLFLOW_TRACKING_URI", PRODUCTION_URI)
    mlflow.set_tracking_uri(tracking)
    mlflow.set_registry_uri(os.environ.get("MLFLOW_REGISTRY_URI", tracking))
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    key = _digest([uri, tracking, mlflow.get_registry_uri(), expected_sha256])
    destination = cache / key
    if destination.exists():
        receipt = json.loads((destination / "detector-source.json").read_text())
        if (sha256(destination / "detector.onnx") != expected_sha256
                or sha256(destination / "pipeline.yaml") != receipt["pipeline_sha256"]):
            raise ValueError("Cached detector artifacts changed")
        return destination
    name, version = uri[len("models:/"):].rsplit("/", 1)
    source = mlflow.MlflowClient().get_model_version(name, version).source
    with tempfile.TemporaryDirectory(prefix=".detector-", dir=cache) as folder:
        folder = Path(folder)
        for filename in ("detector.onnx", "pipeline.yaml"):
            mlflow.artifacts.download_artifacts(artifact_uri=source.rstrip("/") + "/" + filename, dst_path=str(folder))
        if sha256(folder / "detector.onnx") != expected_sha256:
            raise ValueError("Detector hash does not match the pinned E2 artifact")
        (folder / "detector-source.json").write_text(_json(
            {"model_uri": uri, "detector_sha256": expected_sha256, "pipeline_sha256": sha256(folder / "pipeline.yaml")}))
        folder.replace(destination)
    return destination


def _safe_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe prepared-dataset path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing/unsafe prepared-dataset file: {relative}")
    return path


def _validate_rows(rows, minimum_photos):
    groups = defaultdict(list)
    for row in rows:
        if row["species"] not in {"cat", "dog"} or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Invalid prepared species/split")
        if row["source"] not in SOURCES or row["species"] != SOURCES[row["source"]]["species"]:
            raise ValueError("Invalid prepared source label")
        if not row["animal"].startswith(f"{row['source']}:{row['species']}:"):
            raise ValueError("Invalid prepared individual ID")
        groups[row["animal"]].append(row)
    if len({row["path"] for row in rows}) != len(rows) or len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Prepared dataset repeats a path or image hash")
    if any(len(photos) < minimum_photos or len({r["split"] for r in photos}) != 1 for photos in groups.values()):
        raise ValueError("Insufficient distinct photos or identity leakage between splits")
    counts = {}
    for species in ("cat", "dog"):
        for split, required in (("train", 8), ("val", 2), ("test", 2)):
            photos = [row for row in rows if row["species"] == species and row["split"] == split]
            count = {"identities": len({row["animal"] for row in photos}), "photos": len(photos)}
            if count["identities"] < required:
                raise ValueError(f"Too few {species}.{split} identities after filtering; need {required}")
            counts[f"{species}.{split}"] = count
    return counts


def load_prepared(path):
    """Load only a completed snapshot, checking every artifact/image and split before training."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError("Prepared dataset missing: run datasets/prepare_training.ipynb or pettrace-prepare-e3 first")
    expected = None
    if path.is_file() and path.name == "latest.json":
        pointer = json.loads(path.read_text())
        expected = pointer["fingerprint"]
        path = _safe_file(path.parent, Path(pointer["path"]) / "prepared.json").parent
    elif path.is_file():
        path = path.parent
    receipt = json.loads((path / "prepared.json").read_text())
    if receipt.get("schema") != 1 or receipt.get("status") != "complete":
        raise ValueError("Incomplete or unsupported prepared dataset")
    fingerprint = _digest(receipt["inputs"])
    if receipt["fingerprint"] != fingerprint or (expected is not None and expected != fingerprint):
        raise ValueError("Prepared dataset fingerprint changed")
    for required in ("manifest.csv", "audit.json", "sources.json", "review.csv", "animal-screening.json", "summary.json"):
        if required not in receipt["files"]:
            raise ValueError(f"Incomplete prepared inventory: {required}")
    for name, digest in receipt["files"].items():
        if sha256(_safe_file(path, name)) != digest:
            raise ValueError(f"Prepared artifact changed: {name}")
    with (path / "manifest.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise ValueError("Invalid prepared manifest columns")
        rows = list(reader)
    for row in rows:
        if receipt["files"].get(row["path"]) != row["sha256"]:
            raise ValueError("Prepared manifest/image hash mismatch")
        row["path"] = str(_safe_file(path, row["path"]))
        row["original_test"] = row["original_test"] == "True"
    counts = _validate_rows(rows, receipt["inputs"]["config"]["minimum_photos"])
    summary = json.loads((path / "summary.json").read_text())
    if summary["counts"] != counts:
        raise ValueError("Prepared dataset counts changed")
    return PreparedDataset(path, fingerprint, rows, summary)


def _samples(rows, root, *, per_species, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["animal"]].append(row)
    selected, records = [], []
    for species in ("cat", "dog"):
        candidates = sorted(animal for animal, photos in groups.items() if photos[0]["species"] == species)
        random.Random(f"{seed}:{species}:visual-sample").shuffle(candidates)
        selected.extend(candidates[:per_species])
    directory = root / "samples"
    directory.mkdir()
    for number, animal in enumerate(selected):
        photos = groups[animal]
        canvas = Image.new("RGB", (960, 35 + ((len(photos) + 3) // 4) * 265), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), animal, fill="black")
        for index, row in enumerate(photos):
            with Image.open(root / row["path"]) as image:
                thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (230, 230))
            x, y = index % 4 * 240, 35 + index // 4 * 265
            canvas.paste(thumb, (x, y))
            draw.text((x + 3, y + 233), f"{index + 1}: {row['sha256'][:20]}", fill="black")
        filename = f"samples/{number + 1:02d}.jpg"
        canvas.save(root / filename, quality=85)
        records.append({"animal": animal, "species": photos[0]["species"], "sheet": filename,
                        "photos": [row["path"] for row in photos]})
    (root / "visual-sample.json").write_text(_json(
        {"seed": seed, "per_species": per_species, "sample": records, "scope": "Visual sample; identity not verified."}))


def prepare(config):
    """Acquire → verify → YOLO → curate → copy portable images → publish a completed dataset locally."""
    if config.minimum_photos < 4 or config.visual_sample_per_species < 1:
        raise ValueError("E3 requires at least four distinct photos and a positive visual sample size")
    model_root = Path(config.detector_dir).resolve()
    spec = yaml.safe_load((model_root / "pipeline.yaml").read_text())
    steps = [step for step in spec["steps"] if step["op"] == "detect_crop" and step.get("enabled")]
    if len(steps) != 1:
        raise ValueError("Expected one enabled E2 detector step")
    step = steps[0]
    model_hash = sha256(_safe_file(model_root, step["file"]))
    if model_hash != config.expected_detector_sha256:
        raise ValueError("Detector hash does not match the configured frozen artifact")
    selected, sources = acquire_candidates(config)
    review_path = Path(config.data_root).resolve() / f"pilot-seed{config.seed}" / "review.csv"
    write_review(selected, review_path)
    with review_path.open(newline="") as stream:
        review = list(csv.DictReader(stream))
    settings = asdict(config)
    settings.pop("output_root")
    implementation = {name: sha256(Path(__file__).with_name(name)) for name in
                      ("e3_prepare.py", "e3_data.py", "e3_detection.py", "public_data.py", "pilot_download.py", "pipeline.py")}
    inputs = {"config": settings, "detector_step": step, "implementation": implementation,
              "runtime": {"numpy": np.__version__, "onnxruntime": ort.__version__, "pillow": pillow_version},
              "detector_spec_sha256": sha256(model_root / "pipeline.yaml"),
              "review_sha256": sha256(review_path), "photos": [{k: r[k] for k in ("path", "source", "species", "animal", "sha256")} for r in review],
              "sources": sources}
    fingerprint = _digest(inputs)
    output_root = Path(config.output_root or Path(config.data_root) / "prepared").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / fingerprint
    if destination.exists():
        result = load_prepared(destination)
        _atomic_json(output_root / "latest.json", {"fingerprint": fingerprint, "path": destination.name})
        return result
    detector = Detector(str(model_root), step)
    if sha256(_safe_file(model_root, step["file"])) != model_hash:
        raise ValueError("Detector changed while loading")
    exclusions, screening = screen_animals(
        selected, detector, Path(config.data_root) / "animal-screening-cache", detector_sha256=model_hash,
        confidence=config.confidence, iou=config.nms_iou)
    # Detect concurrent review edits before applying filtering decisions.
    if sha256(review_path) != inputs["review_sha256"]:
        raise ValueError("Review changed during preparation; rerun to use the new decisions")
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=output_root) as folder:
        folder = Path(folder)
        retained, audit = curate_review(review_path, folder, seed=config.seed, minimum_photos=config.minimum_photos,
                                        review_policy=config.review_policy, excluded_animals=config.excluded_animals,
                                        excluded_paths=exclusions)
        counts = _validate_rows(retained, config.minimum_photos)
        materialized = []
        for row in retained:
            identity = hashlib.sha256(row["animal"].encode()).hexdigest()
            relative = Path("images") / row["species"] / identity / (row["sha256"] + Path(row["path"]).suffix.lower())
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["path"], target)
            if sha256(target) != row["sha256"]:
                raise ValueError("Source photo changed while copying the prepared dataset")
            materialized.append({**row, "path": relative.as_posix()})
        with (folder / "manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(materialized)
        retained_paths = {row["path"] for row in retained}
        duplicate_paths = {row["path"]: row["reason"] for row in audit["removed_duplicates"]}
        conflicts = set(audit["conflicting_identities_quarantined"])
        excluded_rows = []
        for row in review:
            if row["path"] in retained_paths:
                continue
            reason = ("manual_drop" if row["decision"] == "drop" else
                      "known_ambiguous_identity" if row["animal"] in config.excluded_animals else
                      exclusions.get(row["path"]) or
                      ("conflicting_identity" if row["animal"] in conflicts else
                       duplicate_paths.get(row["path"], "insufficient_distinct_photos")))
            excluded_rows.append({"path": row["path"], "animal": row["animal"], "species": row["species"], "reason": reason})
        with (folder / "excluded.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, ["path", "animal", "species", "reason"])
            writer.writeheader()
            writer.writerows(excluded_rows)
        summary = {"fingerprint": fingerprint, "candidate_photos": len(selected), "retained_photos": len(retained),
                   "candidate_identities": len({r["animal"] for r in selected}),
                   "retained_identities": len({r["animal"] for r in retained}), "counts": counts,
                   "exclusion_reasons": dict(Counter(row["reason"] for row in excluded_rows)),
                   "review_policy": config.review_policy, "limitation": audit["limitation"] + " " + screening["limitation"]}
        (folder / "summary.json").write_text(_json(summary))
        (folder / "sources.json").write_text(_json(sources))
        (folder / "animal-screening.json").write_text(_json(screening))
        shutil.copy2(review_path, folder / "review.csv")
        if sha256(folder / "review.csv") != inputs["review_sha256"]:
            raise ValueError("Review changed during preparation; rerun")
        _samples(materialized, folder, per_species=config.visual_sample_per_species, seed=config.seed)
        (folder / "README.md").write_text(
            f"# E3 · Datos preparados\n\nFingerprint: `{fingerprint}`\n\n"
            f"{len(retained)} fotos de {summary['retained_identities']} identidades, de {len(selected)} candidatas.\n\n"
            "`manifest.csv` referencia imágenes locales relativas. `excluded.csv` registra cada descarte. "
            "`summary.json`, `audit.json` y `animal-screening.json` contienen conteos, decisiones y detecciones. "
            "`samples/` contiene la muestra visual. Cargar con `load_prepared` antes de entrenar.\n\n" + summary["limitation"] + "\n")
        inventory = {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob("*")) if p.is_file()}
        (folder / "prepared.json").write_text(_json(
            {"schema": 1, "status": "complete", "fingerprint": fingerprint, "inputs": inputs, "files": inventory}))
        load_prepared(folder)
        folder.replace(destination)
    _atomic_json(output_root / "latest.json", {"fingerprint": fingerprint, "path": destination.name})
    return load_prepared(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="datasets/public/e3-body")
    parser.add_argument("--output-root")
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--detector-dir")
    model.add_argument("--detector-uri", default=BASE_MODEL)
    parser.add_argument("--dog-root")
    parser.add_argument("--cat-root")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--identities-per-species", type=int, default=300)
    parser.add_argument("--photos-per-identity", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    detector_dir = args.detector_dir or acquire_detector(args.detector_uri, Path(args.data_root) / "detector-models")
    sources = {name: root for name, root in (("dogs-world", args.dog_root), ("lcw", args.cat_root)) if root}
    result = prepare(PrepareConfig(data_root=args.data_root, detector_dir=str(detector_dir), output_root=args.output_root,
                                   source_roots=sources, allow_download=args.allow_download,
                                   identities_per_species=args.identities_per_species,
                                   photos_per_identity=args.photos_per_identity, seed=args.seed))
    print(_json({"path": str(result.root), "fingerprint": result.fingerprint, "summary": result.summary}))


if __name__ == "__main__":
    main()
