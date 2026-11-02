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

from pettrace.data.sources import SOURCES, curate_review, discover_body_ids, select_pilot, write_review
from pettrace.data.screening import screen_animals
from pettrace.data.pilot import acquire_pilot
from pettrace.models.pipeline import Detector
from pettrace.data.curation import sha256
from pettrace.data.verification import verification_session

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
    source_archives: dict = field(default_factory=dict)
    source_percent: int | None = None
    reference_dataset: str | None = None
    identities_per_species: int = 300
    photos_per_identity: int = 10
    minimum_photos: int = 4
    seed: int = 42
    metadata_limit: int = 100000
    max_network_bytes: int = 3 * 2**30
    allow_download: bool = False
    download_workers: int = 4
    review_policy: str = "exploratory"
    excluded_animals: dict = field(default_factory=lambda: dict(KNOWN_EXCLUSIONS))
    confidence: float = 0.25
    nms_iou: float = 0.5
    visual_sample_per_species: int = 4
    expected_detector_sha256: str = DETECTOR_SHA256
    screening_provider: str = "cpu"


@dataclass
class PreparedDataset:
    root: Path
    fingerprint: str
    rows: list[dict]
    summary: dict


def acquire_candidates(config, *, checkpoint=None):
    """Only bounded pilot acquisition; manual roots never trigger network access."""
    root = Path(config.data_root).resolve()
    rows, sources, skipped = [], [], []
    if (set(config.source_roots) | set(config.source_archives)) - set(SOURCES):
        raise ValueError("Unknown source root")
    for source in SOURCES:
        if config.source_percent is not None:
            from pettrace.data.volume import acquire_volume
            if config.source_roots:
                raise ValueError("Source-volume acquisition requires verified archive receipts, not manual roots")
            source_root, receipt = acquire_volume(
                root, source, percent=config.source_percent, seed=config.seed, allow_download=config.allow_download,
                workers=config.download_workers, max_network_bytes=config.max_network_bytes,
                archive_path=config.source_archives.get(source), checkpoint=checkpoint)
            found, ignored = discover_body_ids(source_root, source)
            chosen = receipt["file_sha256"]
            found = [r for r in found if Path(r["path"]).relative_to(source_root).as_posix() in chosen]
            by_path = defaultdict(list)
            for row in found:
                by_path[row["path"]].append(row)
            found = [items[0] for items in by_path.values() if len({r["animal"] for r in items}) == 1]
            # Discovery can include metadata for cached photographs from a larger volume.
            labelled = {r["path"] for r in found}
            for name in chosen:
                if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"} and str(source_root / name) not in labelled:
                    skipped.append({"path": str(source_root / name), "source": source, "reason": "missing_or_ambiguous_identity"})
            rows.extend(found)
            sources.append(receipt)
            continue
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
                max_network_bytes=config.max_network_bytes, workers=config.download_workers)
        found, ignored = discover_body_ids(source_root, source)
        rows.extend(found)
        skipped.extend(ignored)
        sources.append(receipt)
    if config.source_percent is not None:
        return rows, {"sources": sources, "skipped_paths": skipped, "candidate_files": len(rows)}
    selected = select_pilot(rows, identities_per_species=config.identities_per_species,
                            photos_per_identity=config.photos_per_identity, minimum_photos=config.minimum_photos,
                            seed=config.seed)
    return selected, {"sources": sources, "skipped_paths": skipped, "candidate_files": len(rows)}


class DetectorAuthenticationError(ValueError):
    """A recoverable client-authentication error; never includes entered credentials."""

    def _render_traceback_(self):
        return [str(self)]


def acquire_detector(uri, cache, *, expected_sha256=DETECTOR_SHA256):
    """Read only detector + spec from a pinned MLflow version; no encoder, HF data or publication."""
    import mlflow
    from mlflow.exceptions import MlflowException
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
    try:
        source = mlflow.MlflowClient().get_model_version(name, version).source
    except MlflowException as exc:
        if exc.get_http_status_code() == 401:
            raise DetectorAuthenticationError(
                "MLflow rechazó las credenciales (401). Actualiza MLFLOW_TRACKING_USERNAME y "
                "MLFLOW_TRACKING_PASSWORD en esta sesión con configure_access(force=True), "
                "o verifica el usuario del servidor. La descarga de fotos se conserva."
            ) from None
        raise
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
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe prepared-dataset file: {relative}")
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared-dataset file: {relative}")
    if not path.is_file():
        raise ValueError(f"Expected a regular prepared-dataset file: {relative}")
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
        photos = groups[animal][:8]
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


@verification_session()
def prepare(config, *, checkpoint=None):
    """Acquire → verify → YOLO → curate → copy portable images → publish a completed dataset locally."""
    if config.minimum_photos < 4 or config.visual_sample_per_species < 1:
        raise ValueError("E3 requires at least four distinct photos and a positive visual sample size")
    if config.screening_provider not in {"cpu", "cuda"}:
        raise ValueError("Screening provider must be cpu or cuda")
    model_root = Path(config.detector_dir).resolve()
    spec = yaml.safe_load((model_root / "pipeline.yaml").read_text())
    steps = [step for step in spec["steps"] if step["op"] == "detect_crop" and step.get("enabled")]
    if len(steps) != 1:
        raise ValueError("Expected one enabled E2 detector step")
    step = steps[0]
    model_hash = sha256(_safe_file(model_root, step["file"]))
    if model_hash != config.expected_detector_sha256:
        raise ValueError("Detector hash does not match the configured frozen artifact")
    reference_rows = []
    if config.source_percent is not None:
        if config.source_percent not in (20, 50, 100) or not config.reference_dataset:
            raise ValueError("Source volumes require 20/50/100 and the immutable v1 reference dataset")
        from pettrace.evaluation.research import PUBLIC_FINGERPRINT
        reference = load_prepared(config.reference_dataset)
        if reference.fingerprint != PUBLIC_FINGERPRINT:
            raise ValueError("Reference must be the immutable public v1 dataset")
        reference_rows = [r for r in reference.rows if r["split"] in {"val", "test"}]
    selected, sources = acquire_candidates(config, checkpoint=checkpoint)
    review_path = Path(config.data_root).resolve() / f"{'pilot' if config.source_percent is None else 'source-'+str(config.source_percent)}-seed{config.seed}" / "review.csv"
    write_review(selected, review_path)
    with review_path.open(newline="") as stream:
        review = list(csv.DictReader(stream))
    settings = asdict(config)
    settings.pop("output_root")
    settings.pop("source_archives")  # Transport location does not change the source inventory.
    package_root = Path(__file__).resolve().parents[1]
    implementation = {name: sha256(package_root / name) for name in
                      ("data/preparation.py", "data/sources.py", "data/screening.py", "data/curation.py",
                       "data/pilot.py", "data/volume.py", "data/verification.py", "data/source_capacity_v1.json", "models/pipeline.py")}
    if config.screening_provider == "cuda":
        implementation["data/screening_gpu.py"] = sha256(package_root / "data/screening_gpu.py")
    inputs = {"reference_rows": [{k: r[k] for k in MANIFEST_FIELDS if k != "path"} for r in reference_rows], "config": settings, "detector_step": step, "implementation": implementation,
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
    held_ids = {r["animal"] for r in reference_rows}
    held_hashes = {r["sha256"] for r in reference_rows}
    reserved = {r["path"]: "reserved_public_identity_or_image" for r in review
                if r["animal"] in held_ids or r["sha256"] in held_hashes}
    from functools import partial
    from pettrace.data.checkpoints import DriveBackup
    screening_checkpoint = (partial(checkpoint, paths=[Path(config.data_root) / "animal-screening-cache"])
                            if isinstance(checkpoint, DriveBackup) else checkpoint)
    screen = screen_animals
    if config.screening_provider == "cuda":
        from pettrace.data.screening_gpu import screen_cuda
        screen = partial(screen_cuda, model_file=_safe_file(model_root, step["file"]))
        screening_checkpoint = checkpoint  # CUDA tracks completed cache files instead of scanning the directory.
    exclusions, screening = screen(
        [r for r in review if r["path"] not in reserved], detector, Path(config.data_root) / "animal-screening-cache", detector_sha256=model_hash,
        confidence=config.confidence, iou=config.nms_iou, checkpoint=screening_checkpoint)
    exclusions.update(reserved)
    # Detect concurrent review edits before applying filtering decisions.
    if sha256(review_path) != inputs["review_sha256"]:
        raise ValueError("Review changed during preparation; rerun to use the new decisions")
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=output_root) as folder:
        folder = Path(folder)
        retained, audit = curate_review(review_path, folder, seed=config.seed, minimum_photos=config.minimum_photos,
                                        review_policy=config.review_policy, excluded_animals=config.excluded_animals,
                                        excluded_paths=exclusions)
        if reference_rows:
            # Reuse curation unchanged; only the split assignment is fixed for the volume study.
            retained = [{**r, "split": "train"} for r in retained] + reference_rows
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
        if config.source_percent is not None:
            from pettrace.data.volume import CAPACITY
            summary["processing"] = {}
            for source, capacity in CAPACITY.items():
                sampled = next(r["sampled_photos"] for r in sources["sources"] if r["ref"] == capacity["ref"])
                train_count = sum(r["source"] == source and r["split"] == "train" for r in retained)
                reasons = Counter(r["reason"] for r in excluded_rows if r["species"] == capacity["species"])
                reasons["missing_or_ambiguous_identity"] += sum(r["source"] == source for r in sources["skipped_paths"])
                if train_count + sum(reasons.values()) != sampled:
                    raise ValueError("Every sampled photo must have a training or exclusion outcome")
                summary["processing"][source] = {"requested_pct": config.source_percent,
                    "source_photos": capacity["photos"], "sampled_photos": sampled, "processed_photos": sampled,
                    "train_photos": train_count, "excluded_photos": sampled-train_count, "exclusion_reasons": dict(reasons),
                    "note": "Processing includes metadata/held-out rejection before image filtering; v1 val/test are reused separately."}
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
    if isinstance(checkpoint, DriveBackup):
        checkpoint(paths=[destination, output_root / "latest.json", review_path])
    elif checkpoint:
        checkpoint()
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
    parser.add_argument("--source-percent", type=int, choices=(20, 50, 100))
    parser.add_argument("--reference-dataset", help="Verified public v1 prepared root for fixed validation/test")
    parser.add_argument("--max-network-gib", type=int, default=3)
    parser.add_argument("--identities-per-species", type=int, default=300)
    parser.add_argument("--photos-per-identity", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-workers", type=int, default=4)
    args = parser.parse_args()
    detector_dir = args.detector_dir or acquire_detector(args.detector_uri, Path(args.data_root) / "detector-models")
    sources = {name: root for name, root in (("dogs-world", args.dog_root), ("lcw", args.cat_root)) if root}
    result = prepare(PrepareConfig(data_root=args.data_root, detector_dir=str(detector_dir), output_root=args.output_root,
                                   source_roots=sources, allow_download=args.allow_download,
                                   source_percent=args.source_percent, reference_dataset=args.reference_dataset,
                                   max_network_bytes=args.max_network_gib * 2**30,
                                   identities_per_species=args.identities_per_species,
                                   photos_per_identity=args.photos_per_identity, seed=args.seed, download_workers=args.download_workers))
    print(_json({"path": str(result.root), "fingerprint": result.fingerprint, "summary": result.summary}))


if __name__ == "__main__":
    main()
