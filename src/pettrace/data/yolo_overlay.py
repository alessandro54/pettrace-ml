"""Build a pinned YOLO localization overlay for the public refined datasets."""

import csv
import gzip
import hashlib
import io
import json
import shutil
import tarfile
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

from pettrace.data.checkpoints import DriveBackup
from pettrace.data.curation import EXTENSIONS
from pettrace.data.preparation import DETECTOR_SHA256, _safe_file
from pettrace.data.refined_sources import RELEASES
from pettrace.data.screening_gpu import screen_cuda
from pettrace.models.pipeline import Detector


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _release_root(source, cache, token):
    spec = RELEASES[source]
    resolved = HfApi(token=token).dataset_info(spec["repo_id"], revision="v1").sha
    if resolved != spec["commit"]:
        raise ValueError(f"{spec['repo_id']}@v1 no longer resolves to its audited commit")
    root = Path(snapshot_download(
        spec["repo_id"], repo_type="dataset", revision=spec["commit"], token=token,
        local_dir=str(Path(cache) / spec["repo_id"].replace("/", "--")),
        allow_patterns=["versions/v1/**"],
    ))
    version = root / "versions/v1"
    evidence = json.loads((version / "dataset.json").read_text())
    if (evidence.get("source"), evidence.get("release_version")) != (source, "v1"):
        raise ValueError("The downloaded v1 manifest does not describe the requested source")
    metadata = version / "metadata.csv"
    if _sha256(metadata) != evidence["file_sha256"]["versions/v1/metadata.csv"]:
        raise ValueError("The v1 metadata hash does not match its manifest")
    return root, evidence


def _extract_rows(root, evidence, destination):
    """Materialize one verified source release for the existing screening API."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(root) / "versions/v1/metadata.csv"
    with metadata_path.open(newline="") as stream:
        metadata = list(csv.DictReader(stream))
    if len(metadata) != evidence["photos"] or len({row["sha256"] for row in metadata}) != len(metadata):
        raise ValueError("The v1 metadata inventory is incomplete or has duplicate photo keys")
    by_shard = {}
    for row in metadata:
        if row["source"] != evidence["source"] or row["key"] != row["sha256"]:
            raise ValueError("The v1 metadata source or content key changed")
        by_shard.setdefault(row["shard"], {})[row["key"]] = row

    shard_receipts = {row["path"]: row for row in evidence["shards"]}
    if set(by_shard) != set(shard_receipts):
        raise ValueError("The v1 metadata and shard inventory differ")
    photos = {}
    for shard_name, expected in sorted(by_shard.items()):
        archive_path = Path(root) / shard_name
        receipt = shard_receipts[shard_name]
        if _sha256(archive_path) != receipt["sha256"] or len(expected) != receipt["samples"]:
            raise ValueError(f"The v1 shard changed: {shard_name}")
        seen = set()
        with tarfile.open(archive_path) as archive:
            for member in archive:
                name = Path(member.name)
                if not member.isfile() or name.name != member.name:
                    raise ValueError(f"Unsafe v1 TAR member: {member.name}")
                if name.suffix == ".json":
                    continue
                if name.suffix.lower() not in EXTENSIONS or name.stem not in expected or name.stem in seen:
                    raise ValueError(f"Unexpected v1 TAR member: {member.name}")
                target = destination / member.name
                if not target.is_file() or _sha256(target) != name.stem:
                    partial = target.with_suffix(target.suffix + ".partial")
                    with archive.extractfile(member) as incoming, partial.open("wb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
                    if _sha256(partial) != name.stem:
                        partial.unlink(missing_ok=True)
                        raise ValueError(f"Photo bytes do not match their v1 key: {member.name}")
                    partial.replace(target)
                photos[name.stem] = target
                seen.add(name.stem)
        if seen != set(expected):
            raise ValueError(f"Missing images in v1 shard: {shard_name}")

    rows = [{"path": str(photos[row["sha256"]]), "source": row["source"],
             "species": row["species"], "animal": row["identity_id"], "sha256": row["sha256"],
             "split": row["split"]}
            for row in metadata]
    return rows, {row["sha256"]: row for row in metadata}


def _detector(path):
    root = Path(path).resolve()
    spec = yaml.safe_load((root / "pipeline.yaml").read_text())
    steps = [step for step in spec["steps"] if step["op"] == "detect_crop" and step.get("enabled")]
    if len(steps) != 1:
        raise ValueError("The pinned E2 model must contain exactly one enabled detector")
    model = _safe_file(root, steps[0]["file"])
    if _sha256(model) != DETECTOR_SHA256:
        raise ValueError("The detector does not match the pinned E2 SHA-256")
    return Detector(str(root), steps[0]), model


def _write_overlay(output, spec, evidence, metadata, report):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for result in report["records"]:
        boxes = result["boxes"]
        if result["reason"] != "single_expected_animal" or len(boxes) != 1:
            raise ValueError("v2 requires one expected animal in every immutable v1 image")
        source = metadata[result["image_sha256"]]
        box = boxes[0]
        records.append({
            "sha256": result["image_sha256"], "identity_id": source["identity_id"],
            "species": source["species"], "source": source["source"], "split": source["split"],
            "score": box["score"], "class_id": box["class_id"],
            "x1": box["box"][0], "y1": box["box"][1], "x2": box["box"][2], "y2": box["box"][3],
        })
    records.sort(key=lambda row: row["sha256"])
    screening = output / "screening.csv.gz"
    with screening.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, records[0].keys())
                writer.writeheader()
                writer.writerows(records)
    validation = output / "gpu-validation.json"
    validation.write_text(json.dumps(report["gpu_validation"], indent=2, sort_keys=True))
    manifest = {
        "schema": 1,
        "release_version": "v2",
        "source": evidence["source"],
        "photos": len(records),
        "identities": len({row["identity_id"] for row in records}),
        "splits": dict(Counter(row["split"] for row in records)),
        "base": {
            "repo_id": spec["repo_id"], "version": "v1", "commit": spec["commit"],
            "prepared_version": evidence["prepared_version"],
            "prepared_fingerprint": evidence["prepared_fingerprint"],
            "data_path": "versions/v1/data",
        },
        "localization": {
            "detector_model_uri": "models:/pettrace-embedder/2",
            "detector_sha256": DETECTOR_SHA256,
            "config": report["config"],
            "execution": report["execution"],
            "coordinate_space": "EXIF-oriented full-image pixels",
            "crop_policy": "No crops are stored; consumers apply the model artifact's margin deterministically.",
            "limitation": report["limitation"],
        },
        "files": {
            "versions/v2/screening.csv.gz": {"sha256": _sha256(screening), "bytes": screening.stat().st_size},
            "versions/v2/gpu-validation.json": {"sha256": _sha256(validation), "bytes": validation.stat().st_size},
        },
    }
    (output / "dataset.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return output


def _verify_overlay(output, spec):
    output = Path(output)
    manifest = json.loads((output / "dataset.json").read_text())
    if manifest.get("release_version") != "v2" or manifest.get("base", {}).get("commit") != spec["commit"]:
        raise ValueError("Existing v2 output belongs to another base release")
    for name, receipt in manifest["files"].items():
        path = output / Path(name).name
        if not path.is_file() or path.stat().st_size != receipt["bytes"] or _sha256(path) != receipt["sha256"]:
            raise ValueError(f"Existing v2 output changed: {name}")
    return output


def prepare_yolo_overlay(source, detector_dir, work_dir, checkpoint_dir, *, token=None):
    """Create a local v2 overlay; it never mutates a Hugging Face repository."""
    if source not in RELEASES:
        raise ValueError(f"Unsupported refined source: {source}")
    spec = RELEASES[source]
    source_root = Path(work_dir).resolve() / source
    output = source_root / "versions/v2"
    if (output / "dataset.json").is_file():
        return _verify_overlay(output, spec)
    root, evidence = _release_root(source, source_root / "hub", token)
    rows, metadata = _extract_rows(root, evidence, source_root / "images")
    if any(row["species"] != spec["species"] for row in rows):
        raise ValueError("The v1 source contains an unexpected species label")
    cpu, model = _detector(detector_dir)
    cache = source_root / "animal-screening-cache"
    backup = DriveBackup({"cache": cache}, Path(checkpoint_dir).resolve() / source) if checkpoint_dir else None
    if backup and backup.inventory():
        backup.restore()
    with backup or nullcontext():
        exclusions, report = screen_cuda(
            rows, cpu, cache, model, detector_sha256=DETECTOR_SHA256, checkpoint=backup,
        )
    if exclusions:
        rejected = source_root / "v2-rejected.json"
        rejected.write_text(json.dumps(exclusions, indent=2, sort_keys=True))
        raise ValueError(f"Pinned v1 no longer passes the frozen screening policy; inspect {rejected}")
    return _write_overlay(output, spec, evidence, metadata, report)


def update_dataset_card(card, manifest):
    """Document v2 without changing the v1 data-file configuration or immutable tag."""
    start, end = "<!-- pettrace-yolo-v2:start -->", "<!-- pettrace-yolo-v2:end -->"
    section = f"""{start}
## Version v2: pinned YOLO localization

`v2` reuses every full-frame image, identity and split from immutable `v1` and adds
`versions/v2/screening.csv.gz`: one verified animal box per image. It does not store crops or
duplicate the image shards. Consumers apply the model artifact's crop margin deterministically.

- Base commit: `{manifest['base']['commit']}`.
- Photos / identities: {manifest['photos']:,} / {manifest['identities']:,}.
- Detector: `{manifest['localization']['detector_model_uri']}`,
  SHA-256 `{manifest['localization']['detector_sha256']}`.
- Policy: confidence {manifest['localization']['config']['confidence']}, class-agnostic NMS IoU
  {manifest['localization']['config']['nms_iou']}, exactly one animal of the expected species.
- Coordinates: EXIF-oriented full-image pixels.
- Evidence: `versions/v2/dataset.json` and `gpu-validation.json`.

The original v1 screening checked dataset suitability; it did not crop the stored photographs.
YOLO does not verify animal identity, and missed or incorrect detections remain possible.
{end}"""
    if start in card or end in card:
        if card.count(start) != 1 or card.count(end) != 1 or card.index(start) > card.index(end):
            raise ValueError("Dataset card contains malformed v2 documentation markers")
        return card[:card.index(start)] + section + card[card.index(end) + len(end):]
    return card.rstrip() + "\n\n" + section + "\n"
