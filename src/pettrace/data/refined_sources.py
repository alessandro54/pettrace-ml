"""Download the immutable public refined source datasets from Hugging Face."""

import csv
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from pettrace.data.verification import sha256


RELEASES = {
    "lcw": {
        "repo_id": "alessandro54/labeled-cats-in-the-wild-refined",
        "commit": "089518b5cffcf71a7f55dbc7d47d0f2269f98a2b",
        "v2_commit": "581e44a04fd461ef1ba248177b84aff61a2ada21",
        "species": "cat",
    },
    "dogs-world": {
        "repo_id": "alessandro54/dogs-of-the-world-refined",
        "commit": "26a361971d5a77976fadd4aeae5e8294d40333c7",
        "v2_commit": "9fdd6e5f7eaa752f16bd4dae54cd32b88c7e09ba",
        "species": "dog",
    },
}


def _safe(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe dataset path: {relative}")
    return root / relative


@dataclass(frozen=True)
class RefinedSource:
    source: str
    species: str
    repo_id: str
    version: str
    commit: str
    root: Path
    manifest: dict
    base_manifest: dict

    @property
    def metadata_path(self):
        return self.root / "versions/v1/metadata.csv"

    @property
    def localization_path(self):
        path = self.root / f"versions/{self.version}/screening.csv.gz"
        return path if path.is_file() else None

    def metadata(self):
        """Iterate source labels without loading the complete CSV into memory."""
        with self.metadata_path.open(newline="") as stream:
            yield from csv.DictReader(stream)

    def shards(self, split=None):
        """Return local WebDataset shard paths, optionally for one recorded split."""
        paths = [row["path"] for row in self.base_manifest["shards"]]
        if split is not None:
            paths = [path for path in paths if f"/data/{split}/" in f"/{path}"]
        return [_safe(self.root, path) for path in paths]


@dataclass(frozen=True)
class TrainingDataset:
    """Materialized, verified union of the two public refined training sources."""

    version: str
    fingerprint: str
    detector_sha256: str
    rows: list
    sources: tuple

    @property
    def summary(self):
        return {
            "retained_photos": len(self.rows),
            "retained_identities": len({row["animal"] for row in self.rows}),
        }

    @property
    def provenance(self):
        return {
            "version": self.version,
            "prepared_version": "v4.100",
            "prepared_fingerprint": self.fingerprint,
            "detector_sha256": self.detector_sha256,
            "photos": len(self.rows),
            "identities": self.summary["retained_identities"],
            "sources": [
                {"source": source.source, "repo_id": source.repo_id,
                 "version": source.version, "commit": source.commit,
                 "base_commit": source.base_manifest.get("commit", RELEASES[source.source]["commit"])}
                for source in self.sources
            ],
            "equivalence": "Exact v4.100 photo bytes, identities and splits; v2 adds only pinned YOLO boxes.",
        }


def load_refined_source(source, *, version="v1", cache_dir=None, token=None):
    """Download one pinned refined source release and return its local files and provenance."""
    if source not in RELEASES:
        raise ValueError(f"Unsupported refined source: {source}")
    if version not in {"v1", "v2"}:
        raise ValueError("Refined public sources currently support v1 and v2")
    spec = RELEASES[source]
    commit = HfApi(token=token).dataset_info(spec["repo_id"], revision=version).sha
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Hugging Face did not resolve the dataset to an immutable commit")
    expected_commit = spec["commit"] if version == "v1" else spec["v2_commit"]
    if commit != expected_commit:
        raise ValueError(f"{spec['repo_id']}@{version} no longer resolves to its audited commit")
    patterns = ["versions/v1/**"]
    if version == "v2":
        patterns.append("versions/v2/**")
    kwargs = {
        "repo_type": "dataset", "revision": commit, "token": token,
        "allow_patterns": patterns,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    root = Path(snapshot_download(spec["repo_id"], **kwargs)).resolve()
    base_manifest = json.loads((root / "versions/v1/dataset.json").read_text())
    if (base_manifest.get("source"), base_manifest.get("release_version")) != (source, "v1"):
        raise ValueError("The v1 manifest does not describe the requested refined source")
    metadata_path = root / "versions/v1/metadata.csv"
    if sha256(metadata_path) != base_manifest["file_sha256"]["versions/v1/metadata.csv"]:
        raise ValueError("The refined source metadata does not match its manifest")
    if any(not _safe(root, shard["path"]).is_file() for shard in base_manifest["shards"]):
        raise FileNotFoundError("The refined source is missing one or more image shards")
    manifest = base_manifest
    if version == "v2":
        manifest = json.loads((root / "versions/v2/dataset.json").read_text())
        base = manifest.get("base", {})
        if (manifest.get("source"), manifest.get("release_version")) != (source, "v2"):
            raise ValueError("The v2 manifest does not describe the requested refined source")
        if (base.get("repo_id"), base.get("version"), base.get("commit")) != (
                spec["repo_id"], "v1", spec["commit"]):
            raise ValueError("The v2 overlay does not reference the audited v1 release")
        for name, receipt in manifest["files"].items():
            path = _safe(root, name)
            if not path.is_file() or path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
                raise ValueError(f"The v2 overlay file does not match its manifest: {name}")
    return RefinedSource(source, spec["species"], spec["repo_id"], version, commit,
                         root, manifest, base_manifest)


def load_training(*, version="v2", cache_dir=None, token=None):
    """Load the public training corpus with immutable labels, splits and YOLO boxes."""
    if version != "v2":
        raise ValueError("The combined training corpus currently supports only v2")
    cache = Path(cache_dir or Path.home() / ".cache/pettrace/refined").resolve()
    sources = tuple(load_refined_source(name, version=version, cache_dir=cache / "hub", token=token)
                    for name in ("lcw", "dogs-world"))
    prepared = {(source.base_manifest.get("prepared_version"),
                 source.base_manifest.get("prepared_fingerprint")) for source in sources}
    prepared_version, fingerprint = next(iter(prepared)) if len(prepared) == 1 else (None, None)
    if prepared_version != "v4.100" or not fingerprint:
        raise ValueError("The public sources do not share the audited v4.100 preparation")
    detector_hashes = {source.manifest.get("localization", {}).get("detector_sha256") for source in sources}
    detector_sha256 = next(iter(detector_hashes)) if len(detector_hashes) == 1 else None
    if not detector_sha256:
        raise ValueError("The public sources do not share one pinned YOLO detector")

    # Reuse the release extractor used to create v2 so byte and TAR validation stay identical.
    from pettrace.data.yolo_overlay import _extract_rows

    rows = []
    for source in sources:
        print(f"Dataset: verifying and materializing {source.repo_id}@{source.commit[:12]}…", flush=True)
        extracted, metadata = _extract_rows(
            source.root, source.base_manifest, cache / "images" / source.source)
        by_sha = {row["sha256"]: row for row in extracted}
        with gzip.open(source.localization_path, "rt", newline="") as stream:
            localized = list(csv.DictReader(stream))
        if len(localized) != source.manifest.get("photos") or set(by_sha) != {row["sha256"] for row in localized}:
            raise ValueError(f"{source.source} v1 images and v2 boxes differ")
        for box in localized:
            photo = by_sha[box["sha256"]]
            label = metadata[box["sha256"]]
            if (box["identity_id"], box["species"], box["source"], box["split"]) != (
                    label["identity_id"], label["species"], label["source"], label["split"]):
                raise ValueError(f"{source.source} v2 labels differ from immutable v1")
            photo["box"] = [float(box[name]) for name in ("x1", "y1", "x2", "y2")]
            rows.append(photo)
        print(f"Dataset: {source.source} ready ({len(extracted):,} photos).", flush=True)
    if len(rows) != sum(source.base_manifest["photos"] for source in sources):
        raise ValueError("The combined public training corpus is incomplete")
    return TrainingDataset(version, fingerprint, detector_sha256, rows, sources)
