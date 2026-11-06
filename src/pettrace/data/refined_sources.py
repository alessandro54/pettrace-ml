"""Download the immutable public refined source datasets from Hugging Face."""

import csv
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
        "species": "cat",
    },
    "dogs-world": {
        "repo_id": "alessandro54/dogs-of-the-world-refined",
        "commit": "26a361971d5a77976fadd4aeae5e8294d40333c7",
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
    if version == "v1" and commit != spec["commit"]:
        raise ValueError(f"{spec['repo_id']}@v1 no longer resolves to its audited commit")
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
