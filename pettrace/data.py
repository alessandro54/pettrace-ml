"""Load a pinned version of the `pettrace-reid` dataset (private Hugging Face dataset).

Training and evaluation read only published versions — never R2 or the app API — so every result
traces back to an immutable snapshot. Needs a read token in HF_TOKEN.
"""

import csv
import hashlib
import os
from dataclasses import dataclass, field

REPO = os.environ.get("HF_DATASET_REPO", "alessandro54/pettrace-reid")


@dataclass
class Photo:
    path: str  # absolute path of the image
    animal: str
    species: str
    split: str
    light: float


@dataclass
class Dataset:
    repo: str
    version: str
    commit: str  # Hub commit the tag points to
    fingerprint: str  # sha256 of metadata.csv (first 16 hex): changes iff labels/photos change
    root: str
    animals: dict[str, dict] = field(default_factory=dict)  # animal_id → animals.csv row
    photos: list[Photo] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.repo.split('/')[-1]}@{self.version}"

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo}/tree/{self.version}"

    def split(self, name: str) -> list[Photo]:
        return [p for p in self.photos if p.split == name]


def load(version: str, repo: str = REPO, cache_dir: str | None = None) -> Dataset:
    from huggingface_hub import HfApi, snapshot_download

    root = snapshot_download(repo, repo_type="dataset", revision=version, cache_dir=cache_dir)
    commit = HfApi().dataset_info(repo, revision=version).sha
    return from_dir(root, repo=repo, version=version, commit=commit)


def from_dir(root: str, repo: str = REPO, version: str = "local", commit: str = "") -> Dataset:
    meta_path = os.path.join(root, "metadata.csv")
    with open(meta_path, "rb") as f:
        fingerprint = hashlib.sha256(f.read()).hexdigest()[:16]
    with open(os.path.join(root, "animals.csv"), newline="") as f:
        animals = {row["animal_id"]: row for row in csv.DictReader(f)}
    with open(meta_path, newline="") as f:
        photos = [
            Photo(
                path=os.path.join(root, row["file_name"]),
                animal=row["animal_id"],
                species=row["species"],
                split=row["split"],
                light=float(row.get("light") or 0),
            )
            for row in csv.DictReader(f)
        ]
    return Dataset(repo, version, commit, fingerprint, root, animals, photos)
