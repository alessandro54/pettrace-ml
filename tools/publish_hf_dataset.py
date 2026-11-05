"""Build and optionally publish one refined source dataset to Hugging Face.

This is intentionally operational tooling, not part of the ``pettrace`` package.
The package only consumes immutable dataset revisions.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import HfApi, hf_hub_download

from pettrace.data.versions import load_local_training


SOURCES = {
    "lcw": {
        "repo": "alessandro54/lcw-refined",
        "license": "apache-2.0",
        "license_name": "Apache 2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0.txt",
        "license_sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        "source_url": "https://www.kaggle.com/datasets/dseidli/lcwlabeled-cats-in-the-wild",
        "ref": "dseidli/lcwlabeled-cats-in-the-wild",
    },
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _license_text(spec):
    with urlopen(spec["license_url"], timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != spec["license_sha256"]:
        raise ValueError("The downloaded license text does not match the pinned SHA-256")
    return content


def _source_receipt(prepared, spec):
    receipt = json.loads((prepared.root / "sources.json").read_text())
    matches = [row for row in receipt.get("sources", []) if row.get("ref") == spec["ref"]]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source receipt for {spec['ref']}")
    declared = matches[0].get("publisher_license") or matches[0].get("license", "")
    if spec["license_name"].lower() not in declared.lower():
        raise ValueError(f"Source receipt declares {declared!r}, not {spec['license_name']}")
    # Per-photo hashes are already carried by metadata.csv and can make the receipt enormous.
    return {key: value for key, value in matches[0].items()
            if key not in {"file_sha256", "selected_identities"}}


def _tar_info(name, size):
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    return info


def _add_sample(archive, row, key):
    photo = Path(row["path"])
    suffix = photo.suffix.lower()
    with photo.open("rb") as stream:
        archive.addfile(_tar_info(f"{key}{suffix}", photo.stat().st_size), stream)
    metadata = json.dumps({
        "identity_id": row["animal"],
        "species": row["species"],
        "source": row["source"],
        "split": row["split"],
        "sha256": row["sha256"],
        "original_test": bool(row["original_test"]),
    }, sort_keys=True).encode()
    archive.addfile(_tar_info(f"{key}.json", len(metadata)), io.BytesIO(metadata))


def _card(spec, *, source, release_version, prepared_version, fingerprint, counts):
    data_files = "\n".join(
        f"    - split: {split}\n      path: versions/{release_version}/data/{split}/*.tar"
        for split in ("train", "val", "test") if counts.get(split)
    )
    return f"""---
license: {spec['license']}
task_categories:
- image-feature-extraction
- image-classification
tags:
- animal-re-identification
- cats
configs:
- config_name: default
  data_files:
{data_files}
---

# LCW Refined

Deterministic, refined LCW cat identities for PetTrace representation-learning experiments.

- Upstream: [LCW on Kaggle]({spec['source_url']}), version 1.
- Upstream license declared by the publisher: {spec['license_name']}.
- PetTrace source snapshot: `pettrace-training@{prepared_version}`.
- Refined release: `{release_version}`.
- Prepared fingerprint: `{fingerprint}`.
- Samples: {sum(counts.values()):,} ({', '.join(f'{k}: {v:,}' for k, v in counts.items())}).

## Refinement

PetTrace removes exact duplicates, quarantines cross-identity and near-copy conflicts, applies the
recorded animal-screening policy, and preserves immutable train/validation/test assignments. These
operations do not prove that every source identity is correct. See
`versions/{release_version}/dataset.json` and `metadata.csv` for lineage and hashes.

This repository redistributes only the refined LCW subset, not the original Kaggle archive. The
original publisher and source remain attributed above. PetTrace does not claim ownership of the
underlying photographs or the LCW name. Apache-2.0 is the license declared by the dataset publisher;
users remain responsible for checking any third-party image rights that may apply.
"""


def _expected_files(evidence):
    version = evidence["release_version"]
    return {"README.md", "LICENSE", f"versions/{version}/metadata.csv",
            f"versions/{version}/dataset.json", *(shard["path"] for shard in evidence["shards"])}


def _verify_staging(root, evidence):
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != _expected_files(evidence):
        raise ValueError("Staging inventory changed; refuse to upload unexpected or missing files")
    for path, digest in evidence["file_sha256"].items():
        if _sha256(root / path) != digest:
            raise ValueError(f"Staged file changed: {path}")
    for shard in evidence["shards"]:
        if _sha256(root / shard["path"]) != shard["sha256"]:
            raise ValueError(f"Staged shard changed: {shard['path']}")


def build_release(prepared_path, destination, *, source="lcw", release_version="v1",
                  prepared_version="v4.100", shard_size=2 * 1024**3, license_text=None):
    """Create deterministic WebDataset shards and evidence without contacting Hugging Face."""
    if source not in SOURCES:
        raise ValueError(f"Unsupported source: {source}")
    if release_version != "v1" or prepared_version != "v4.100":
        raise ValueError("The first LCW release must map pettrace-training@v4.100 to v1")
    if shard_size < 1024**2:
        raise ValueError("Shard size must be at least 1 MiB")
    prepared, reference = load_local_training(prepared_path)
    if reference is None or reference.get("version") != prepared_version:
        raise ValueError(f"--prepared must be the immutable {prepared_version}.json reference")
    spec = SOURCES[source]
    source_receipt = _source_receipt(prepared, spec)
    rows = sorted((row for row in prepared.rows if row["source"] == source),
                  key=lambda row: (row["split"], row["sha256"]))
    if not rows:
        raise ValueError(f"Prepared snapshot contains no {source} photos")
    keys = [row["sha256"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("A refined source release requires unique photo SHA-256 keys")

    destination = Path(destination).resolve()
    version_root = destination / "versions" / release_version
    existing = version_root / "dataset.json"
    if existing.is_file():
        evidence = json.loads(existing.read_text())
        expected = (source, release_version, prepared_version, prepared.fingerprint)
        actual = tuple(evidence.get(key) for key in
                       ("source", "release_version", "prepared_version", "prepared_fingerprint"))
        if actual != expected:
            raise ValueError("Staging directory already contains a different release")
        _verify_staging(destination, evidence)
        return destination
    if destination.exists():
        raise ValueError("Staging directory is incomplete; inspect it and choose a new empty directory")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        records, shards = [], []
        for split in ("train", "val", "test"):
            selected = [row for row in rows if row["split"] == split]
            shard_number = -1
            archive = None
            current_bytes = 0
            try:
                for row in selected:
                    photo_bytes = Path(row["path"]).stat().st_size
                    if archive is None or (current_bytes and current_bytes + photo_bytes > shard_size):
                        if archive is not None:
                            archive.close()
                        shard_number += 1
                        relative = Path("versions") / release_version / "data" / split / f"{source}-{shard_number:05d}.tar"
                        path = temporary / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        archive = tarfile.open(path, "w")
                        current_bytes = 0
                    key = row["sha256"]
                    _add_sample(archive, row, key)
                    current_bytes += photo_bytes
                    records.append({"key": key, "identity_id": row["animal"], "species": row["species"],
                                    "source": source, "split": split, "sha256": row["sha256"],
                                    "original_test": bool(row["original_test"]),
                                    "shard": relative.as_posix()})
            finally:
                if archive is not None:
                    archive.close()

        for path in sorted((temporary / "versions" / release_version / "data").rglob("*.tar")):
            shards.append({"path": path.relative_to(temporary).as_posix(), "bytes": path.stat().st_size,
                           "sha256": _sha256(path),
                           "samples": sum(row["shard"] == path.relative_to(temporary).as_posix()
                                          for row in records)})
        version = temporary / "versions" / release_version
        with (version / "metadata.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        counts = dict(Counter(row["split"] for row in records))
        readme = temporary / "README.md"
        readme.write_text(_card(
            spec, source=source, release_version=release_version, prepared_version=prepared_version,
            fingerprint=prepared.fingerprint, counts=counts))
        license_file = temporary / "LICENSE"
        license_file.write_bytes(license_text if license_text is not None else _license_text(spec))
        evidence = {
            "schema": 1,
            "source": source,
            "release_version": release_version,
            "prepared_version": prepared_version,
            "prepared_fingerprint": prepared.fingerprint,
            "license": spec["license"],
            "source_url": spec["source_url"],
            "photos": len(records),
            "identities": len({row["identity_id"] for row in records}),
            "splits": counts,
            "source_receipt": source_receipt,
            "shards": shards,
            "file_sha256": {
                "README.md": _sha256(readme),
                "LICENSE": _sha256(license_file),
                f"versions/{release_version}/metadata.csv": _sha256(version / "metadata.csv"),
            },
        }
        (version / "dataset.json").write_text(json.dumps(evidence, indent=2, sort_keys=True))
        temporary.replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return destination


def publish_release(staging, *, source="lcw", release_version="v1", repo_id=None):
    """Upload one verified staging directory and create an immutable public tag."""
    spec = SOURCES[source]
    repo_id = repo_id or spec["repo"]
    staging = Path(staging).resolve()
    evidence_path = staging / "versions" / release_version / "dataset.json"
    evidence = json.loads(evidence_path.read_text())
    if evidence.get("source") != source or evidence.get("release_version") != release_version:
        raise ValueError("Staging evidence does not match the requested publication")
    _verify_staging(staging, evidence)

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    if api.dataset_info(repo_id).private:
        raise ValueError("Refined source releases require a public dataset repository")
    refs = api.list_repo_refs(repo_id, repo_type="dataset")
    tags = {ref.name: ref.target_commit for ref in refs.tags}
    filename = f"versions/{release_version}/dataset.json"
    if release_version in tags:
        remote = json.loads(Path(hf_hub_download(
            repo_id, filename=filename, repo_type="dataset", revision=tags[release_version])).read_text())
        if remote != evidence:
            raise ValueError(f"{release_version} already identifies another dataset")
        return {"repo": repo_id, "version": release_version, "commit": tags[release_version],
                "prepared_fingerprint": evidence["prepared_fingerprint"]}

    expected = _expected_files(evidence)
    uploaded = api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(staging),
                                 allow_patterns=sorted(expected),
                                 commit_message=f"Publish {source} refined {release_version}")
    remote = set(api.list_repo_files(repo_id, repo_type="dataset", revision=uploaded.oid))
    if not expected <= remote:
        raise ValueError("HF upload is incomplete; no version tag was created")
    api.create_tag(repo_id, repo_type="dataset", tag=release_version, revision=uploaded.oid, exist_ok=False)
    return {"repo": repo_id, "version": release_version, "commit": uploaded.oid,
            "prepared_fingerprint": evidence["prepared_fingerprint"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, help="Path to the verified v4.100 prepared snapshot")
    parser.add_argument("--staging", required=True, help="Persistent output directory for verified shards")
    parser.add_argument("--source", choices=sorted(SOURCES), default="lcw")
    parser.add_argument("--prepared-version", default="v4.100")
    parser.add_argument("--release-version", default="v1")
    parser.add_argument("--shard-size-gib", type=float, default=2.0)
    parser.add_argument("--repo-id")
    parser.add_argument("--upload", action="store_true", help="Perform the external HF publication")
    args = parser.parse_args()
    staging = build_release(args.prepared, args.staging, source=args.source,
                            prepared_version=args.prepared_version, release_version=args.release_version,
                            shard_size=int(args.shard_size_gib * 1024**3))
    print(f"Staged and verified: {staging}")
    if args.upload:
        print(json.dumps(publish_release(staging, source=args.source,
                                         release_version=args.release_version, repo_id=args.repo_id), indent=2))


if __name__ == "__main__":
    main()
