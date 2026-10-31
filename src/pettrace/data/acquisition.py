"""Acquire public body-ID datasets, retain provenance, then build an audited manifest."""

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import requests

from pettrace.data.curation import audit_and_split, discover

MPDD_URL = "https://data.mendeley.com/public-files/datasets/v5j6m8dzhv/files/05d1d583-faf6-410d-89a5-a6b1134b6e5e/file_downloaded"
MPDD_SHA256 = "6c800c1b4aa67629544dec7444dee85bc57781abde1ae1d077ea9ef1804284cd"


def download(url, destination, expected_hash=None, auth=None):
    destination = Path(destination)
    digest = hashlib.sha256()
    if not destination.exists():
        partial = destination.with_suffix(".partial")
        try:
            with requests.get(url, stream=True, timeout=(30, 120), auth=auth) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", 0))
                received, last = 0, 0
                with partial.open("wb") as f:
                    for block in response.iter_content(1 << 20):
                        if block:
                            f.write(block)
                            received += len(block)
                            if received - last >= 128 << 20:
                                print(f"Downloaded {received / 2**20:.0f} MiB / {total / 2**20:.0f} MiB", flush=True)
                                last = received
            if not zipfile.is_zipfile(partial):
                raise ValueError("Download is not a ZIP; authenticate/download manually from the source")
            partial.replace(destination)
        except Exception:
            # Leave a clearly named partial file for diagnosis, never treat it as a completed dataset.
            raise
    with destination.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    result = digest.hexdigest()
    if expected_hash and result != expected_hash:
        raise ValueError(f"Archive SHA-256 mismatch: {destination.name}")
    return result


def extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        nested_archives = [destination / info.filename for info in z.infolist()
                           if info.filename.lower().endswith(".zip")]
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive path or symlink")
        z.extractall(destination)
    # Some publisher downloads are a wrapper ZIP containing the actual MPDD archive.
    for nested in nested_archives:
        with nested.open("rb") as f:
            payload = io.BytesIO(f.read())
        extract(payload, nested.parent)


def prepare(root, cat_root=None, mpdd_root=None, cat_version=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sources = []
    if mpdd_root is None:
        archive = root / "MPDD-v1.zip"
        archive_hash = download(MPDD_URL, archive, MPDD_SHA256)
        mpdd_root = root / "mpdd"
        if not (mpdd_root / ".extracted").exists():
            extract(archive, mpdd_root)
            (mpdd_root / ".extracted").write_text(archive_hash)
        sources.append({"name": "MPDD", "version": 1, "sha256": archive_hash,
                        "url": "https://data.mendeley.com/datasets/v5j6m8dzhv/1", "license": "CC BY 4.0"})
    else:
        sources.append({"name": "MPDD", "root": str(mpdd_root), "version": "user-provided archive",
                        "license": "CC BY 4.0; verify supplied source matches publisher"})
    if cat_root is None:
        auth = None
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            auth = (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])
        metadata_url = "https://www.kaggle.com/api/v1/datasets/view/timost1234/cat-individuals"
        if cat_version is None:
            metadata = requests.get(metadata_url, timeout=30, auth=auth)
            metadata.raise_for_status()
            cat_version = metadata.json().get("currentVersionNumber")
            if not cat_version:
                raise ValueError("Cannot resolve CatIndividuals version; supply an extracted cat_root")
        archive = root / f"CatIndividuals-v{cat_version}.zip"
        url = f"https://www.kaggle.com/api/v1/datasets/download/timost1234/cat-individuals?datasetVersionNumber={cat_version}"
        archive_hash = download(url, archive, auth=auth)
        cat_root = root / f"cats-v{cat_version}"
        if not (cat_root / ".extracted").exists():
            extract(archive, cat_root)
            (cat_root / ".extracted").write_text(archive_hash)
        sources.append({"name": "CatIndividualImages", "version": cat_version, "sha256": archive_hash,
                        "url": "https://www.kaggle.com/datasets/timost1234/cat-individuals",
                        "license": "CC BY 4.0 per WildlifeDatasets 1.0.9; retain publisher terms"})
    else:
        sources.append({"name": "CatIndividualImages", "root": str(cat_root), "version": "user-provided archive",
                        "license": "Retain and verify the downloaded dataset's license"})
    dogs, skipped_dogs = discover(mpdd_root, "mpdd", "dog")
    cats, skipped_cats = discover(cat_root, "cat-individuals", "cat")
    rows, audit = audit_and_split(dogs + cats, root / "curated")
    (root / "curated/sources.json").write_text(json.dumps({"sources": sources,
        "skipped_paths": skipped_dogs + skipped_cats}, indent=2))
    print(json.dumps(audit["counts"], indent=2))
    print("Conservative duplicate removals:", len(audit["removed_duplicates"]))
    print("No automatic audit can confirm different photo sessions; inspect the public examples.")
    return root / "curated/manifest.csv", rows, audit
