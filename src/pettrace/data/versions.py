"""Version prepared public identities separately from the private PetTrace benchmark."""

import json
import os
import re
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import validate_repo_id

from pettrace.data.benchmark import REPO as BENCHMARK_REPO
from pettrace.data.preparation import load_prepared
from pettrace.data.verification import verification_session

REPO = "alessandro54/pettrace-training"


def _check(repo_id, version):
    validate_repo_id(repo_id)
    if repo_id in {BENCHMARK_REPO, "alessandro54/pettrace-reid"}:
        raise ValueError("Training releases must not modify or reuse the private evaluation benchmark")
    if not re.fullmatch(r"v[1-9]\d*(?:\.\d+){0,2}", version):
        raise ValueError("Use a training dataset version such as v1 or v1.1")


def _receipt(repo_id, version, commit, fingerprint):
    return {"repo": repo_id, "version": version, "commit": commit, "fingerprint": fingerprint,
            "path_in_repo": f"versions/{version}"}


def record_version(path, output_root, *, version="v1", repo_id=REPO):
    """Give a local snapshot a stable version name without publishing anything."""
    _check(repo_id, version)
    prepared = load_prepared(path)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if prepared.root.parent != output_root:
        raise ValueError("Named versions must refer to snapshots directly inside output_root")
    receipt = {"repo": repo_id, "version": version, "fingerprint": prepared.fingerprint, "path": prepared.root.name}
    target = output_root / f"{version}.json"
    _record_reference(receipt, target)
    return target


def _record_reference(receipt, target):
    version = receipt["version"]
    if target.exists():
        if json.loads(target.read_text()) != receipt:
            raise ValueError(f"{version} already identifies another snapshot; use a new version")
    else:
        with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, indent=2)
        try:
            try:
                os.link(temporary, target)  # Atomic exclusive publication; never replace another release.
            except FileExistsError:
                if json.loads(target.read_text()) != receipt:
                    raise ValueError(f"{version} already identifies another snapshot; use a new version") from None
        finally:
            temporary.unlink(missing_ok=True)


def _local_reference(path):
    receipt = json.loads(path.read_text())
    _check(receipt["repo"], receipt["version"])
    relative = Path(receipt["path"])
    root = (path.parent / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not root.is_relative_to(path.parent):
        raise ValueError("Unsafe training version path")
    return root, receipt


def load_local_training(path):
    path = Path(path).resolve()
    if path.is_file() and path.name not in {"latest.json", "prepared.json"}:
        root, receipt = _local_reference(path)
        prepared = load_prepared(root)
        if prepared.fingerprint != receipt["fingerprint"]:
            raise ValueError("Named training version fingerprint changed")
        return prepared, receipt
    return load_prepared(path), None


def load_training(*, repo_id=REPO, version="v1", revision=None, expected_fingerprint=None, cache_dir=None):
    """Verify an already pinned local snapshot, or resolve/download before verification."""
    _check(repo_id, version)
    destination = Path(cache_dir or os.environ.get("PETTRACE_TRAINING_CACHE", "datasets/build/hub-training"))
    destination = destination.resolve() / repo_id.replace("/", "--")
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision) and expected_fingerprint:
        cached = destination / revision / "versions" / version
        if (cached / "prepared.json").is_file():
            try:
                prepared = load_prepared(cached)
            except FileNotFoundError:
                pass  # Resume missing files from the same immutable commit.
            else:
                if prepared.fingerprint != expected_fingerprint:
                    raise ValueError("Training dataset fingerprint does not match the requested release")
                return prepared, _receipt(repo_id, version, revision, prepared.fingerprint)
    commit = HfApi().dataset_info(repo_id, revision=revision or version).sha
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("HF did not resolve the training dataset to a commit")
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision) and commit != revision:
        raise ValueError("HF resolved a different commit than the requested revision")
    destination = destination / commit
    # local_dir materializes regular files; Hub's symlink cache cannot be a portable snapshot root.
    root = snapshot_download(repo_id, repo_type="dataset", revision=commit, local_dir=str(destination),
                             allow_patterns=[f"versions/{version}/**"])
    prepared = load_prepared(Path(root) / "versions" / version)
    if expected_fingerprint and prepared.fingerprint != expected_fingerprint:
        raise ValueError("Training dataset fingerprint does not match the requested release")
    return prepared, _receipt(repo_id, version, commit, prepared.fingerprint)


@verification_session()
def load_dataset(version, *, cache_dir="datasets/build/pettrace-training", drive_root=None):
    """Load a refined version from local cache, mounted Drive, or its published Hub release.

    Returns (PreparedDataset, provenance). Never acquires raw sources or prepares data.
    The first Hub resolution is recorded locally; later calls keep that exact commit.
    """
    from pettrace.data.checkpoints import BackupNotFoundError, DriveBackup

    _check(REPO, version)
    cache = Path(cache_dir).resolve()
    pointer = cache / f"{version}.json"

    def reference():
        root, source = _local_reference(pointer)
        if source["repo"] != REPO or source["version"] != version:
            raise ValueError("Cached training reference does not match the requested dataset")
        return root, source

    source = None
    if pointer.is_file():
        root, source = reference()
        try:
            return load_local_training(pointer)
        except FileNotFoundError:
            if source.get("commit"):
                prepared, _ = load_training(version=version, revision=source["commit"],
                                            expected_fingerprint=source["fingerprint"], cache_dir=cache / ".hub")
                return prepared, source
            if drive_root is None:
                raise

    if drive_root is not None:
        drive = Path(drive_root).resolve()
        if not drive.is_dir():
            raise FileNotFoundError("Mount Drive before loading its refined datasets")
        backup = DriveBackup({"prepared": cache, "data": cache.parent / "raw",
                              "reference": cache.parent / "hub-training"}, drive / "checkpoints/preparation")
        if source is None:
            try:
                backup.restore(files={f"prepared/{version}.json"})
            except BackupNotFoundError:
                pass
            else:
                root, source = reference()
        if source is not None:
            prefix = f"prepared/{root.relative_to(cache).as_posix()}"
            backup.restore(files={f"{prefix}/prepared.json"})
            manifest = json.loads((root / "prepared.json").read_text())
            if manifest.get("fingerprint") != source["fingerprint"]:
                raise ValueError("Drive snapshot does not match the requested fingerprint")
            backup.restore(files={f"{prefix}/{name}" for name in manifest["files"]})
            return load_local_training(pointer)

    prepared, source = load_training(version=version, cache_dir=cache / ".hub")
    source = {**source, "path": prepared.root.relative_to(cache).as_posix()}
    _record_reference(source, pointer)
    return prepared, source
