"""Reusable raw → refined preparation on a local disk with mounted Drive backups."""

import json
import os
from pathlib import Path

from pettrace.data.checkpoints import BackupNotFoundError, DriveBackup, stage_archive
from pettrace.data.preparation import (
    BASE_MODEL, DetectorAuthenticationError, PrepareConfig, acquire_detector, prepare,
)
from pettrace.data.versions import REPO, _local_reference, load_local_training, load_training, publish_training, record_version
from pettrace.data.verification import verification_session


def configure_access(names=("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"), *, force=False):
    """Read environment/Colab Secrets, then hidden prompts; never print credentials."""
    import getpass
    import warnings

    try:
        from google.colab import userdata
    except ImportError:
        userdata = None
    values = {}
    for name in names:
        refresh = force and name != "MLFLOW_TRACKING_URI"
        value = None if refresh else os.environ.get(name)
        if not value and userdata is not None and not refresh:
            try:
                value = userdata.get(name)
            except Exception:
                pass  # VS Code may not expose Colab Secrets.
        if not value:
            if name == "MLFLOW_TRACKING_URI":
                value = "https://ml.pettrace.app"
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", getpass.GetPassWarning)
                    value = getpass.getpass(f"{name}: ")
        if not value.strip():
            raise ValueError(f"Missing {name}.")
        values[name] = value
    os.environ.update(values)


def _hub_session():
    from pettrace.experiments.e1 import hub_session
    session = hub_session()
    adapter = session.get_adapter("https://huggingface.co")
    adapter.max_retries = adapter.max_retries.new(backoff_factor=60, backoff_max=300)
    return session


class TrainingPreparation:
    """Prepare one immutable source volume; no Colab import, mounting or training required.

    Drive must already be mounted. Only one process may write its checkpoint folder.
    restore() reads refined data first; prepare() also works without an explicit restore().
    """

    def __init__(self, *, drive_root, work_dir, source_percent=50, workers=4, screening_provider="cuda",
                 candidate_index=None, version=None):
        if source_percent not in (20, 50, 100) or not isinstance(workers, int) or not 1 <= workers <= 16:
            raise ValueError("Use 20/50/100 percent and 1–16 workers")
        if screening_provider not in {"cpu", "cuda"}:
            raise ValueError("Screening provider must be cpu or cuda")
        if candidate_index is not None and (not version or version.startswith("v2.")):
            raise ValueError("An indexed preparation needs a new version, for example v3.100; preserve v2 results")
        if version is not None:
            from pettrace.data.versions import _check
            _check(REPO, version)
        self.candidate_index = str(Path(candidate_index).resolve()) if candidate_index is not None else None
        self.screening_provider = screening_provider
        self.drive_root, self.work_dir = Path(drive_root).resolve(), Path(work_dir).resolve()
        if not self.drive_root.is_dir():
            raise FileNotFoundError("Mount the Drive folder before preparing data")
        self.source_percent, self.workers = source_percent, workers
        self.data_root = self.work_dir / 'datasets/public/e3-body'
        self.output_root = self.work_dir / 'datasets/build/pettrace-training'
        self.reference_root = self.work_dir / 'datasets/build/hub-training'
        self.version = version or f'v2.{source_percent}'
        self.version_path = self.output_root / f'{self.version}.json'
        self.backup = DriveBackup({'data': self.data_root, 'prepared': self.output_root,
                                  'reference': self.reference_root}, self.drive_root / 'checkpoints/preparation')
        self.prepared = None
        self._restored = False

    @verification_session()
    def restore(self):
        """Return a verified refined version, or None; never copy source ZIPs."""
        result = None
        if self.version_path.is_file():
            try:
                result, _ = load_local_training(self.version_path)
            except FileNotFoundError:
                pass  # A named version may have been only partly restored when a runtime died.
        if result is None:
            if not self.version_path.is_file():
                try:
                    self.backup.restore(files={f'prepared/{self.version}.json'})
                except BackupNotFoundError:
                    self._restored = True
                    return None
            root, receipt = _local_reference(self.version_path)
            prefix = f'prepared/{root.relative_to(self.output_root).as_posix()}'
            self.backup.restore(files={f'{prefix}/prepared.json'})
            manifest = json.loads((root / 'prepared.json').read_text())
            if manifest.get('fingerprint') != receipt['fingerprint']:
                raise ValueError('Prepared version fingerprint changed')
            self.backup.restore(files={f'{prefix}/{name}' for name in manifest['files']})
            result, _ = load_local_training(self.version_path)
        if result is not None:
            if any(result.summary.get('processing', {}).get(s, {}).get('requested_pct') != self.source_percent
                   for s in ('dogs-world', 'lcw')):
                raise ValueError("Prepared version does not match the requested source percentage")
            if self.candidate_index is not None:
                from pettrace.data.verification import sha256
                receipt = json.loads((result.root / 'prepared.json').read_text())
                if receipt['inputs'].get('candidate_index_sha256') != sha256(Path(self.candidate_index) / 'lookup.json'):
                    raise ValueError("Prepared version uses a different candidate index; use a new version")
            self.prepared = result
        self._restored = True
        return self.prepared

    @verification_session()
    def stage(self, *, detector_dir=None):
        """Build a compact indexed input on CPU; no YOLO inference or GPU allocation."""
        from pettrace.data.staging import stage
        return stage(self, detector_dir=detector_dir)

    @verification_session()
    def prepare(self, *, detector_dir=None):
        """Reuse refined data or acquire, clean, version and back it up without training."""
        if not self._restored:
            self.restore()
        if self.prepared is not None:
            return self.prepared
        if self.candidate_index is not None:
            return self._prepare_indexed()
        if self.screening_provider == "cuda":
            from pettrace.data.screening_gpu import check_cuda_environment
            check_cuda_environment()
        self.backup.restore()
        archives = {}
        for name, filename in {'dogs-world': 'dogs-world.zip', 'lcw': 'lcw.zip'}.items():
            receipt_path = self.data_root / f'{name}-{self.source_percent}-seed42.json'
            receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
            if receipt.get('extraction_pending') is False:
                continue
            archives[name] = str(stage_archive(self.drive_root / 'raw' / filename,
                                              self.work_dir / 'pettrace-archives' / filename))
        reference, detector_dir = self._reference_and_detector(detector_dir)
        # A supplied detector may be outside the backup roots; it remains hash-verified by prepare().
        paths = [reference.root]
        if Path(detector_dir).resolve().is_relative_to(self.data_root):
            paths.append(detector_dir)
        self.backup(force=True, paths=paths)
        config = self._config(detector_dir, reference.root, source_archives=archives)
        with self.backup:
            result = prepare(config, checkpoint=self.backup)
            record_version(result.root, self.output_root, version=self.version, repo_id=REPO)
        self.prepared = result
        return result

    def _reference_and_detector(self, detector_dir):
        from huggingface_hub import configure_http_backend
        from huggingface_hub.utils import disable_progress_bars
        from pettrace.evaluation.research import PUBLIC_REPO, PUBLIC_VERSION, PUBLIC_COMMIT, PUBLIC_FINGERPRINT

        reference_path = self.reference_root / PUBLIC_REPO.replace('/', '--') / PUBLIC_COMMIT / 'versions' / PUBLIC_VERSION
        if not (reference_path / 'prepared.json').is_file():
            configure_access(names=('HF_TOKEN',))
            configure_http_backend(backend_factory=_hub_session)
            disable_progress_bars()
        reference, _ = load_training(repo_id=PUBLIC_REPO, version=PUBLIC_VERSION, revision=PUBLIC_COMMIT,
                                     expected_fingerprint=PUBLIC_FINGERPRINT, cache_dir=self.reference_root)
        if detector_dir is None:
            try:
                detector_dir = acquire_detector(BASE_MODEL, self.data_root / 'detector-models')
            except DetectorAuthenticationError:
                configure_access(names=('MLFLOW_TRACKING_USERNAME', 'MLFLOW_TRACKING_PASSWORD'), force=True)
                detector_dir = acquire_detector(BASE_MODEL, self.data_root / 'detector-models')
        return reference, Path(detector_dir)

    def _config(self, detector_dir, reference, **kwargs):
        return PrepareConfig(data_root=str(self.data_root), detector_dir=str(detector_dir),
                               output_root=str(self.output_root),
                               source_percent=self.source_percent, reference_dataset=str(reference),
                               minimum_photos=4, confidence=0.25, nms_iou=0.5, seed=42,
                               allow_download=False, download_workers=self.workers,
                               screening_provider=self.screening_provider, candidate_index=self.candidate_index,
                               **kwargs)

    def _prepare_indexed(self):
        from dataclasses import replace
        from pettrace.data.staging import _read_ready, restore_staged
        compact, ready = _read_ready(self)  # Fail before GPU work or any archive transfer.
        if self.screening_provider == 'cuda':
            from pettrace.data.screening_gpu import check_cuda_environment
            check_cuda_environment()
        reference, detector = restore_staged(self)
        # Cache deltas from an interrupted screening also live in the compact namespace.
        cached = {name for name in compact.inventory() if name.startswith('data/animal-screening-cache/')}
        if cached:
            compact.restore(files=cached)
        config = replace(self._config(detector, reference, indexed_input=True),
                         candidate_index=str(compact._target(ready['index'])))
        try:
            result = prepare(config, checkpoint=compact)
            record_version(result.root, self.output_root, version=self.version, repo_id=REPO)
            self.backup(force=True, paths=[result.root, self.version_path])
        finally:
            compact(force=True, paths=[])  # Flush completed work without scanning historical raw roots.
        self.prepared = result
        return result

    def publish(self):
        """Explicit optional HF publication of an already prepared version; never starts preparation."""
        if self.prepared is None and self.restore() is None:
            raise FileNotFoundError("Prepare the refined version before publishing")
        configure_access(names=('HF_TOKEN',))
        receipt = publish_training(self.prepared.root, repo_id=REPO, version=self.version)
        path = self.output_root / f'{self.version}-hub-release.json'
        path.write_text(json.dumps(receipt, indent=2))
        self.backup(force=True, paths=[path])
        return receipt
