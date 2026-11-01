"""Lossless transport for JSON evidence exceeding the tracking proxy upload limit."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from contextlib import contextmanager

CHUNK_BYTES = 512 * 1024


def log_report_artifacts(local_dir, artifact_path=None, run_id=None):
    import mlflow

    root = Path(local_dir)
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        parent = (Path(artifact_path or '') / relative.parent).as_posix()
        parent = None if parent == '.' else parent
        if path.suffix != '.json' or path.stat().st_size <= CHUNK_BYTES:
            mlflow.log_artifact(str(path), artifact_path=parent, run_id=run_id)
            continue
        # Keep local evidence untouched; only its remote representation changes.
        with tempfile.TemporaryDirectory() as tmp:
            packed = Path(tmp) / 'payload.gz'
            digest = hashlib.sha256()
            size = 0
            with path.open('rb') as source, packed.open('wb') as target:
                with gzip.GzipFile(filename='', fileobj=target, mode='wb', mtime=0) as output:
                    while block := source.read(CHUNK_BYTES):
                        digest.update(block)
                        size += len(block)
                        output.write(block)
            parts = []
            with packed.open('rb') as source:
                while block := source.read(CHUNK_BYTES):
                    part = Path(tmp) / f'{path.name}.gz.part{len(parts):05d}'
                    part.write_bytes(block)
                    mlflow.log_artifact(str(part), artifact_path=parent, run_id=run_id)
                    parts.append({'name': part.name, 'size': len(block),
                                  'sha256': hashlib.sha256(block).hexdigest()})
            manifest = Path(tmp) / f'{path.name}.transport.json'
            manifest.write_text(json.dumps({'schema': 1, 'original': path.name,
                'size': size, 'sha256': digest.hexdigest(), 'encoding': 'gzip', 'parts': parts,
                'restore': 'Concatenate parts in listed order, then gzip-decompress; verify original SHA-256.'}, indent=2))
            mlflow.log_artifact(str(manifest), artifact_path=parent, run_id=run_id)


@contextmanager
def compact_json_uploads():
    """Allow an already-running v0.2.4 kernel to retry without replacing its package."""
    import mlflow

    original = mlflow.log_artifacts
    mlflow.log_artifacts = log_report_artifacts
    try:
        yield
    finally:
        mlflow.log_artifacts = original
