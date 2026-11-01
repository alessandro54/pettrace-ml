import gzip
import hashlib
import json

import mlflow
import pytest

from pettrace.artifact_upload import CHUNK_BYTES, compact_json_uploads, log_report_artifacts


def test_large_json_roundtrip_and_originals_preserved(tmp_path, monkeypatch):
    import os
    payload = json.dumps({'data': os.urandom(CHUNK_BYTES * 2).hex()}).encode()
    (tmp_path / 'sources.json').write_bytes(payload)
    (tmp_path / 'model.onnx').write_bytes(b'model')
    uploaded = {}
    def upload(path, artifact_path=None, run_id=None):
        from pathlib import Path
        assert artifact_path == 'report' and run_id == 'existing'
        uploaded[Path(path).name] = Path(path).read_bytes()
    monkeypatch.setattr(mlflow, 'log_artifact', upload)
    log_report_artifacts(tmp_path, 'report', 'existing')
    manifest = json.loads(uploaded['sources.json.transport.json'])
    assert len(manifest['parts']) > 1
    chunks = []
    for part in manifest['parts']:
        chunk = uploaded[part['name']]
        assert len(chunk) <= CHUNK_BYTES
        assert hashlib.sha256(chunk).hexdigest() == part['sha256']
        chunks.append(chunk)
    assert gzip.decompress(b''.join(chunks)) == payload
    assert manifest['sha256'] == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / 'sources.json').read_bytes() == payload
    assert uploaded['model.onnx'] == b'model'
    assert 'sources.json' not in uploaded


def test_patch_restores_after_failure():
    original = mlflow.log_artifacts
    with pytest.raises(RuntimeError):
        with compact_json_uploads():
            assert mlflow.log_artifacts is log_report_artifacts
            raise RuntimeError('upload failed')
    assert mlflow.log_artifacts is original
