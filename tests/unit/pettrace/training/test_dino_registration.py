import json
from types import SimpleNamespace as NS

import pytest

from pettrace.experiments.dinov2 import register_research_bundle


def test_retry_after_receipt_upload_failure_reuses_version(tmp_path):
    (tmp_path / 'publication.json').write_text(json.dumps([
        dict(run_id='public', published=True), dict(run_id='private', published=True)]))
    versions, uploads = [], []
    def create(name, source, run_id, tags):
        assert tags['serving_compatible'] == 'false'
        version = NS(version='1', source=source, run_id=run_id, status='READY')
        versions.append(version)
        return version
    def upload(*args):
        uploads.append(args)
        if len(uploads) == 1:
            raise OSError('upload interrupted')
    client = NS(
        get_run=lambda _: NS(info=NS(status='FINISHED', artifact_uri='mlflow-artifacts:/3/run/artifacts')),
        list_artifacts=lambda *_: [NS(path='research-model/'+f) for f in ('model.json', 'golden.npy')],
        create_registered_model=lambda _: None, search_model_versions=lambda _: versions,
        create_model_version=create, get_model_version=lambda *_: versions[0], log_artifact=upload)
    with pytest.raises(OSError, match='interrupted'):
        register_research_bundle(tmp_path, 'private', client)
    receipt = register_research_bundle(tmp_path, 'private', client)
    assert len(versions) == 1
    assert receipt['alias_promoted'] is False
    assert receipt['source'] == 'mlflow-artifacts:/3/run/artifacts/research-model'
    client.get_run = lambda _: NS(info=NS(status='FAILED'))
    with pytest.raises(RuntimeError, match='Finish all'):
        register_research_bundle(tmp_path, 'private', client)
