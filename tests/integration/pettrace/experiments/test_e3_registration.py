import json
from types import SimpleNamespace

import pytest

from pettrace.experiments.e3 import register_candidate


def test_registration_retries_reuse_version(tmp_path):
    versions=[]
    def create(name, source, run_id, tags):
        version=SimpleNamespace(version='7',source=source,run_id=run_id,status='READY')
        versions.append(version)
        return version
    client=SimpleNamespace(get_run=lambda _:SimpleNamespace(info=SimpleNamespace(status='FINISHED',artifact_uri='mlflow-artifacts:/3/run/artifacts')),
        search_model_versions=lambda _:versions,create_model_version=create,
        get_model_version=lambda *_:versions[0],log_artifact=lambda *_,**kw:None)
    import mlflow
    original = mlflow.MlflowClient
    mlflow.MlflowClient = lambda: client
    receipt = tmp_path / 'publication.json'
    try:
        receipt.write_text(json.dumps({'published': False, 'run_id': 'run', 'public_run_id': 'public'}))
        with pytest.raises(RuntimeError, match='Publish both domains'):
            register_candidate(tmp_path, 'run', 'public')
        assert not versions
        receipt.write_text(json.dumps({'published': True, 'run_id': 'run', 'public_run_id': 'public'}))
        register_candidate(tmp_path, 'run', 'public', experiment='E4')
        register_candidate(tmp_path, 'run', 'public')
        assert len(versions)==1
        assert versions[0].source.endswith('/report/model')
        assert json.loads((tmp_path/'registration.json').read_text())['alias_promoted'] is False
    finally:
        mlflow.MlflowClient = original
