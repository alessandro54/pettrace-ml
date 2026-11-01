import json
from pathlib import Path
from types import SimpleNamespace
from textwrap import dedent


def test_registration_retries_reuse_version(tmp_path):
    n=json.loads(Path('notebooks/e3_mlp_head_triplet_20.ipynb').read_text())
    source=''.join(next(c for c in n['cells'] if c.get('id')=='e3-28')['source'])
    source=dedent(source[source.index('    if REGISTER_E3:'):])
    versions=[]
    def create(name, source, run_id, tags):
        version=SimpleNamespace(version='7',source=source,run_id=run_id,status='READY')
        versions.append(version)
        return version
    client=SimpleNamespace(get_run=lambda _:SimpleNamespace(info=SimpleNamespace(artifact_uri='mlflow-artifacts:/3/run/artifacts')),
        search_model_versions=lambda _:versions,create_model_version=create,
        get_model_version=lambda *_:versions[0],log_artifact=lambda *_,**kw:None)
    scope=dict(REGISTER_E3=True,mlflow=SimpleNamespace(MlflowClient=lambda:client),run_id='run',REPORT=tmp_path,json=json)
    exec(source,scope)
    exec(source,scope)
    assert len(versions)==1
    assert versions[0].source.endswith('/report/model')
    assert json.loads((tmp_path/'registration.json').read_text())['alias_promoted'] is False
