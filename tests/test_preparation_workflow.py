import json
from types import SimpleNamespace

import pytest

from pettrace.data import workflow
from pettrace.data.curation import sha256


@pytest.fixture
def job(tmp_path):
    drive = tmp_path / 'drive'
    drive.mkdir()
    return workflow.TrainingPreparation(drive_root=drive, work_dir=tmp_path / 'work')


def test_refined_fast_path_needs_no_gpu_zip_credentials_or_network(job, monkeypatch):
    job.version_path.parent.mkdir(parents=True)
    job.version_path.write_text('existing immutable receipt')
    prepared = SimpleNamespace(summary={'processing': {name: {'requested_pct': 50} for name in ('dogs-world', 'lcw')}})
    monkeypatch.setattr(workflow, 'load_local_training', lambda _: (prepared, {}))
    monkeypatch.setattr(workflow, 'configure_access', lambda *a, **kw: pytest.fail('No credentials needed'))
    monkeypatch.setattr(workflow, 'stage_archive', lambda *a, **kw: pytest.fail('No ZIP needed'))
    monkeypatch.setattr('pettrace.data.screening_gpu.check_cuda_environment', lambda: pytest.fail('No GPU needed'))
    assert job.prepare() is prepared
    assert job.restore() is prepared
    assert job.version_path.read_text() == 'existing immutable receipt'


def test_gpu_preflight_fails_before_copying_raw(job, monkeypatch):
    monkeypatch.setattr(workflow, 'stage_archive', lambda *a: pytest.fail('No ZIP before preflight'))
    def unavailable():
        raise RuntimeError('No CUDA device')
    monkeypatch.setattr('pettrace.data.screening_gpu.check_cuda_environment', unavailable)
    with pytest.raises(RuntimeError, match='No CUDA'):
        job.prepare()
    assert job.prepared is None
    assert not job.version_path.exists()


def test_methods_backup_interruptions_and_skip_complete_acquisitions(job, monkeypatch):
    from pettrace.evaluation.research import PUBLIC_REPO, PUBLIC_COMMIT, PUBLIC_VERSION
    job.data_root.mkdir(parents=True)
    for name in ('dogs-world', 'lcw'):
        (job.data_root / f'{name}-50-seed42.json').write_text(json.dumps({'extraction_pending': False}))
    reference = job.reference_root / PUBLIC_REPO.replace('/', '--') / PUBLIC_COMMIT / 'versions' / PUBLIC_VERSION
    reference.mkdir(parents=True)
    (reference / 'manifest.csv').write_text('fixture')
    (reference / 'prepared.json').write_text('fixture verified by mocked loader')
    monkeypatch.setattr('pettrace.data.screening_gpu.check_cuda_environment', lambda: None)
    monkeypatch.setattr(workflow, 'configure_access', lambda *a, **kw: pytest.fail('Pinned cached reference needs no credentials'))
    monkeypatch.setattr(workflow, 'stage_archive', lambda *a: pytest.fail('Completed sources do not recopy ZIPs'))
    monkeypatch.setattr(workflow, 'load_training', lambda **kw: (SimpleNamespace(root=reference), {}))
    monkeypatch.setattr('huggingface_hub.configure_http_backend', lambda **kw: None)
    detector = job.work_dir / 'external-model'
    cached = job.data_root / 'animal-screening-cache/detection.json'
    calls = []
    def prepare(config, *, checkpoint):
        calls.append(config)
        assert config.screening_provider == 'cuda' and config.source_archives == {}
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text('partial progress')
        raise KeyboardInterrupt()
    monkeypatch.setattr(workflow, 'prepare', prepare)
    with pytest.raises(KeyboardInterrupt):
        job.prepare(detector_dir=detector)
    assert job.prepared is None and not job.version_path.exists()
    receipts = [json.loads(p.read_text()) for p in job.backup.destination.glob('*.json')]
    key = 'data/animal-screening-cache/detection.json'
    assert any(r['files'].get(key, {}).get('sha256') == sha256(cached) for r in receipts)
    # Retry uses the same checkpoint object and commits a version only after preparation succeeds.
    ready = SimpleNamespace(root=job.output_root / 'ready')
    def completed(config, *, checkpoint):
        ready.root.mkdir(parents=True)
        (ready.root / 'prepared.json').write_text('verified fixture')
        return ready
    monkeypatch.setattr(workflow, 'prepare', completed)
    monkeypatch.setattr(workflow, 'record_version', lambda *a, **kw: job.version_path.write_text('named version'))
    assert job.prepare(detector_dir=detector) is ready
    assert job.version_path.read_text() == 'named version'
    receipts = [json.loads(p.read_text()) for p in job.backup.destination.glob('*.json')]
    assert any('prepared/v2.50.json' in r['files'] for r in receipts)


def test_publish_cannot_trigger_preparation(job, monkeypatch):
    monkeypatch.setattr(workflow, 'prepare', lambda *a, **kw: pytest.fail('Publish cannot prepare data'))
    with pytest.raises(FileNotFoundError, match='before publishing'):
        job.publish()


@pytest.mark.parametrize('kwargs', [{'source_percent': 25}, {'workers': 0}, {'screening_provider': 'auto'}])
def test_bad_configuration_fails_before_work(tmp_path, kwargs):
    with pytest.raises(ValueError):
        workflow.TrainingPreparation(drive_root=tmp_path, work_dir=tmp_path / 'work', **kwargs)
