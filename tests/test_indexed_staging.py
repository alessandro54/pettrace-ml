import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from pettrace.data import preparation, staging, volume
from pettrace.data.lookup import load_candidates, lookup_duplicates
from pettrace.data.preparation import load_prepared, prepare
from pettrace.data.sources import REVIEW_FIELDS, discover_body_ids, write_review
from pettrace.data.verification import sha256
from pettrace.data.workflow import TrainingPreparation
from test_e3_prepare import setup_pipeline


@pytest.fixture
def indexed_job(tmp_path, monkeypatch):
    config = setup_pipeline(tmp_path, monkeypatch)
    reference = prepare(config)
    monkeypatch.setattr('pettrace.evaluation.research.PUBLIC_FINGERPRINT', reference.fingerprint)
    drive = tmp_path / 'drive'
    drive.mkdir()
    job = TrainingPreparation(drive_root=drive, work_dir=tmp_path / 'work', screening_provider='cpu',
                              candidate_index=drive / 'index', version='v3.50')
    job.data_root.mkdir(parents=True)
    model = job.data_root / 'detector-models/frozen'
    shutil.copytree(config.detector_dir, model)
    monkeypatch.setattr(staging, 'DETECTOR_SHA256', config.expected_detector_sha256)
    ref = job.reference_root / 'reference'
    shutil.copytree(reference.root, ref)
    monkeypatch.setattr(job, '_reference_and_detector', lambda _: (load_prepared(ref), model))
    make_config = job._config
    monkeypatch.setattr(job, '_config', lambda *a, **kw: replace(make_config(*a, **kw),
                        expected_detector_sha256=config.expected_detector_sha256, visual_sample_per_species=1))
    shutil.copytree(Path(config.data_root) / 'animal-screening-cache', job.data_root / 'animal-screening-cache')
    rows = []
    for source, directory in config.source_roots.items():
        destination = job.data_root / 'source-cache' / source
        shutil.copytree(directory, destination)
        found, _ = discover_body_ids(destination, source)
        rows.extend(found)
        capacity = {**volume.CAPACITY[source], 'photos': len(found) * 2}
        monkeypatch.setitem(volume.CAPACITY, source, capacity)
        receipt = {**capacity, 'source_photos': len(found) * 2, 'sampled_photos': len(found),
                   'extraction_pending': False,
                   'config': {'source_percent': 50, 'seed': 42, 'selection': 'sha256_order_full_archive_photos_v1'},
                   'file_sha256': {Path(r['path']).relative_to(destination).as_posix(): sha256(r['path']) for r in found}}
        (job.data_root / f'{source}-50-seed42.json').write_text(json.dumps(receipt))
    raw = write_review(rows, job.data_root / 'source-50-seed42/review.csv')
    # Exclude a complete identity before staging, while retaining its inventory evidence.
    reviewed = staging._read(raw)
    for row in reviewed:
        if row['animal'] == 'lcw:cat:1':
            row['decision'] = 'drop'
    with raw.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(reviewed)
    job.candidate_index = str(lookup_duplicates(raw, drive / 'indices'))
    return job, model, ref


def test_compact_input_skips_raw_acquisition_and_keeps_results_after_runtime_loss(indexed_job, monkeypatch):
    job, model, ref = indexed_job
    config = job._config(model, ref)
    rows, sources = staging.indexed_inventory(config)
    monkeypatch.setattr(preparation, 'acquire_candidates', lambda *a, **kw: (rows, sources))
    baseline = prepare(replace(config, output_root=str(job.work_dir / 'baseline')))
    baseline_signature = {(r['animal'], r['sha256'], r['split']) for r in baseline.rows}
    candidates, excluded = load_candidates(job.candidate_index, Path(job.candidate_index) / 'review.csv')
    job.backup(force=True)
    ready = job.stage()
    assert ready['photos'] == len(candidates) < len(rows)
    names = set(ready['files'])
    assert all(f"data/{Path(p).relative_to(job.data_root).as_posix()}" not in names for p in excluded)
    assert ready['cache_entries'] == len(candidates)
    # A fresh runtime must not touch even a single legacy TAR (or require discarded bytes).
    for pack in job.backup.destination.glob('*.tar'):
        pack.write_bytes(b'Unrelated historical data must not be read')
    shutil.rmtree(job.data_root)
    shutil.rmtree(job.reference_root)
    monkeypatch.setattr(preparation, 'acquire_candidates', lambda *a, **kw: pytest.fail('No raw acquisition'))
    monkeypatch.setattr(job, '_reference_and_detector', lambda *a: pytest.fail('No credentials or network'))
    result = job.prepare()
    assert {(r['animal'], r['sha256'], r['split']) for r in result.rows} == baseline_signature
    assert result.summary['processing'] == baseline.summary['processing']
    assert (result.root / 'review.csv').read_bytes() == (Path(job.candidate_index) / 'review.csv').read_bytes()
    assert all(not Path(p).exists() for p in excluded)
    assert not any(n.startswith('prepared/') for n in staging.bundle(job)[0].inventory())
    assert 'prepared/v3.50.json' in job.backup.inventory()
    # A surviving version pointer with missing image files recovers only this refined snapshot.
    shutil.rmtree(result.root)
    recovered = TrainingPreparation(drive_root=job.drive_root, work_dir=job.work_dir,
                                    candidate_index=job.candidate_index, version='v3.50').restore()
    assert recovered.fingerprint == result.fingerprint


def test_stage_completion_is_last_and_retry_reuses_confirmed_packs(indexed_job, monkeypatch):
    job, _, _ = indexed_job
    original = staging._atomic_json
    def fail(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(staging, '_atomic_json', fail)
    with pytest.raises(KeyboardInterrupt):
        job.stage()
    compact, ready, _ = staging.bundle(job)
    assert not ready.exists()
    packs = set(compact.destination.glob('*.tar'))
    assert packs
    monkeypatch.setattr(staging, '_atomic_json', original)
    job.stage()
    assert set(compact.destination.glob('*.tar')) == packs
    monkeypatch.setattr(staging, 'indexed_inventory', lambda *a: pytest.fail('Completed staging is reused'))
    job.stage()


def test_missing_staging_fails_before_gpu_or_archives(indexed_job, monkeypatch):
    job, _, _ = indexed_job
    job.screening_provider = 'cuda'
    monkeypatch.setattr('pettrace.data.screening_gpu.check_cuda_environment', lambda: pytest.fail('No GPU yet'))
    with pytest.raises(FileNotFoundError, match='stage'):
        job.prepare()


def test_changed_runtime_fails_before_gpu_and_interrupted_detections_resume(indexed_job, monkeypatch):
    job, _, _ = indexed_job
    job.stage()
    original_runtime = staging._runtime_config
    monkeypatch.setattr(staging, '_runtime_config', lambda size: {'changed': True})
    job.screening_provider = 'cuda'
    monkeypatch.setattr('pettrace.data.screening_gpu.check_cuda_environment', lambda: pytest.fail('No GPU yet'))
    with pytest.raises(ValueError, match='same pinned'):
        job.prepare()
    monkeypatch.setattr(staging, '_runtime_config', original_runtime)
    job.screening_provider = 'cpu'
    from pettrace.data import workflow
    delta = job.data_root / 'animal-screening-cache/cuda/fixture/new.json'
    def interrupted(config, checkpoint):
        delta.parent.mkdir(parents=True, exist_ok=True)
        delta.write_text('completed detection fixture')
        checkpoint(paths=[delta])
        raise KeyboardInterrupt()
    monkeypatch.setattr(workflow, 'prepare', interrupted)
    with pytest.raises(KeyboardInterrupt):
        job.prepare()
    delta.unlink()
    def resumed(config, checkpoint):
        assert delta.read_text() == 'completed detection fixture'
        raise RuntimeError('Resume verified')
    monkeypatch.setattr(workflow, 'prepare', resumed)
    with pytest.raises(RuntimeError, match='Resume verified'):
        job.prepare()


def test_indexed_inventory_rejects_changed_receipts_and_candidate_bytes(indexed_job):
    job, model, ref = indexed_job
    receipt = job.data_root / 'lcw-50-seed42.json'
    saved = receipt.read_bytes()
    value = json.loads(saved)
    value['sampled_photos'] -= 1
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='pinned volume'):
        staging.indexed_inventory(job._config(model, ref))
    receipt.write_bytes(saved)
    candidates, _ = load_candidates(job.candidate_index, Path(job.candidate_index) / 'review.csv')
    Path(candidates[0]['path']).write_bytes(b'tampered candidate')
    with pytest.raises(ValueError, match='Candidate photo changed'):
        job.stage()
    assert not staging.bundle(job)[1].exists()
