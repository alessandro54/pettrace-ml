"""The queue must not train on partial preparation or changed held-out photos."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('e3_volume', Path(__file__).parents[1] / 'tools/run_e3_volume.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def test_preparation_waits_for_success_and_writer_exit(tmp_path):
    state = tmp_path / 'preparation-status.json'
    state.write_text(json.dumps({'state': 'preparing'}))
    assert queue.preparation_ready(tmp_path, alive=True) is None
    with pytest.raises(RuntimeError, match='stopped'):
        queue.preparation_ready(tmp_path, alive=False)
    series = {str(p): {'version': f'v4.{p}', 'fingerprint': str(p)} for p in (20, 50, 100)}
    state.write_text(json.dumps({'state': 'prepared', 'versions': {p: v['fingerprint'] for p, v in series.items()}}))
    receipt = tmp_path / 'volume-series.json'
    receipt.write_text(json.dumps(series))
    assert queue.preparation_ready(tmp_path, alive=True) is None
    assert queue.preparation_ready(tmp_path, alive=False) == series
    series['50']['fingerprint'] = 'changed'
    receipt.write_text(json.dumps(series))
    with pytest.raises(ValueError, match='fingerprint'):
        queue.preparation_ready(tmp_path, alive=False)
    del series['100']
    receipt.write_text(json.dumps(series))
    with pytest.raises(ValueError, match='Incomplete'):
        queue.preparation_ready(tmp_path, alive=False)
    state.write_text(json.dumps({'state': 'failed'}))
    with pytest.raises(RuntimeError, match='failed'):
        queue.preparation_ready(tmp_path, alive=True)


def test_fixed_benchmarks_and_no_identity_or_image_leakage():
    def row(animal, split, digest):
        return {'animal': animal, 'species': 'cat', 'split': split, 'sha256': digest}
    public = [row('old-train', 'train', 'old'), row('val', 'val', 'v'), row('test', 'test', 't')]
    rows = [row('new-train', 'train', 'new'), *public[1:]]
    queue.verify_splits(rows, public, {'private'})
    with pytest.raises(ValueError, match='validation/test'):
        queue.verify_splits(rows[:-1], public, {'private'})
    for extra in (row('test', 'train', 'other'), row('other', 'train', 'v'), row('other', 'train', 'private')):
        with pytest.raises(ValueError, match='overlap'):
            queue.verify_splits([*rows, extra], public, {'private'})
