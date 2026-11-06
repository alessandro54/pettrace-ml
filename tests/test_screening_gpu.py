import io
import json
import sys
import tarfile
from itertools import cycle
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pettrace.data import screening, screening_gpu as cuda
from pettrace.data.curation import sha256


class Detector:
    size, classes, input_name = 640, [15, 16], 'images'

    def __init__(self):
        self.calls = 0
        self.session = SimpleNamespace(run=self.run)

    def letterbox(self, image):
        return np.asarray(image)[None], 1, 0, 0

    def run(self, _, inputs):
        self.calls += 1
        raw = np.zeros((1, 84, 1), dtype=np.float32)
        raw[0, :4, 0] = (8, 8, 6, 6)
        raw[0, 19 if inputs['images'][0, 0, 0, 0] == 40 else 20, 0] = .95
        return [raw]


@pytest.mark.parametrize('version', ['12.8', '13.0', '13.1'])
def test_cuda_preflight_accepts_supported_colab_cuda(monkeypatch, version):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
                            version=SimpleNamespace(cuda=version))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(cuda.ort, '__version__', '1.31.0')
    monkeypatch.setattr(cuda.sys, 'platform', 'linux')
    monkeypatch.setattr(cuda.platform, 'machine', lambda: 'x86_64')
    assert cuda.check_cuda_environment() is torch


@pytest.mark.parametrize('version', ['11.8', '12.6', '14.0', None])
def test_cuda_preflight_rejects_unsupported_cuda(monkeypatch, version):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
                            version=SimpleNamespace(cuda=version))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(cuda.ort, '__version__', '1.31.0')
    monkeypatch.setattr(cuda.sys, 'platform', 'linux')
    monkeypatch.setattr(cuda.platform, 'machine', lambda: 'x86_64')
    with pytest.raises(RuntimeError, match=r'CUDA 12\.8\+ or CUDA 13\.x'):
        cuda.check_cuda_environment()


@pytest.fixture
def photos(tmp_path):
    rows = []
    for species, marker in [('cat', 40), ('dog', 80)]:
        for number in range(3):
            path = tmp_path / f'{species}{number}.png'
            pixels = np.full((32, 32, 3), marker, dtype=np.uint8)
            pixels[-1, -1] = number
            Image.fromarray(pixels).save(path)
            rows.append(dict(path=str(path), source=species, species=species, animal=species, sha256=sha256(path)))
    return rows


@pytest.fixture
def faster_gpu(monkeypatch):
    # Clock controls measured latency only; real image preprocessing/NMS/cache code still executes.
    times = cycle((0, .02, 1, 1.005))
    monkeypatch.setattr(cuda, 'perf_counter', lambda: next(times))
    monkeypatch.setattr(cuda, 'check_cuda_environment', lambda: SimpleNamespace(
        version=SimpleNamespace(cuda='13.0')))
    gpu = Detector()
    monkeypatch.setattr(cuda, 'cuda_detector', lambda *a: (gpu, {'provider': 'CUDAExecutionProvider'}))
    return gpu


def test_reuses_cpu_cache_and_resumes_cuda_without_bulk_reinference(tmp_path, photos, faster_gpu):
    cpu, cache = Detector(), tmp_path / 'cache'
    kwargs = dict(detector_sha256='frozen')
    screening.screen_animals(photos[:3], cpu, cache, **kwargs)
    original = {p.name: p.read_bytes() for p in cache.glob('*.json')}
    callbacks = []
    rejected, report = cuda.screen_cuda(photos, cpu, cache, 'model', checkpoint=lambda: callbacks.append(1), **kwargs)
    assert not rejected and report['counts'] == {'single_expected_animal': 6}
    assert report['gpu_validation']['passed']
    assert {r['execution_provider'] for r in report['records']} == {'CPUExecutionProvider', 'CUDAExecutionProvider'}
    assert [r['path'] for r in report['records']] == [r['path'] for r in photos]
    # Six validation images + one warmup, then only the three pending images.
    assert faster_gpu.calls == 10
    assert cpu.calls == 10  # 3 original + 7 validation, no repeat CPU bulk inference.
    assert original == {p.name: p.read_bytes() for p in cache.glob('*.json')}
    assert callbacks
    previous = faster_gpu.calls
    cuda.screen_cuda(photos, cpu, cache, 'model', **kwargs)
    assert faster_gpu.calls - previous == 7  # Validation only; all bulk detections now cached.


def test_validation_failure_stops_before_bulk_and_keeps_evidence(tmp_path, photos, faster_gpu, monkeypatch):
    original = screening.animal_boxes

    def changed(detector, *args, **kwargs):
        boxes = original(detector, *args, **kwargs)
        if detector is faster_gpu:
            boxes[0]['class_id'] = 16 if boxes[0]['class_id'] == 15 else 15
        return boxes

    monkeypatch.setattr(screening, 'animal_boxes', changed)
    with pytest.raises(ValueError, match='differs'):
        cuda.screen_cuda(photos, Detector(), tmp_path / 'cache', 'model', detector_sha256='frozen')
    saved = next((tmp_path / 'cache').glob('cuda/*/gpu-validation.json'))
    assert json.loads(saved.read_text())['passed'] is False
    assert [p.name for p in saved.parent.glob('*.json')] == ['gpu-validation.json']


def test_cpu_gpu_speed_and_numeric_tolerances_are_enforced(tmp_path, photos, faster_gpu, monkeypatch):
    slow_times = cycle((0, .01, 1, 1.02))
    monkeypatch.setattr(cuda, 'perf_counter', lambda: next(slow_times))
    with pytest.raises(RuntimeError, match='not faster'):
        cuda.validate_equivalence(photos, Detector(), faster_gpu, tmp_path / 'slow.json')
    assert not json.loads((tmp_path / 'slow.json').read_text())['passed']
    original = screening.animal_boxes
    def shifted(detector, *a, **kw):
        boxes = original(detector, *a, **kw)
        if detector is faster_gpu:
            boxes[0]['box'][0] += .2
        return boxes
    monkeypatch.setattr(screening, 'animal_boxes', shifted)
    with pytest.raises(ValueError, match='differs'):
        cuda.validate_equivalence(photos, Detector(), faster_gpu, tmp_path / 'shift.json')


def test_corrupt_cpu_cache_is_not_silently_recomputed(tmp_path, photos, faster_gpu):
    cache, cpu = tmp_path / 'cache', Detector()
    screening.screen_animals(photos, cpu, cache, detector_sha256='frozen')
    saved = next(cache.glob('*.json'))
    saved.write_text('{}')
    with pytest.raises(ValueError, match='cache mismatch'):
        cuda.screen_cuda(photos, cpu, cache, 'model', detector_sha256='frozen')
    assert saved.read_text() == '{}'
    assert faster_gpu.calls == 7  # Validation only.


@pytest.mark.parametrize(('version', 'major'), [('12.8', '12'), ('13.0', '13')])
def test_plugin_is_hash_verified_and_cannot_extract_arbitrary_paths(tmp_path, monkeypatch, version, major):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as tar:
        data = b'provider-binary'
        member = tarfile.TarInfo('../../libonnxruntime_providers_cuda.so')
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    payload = archive.getvalue()
    urls = []
    monkeypatch.setattr(cuda, 'urlopen', lambda url, **kw: urls.append(url) or io.BytesIO(payload))
    with pytest.raises(ValueError, match='archive checksum'):
        cuda._plugin(tmp_path / 'plugin', version)
    import hashlib
    monkeypatch.setitem(cuda.PLUGIN_SPECS, major,
                        {**cuda.PLUGIN_SPECS[major], 'sha256': hashlib.sha256(payload).hexdigest()})
    library = cuda._plugin(tmp_path / 'plugin', version)
    assert f'cuda{major}_' in urls[-1]
    assert library.parent == tmp_path / 'plugin' / f'cuda{major}' and library.read_bytes() == data
    monkeypatch.setattr(cuda, 'urlopen', lambda *a, **kw: pytest.fail('Should reuse verified plugin'))
    assert cuda._plugin(tmp_path / 'plugin', version) == library
    library.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        cuda._plugin(tmp_path / 'plugin', version)


@pytest.mark.parametrize('provider', ['CPUExecutionProvider', 'CUDAExecutionProvider'])
def test_session_checks_actual_conv_placement(tmp_path, monkeypatch, provider):
    library = tmp_path / 'plugin.so'
    library.write_bytes(b'plugin')
    monkeypatch.setattr(cuda, '_plugin', lambda *a: library)
    monkeypatch.setattr(cuda, '_REGISTERED', False)
    monkeypatch.setattr(cuda, 'check_cuda_environment', lambda: SimpleNamespace(
        cuda=SimpleNamespace(get_device_name=lambda i: 'fixture'), version=SimpleNamespace(cuda='12.8')))
    monkeypatch.setattr(cuda.ort, 'register_execution_provider_library', lambda *a: None)
    monkeypatch.setattr(cuda.ort, 'get_ep_devices', lambda: [SimpleNamespace(ep_name='CUDAExecutionProvider')])
    options = SimpleNamespace(add_session_config_entry=lambda *a: None, add_provider_for_devices=lambda *a: None)
    monkeypatch.setattr(cuda.ort, 'SessionOptions', lambda: options)
    assignment = SimpleNamespace(ep_name=provider, get_nodes=lambda: [SimpleNamespace(op_type='Conv')])
    session = SimpleNamespace(disable_fallback=lambda: None, get_provider_graph_assignment_info=lambda: [assignment])
    monkeypatch.setattr(cuda.ort, 'InferenceSession', lambda *a, **kw: session)
    if provider == 'CPUExecutionProvider':
        with pytest.raises(RuntimeError, match='not all assigned to CUDA'):
            cuda.cuda_detector(Detector(), 'model', tmp_path)
    else:
        detector, evidence = cuda.cuda_detector(Detector(), 'model', tmp_path)
        assert detector.session is session and evidence['graph_nodes'] == {'CUDAExecutionProvider': 1}


def test_periodic_backup_saves_completed_results_without_scanning_the_cache(tmp_path, photos, faster_gpu, monkeypatch):
    from pathlib import Path
    from pettrace.data.checkpoints import DriveBackup
    cpu, cache = Detector(), tmp_path / 'cache'
    screening.screen_animals(photos[:3], cpu, cache, detector_sha256='frozen')
    backup = DriveBackup({'screening': cache}, tmp_path / 'drive', interval=0)
    backup(force=True)
    monkeypatch.setattr(Path, 'rglob', lambda *a: pytest.fail('Periodic screening backups must not walk directories'))
    _, report = cuda.screen_cuda(photos, cpu, cache, 'model', detector_sha256='frozen', checkpoint=backup)
    for path in cache.glob('cuda/*/*.json'):
        key = 'screening/' + path.relative_to(cache).as_posix()
        assert key in backup._saved
    assert all(t >= 0 for t in report['timings_seconds'].values())
    assert report['timings_seconds']['total'] >= report['timings_seconds']['checkpoints']
    restored = tmp_path / 'restored'
    DriveBackup({'screening': restored}, tmp_path / 'drive').restore()
    assert len(list(restored.glob('*.json'))) == 3
    assert len(list(restored.glob('cuda/*/*.json'))) == 4  # Three pending detections + validation.
