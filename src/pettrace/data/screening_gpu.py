"""Validated CUDA acceleration around the unchanged, cached E2 screening algorithm."""

from collections import Counter
from contextlib import redirect_stdout
from copy import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import tempfile
from time import monotonic, perf_counter
from urllib.request import urlopen

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from pettrace.data import screening
from pettrace.data.checkpoints import DriveBackup
from pettrace.data.curation import sha256
from pettrace.models import pipeline

PLUGIN_VERSION = '0.3.0'
PLUGIN_BASE = 'https://github.com/microsoft/onnxruntime/releases/download/plugin-ep-cuda/v0.3.0/'
PLUGIN_SPECS = {
    '12': {'minimum': (12, 8), 'file': 'cuda_ep_cuda12_0.3.0_linux-x64.tar.gz',
           'sha256': 'ef6864a088b8c2a74e46b18a016eac8dd47950cd3a2a224603998c4fbb09d332'},
    '13': {'minimum': (13, 0), 'file': 'cuda_ep_cuda13_0.3.0_linux-x64.tar.gz',
           'sha256': '0818121f81fc4461f9ba2b588dbcd03d62efb039a1d87169dfe74e03c37acd44'},
}
OPTIONS = {'use_tf32': '0', 'cudnn_conv_algo_search': 'HEURISTIC'}
_REGISTERED = False


def cache_config(detector, detector_sha256, confidence, iou):
    # Mirror the frozen module's key, not its implementation: editing it would invalidate CPU caches.
    return {'detector_sha256': detector_sha256, 'confidence': confidence, 'nms_iou': iou,
            'nms': 'class-agnostic', 'classes': screening.SPECIES, 'input_size': detector.size,
            'implementation_sha256': sha256(screening.__file__), 'pipeline_sha256': sha256(pipeline.__file__),
            'onnxruntime': ort.__version__, 'numpy': np.__version__, 'pillow': screening.PIL.__version__}


def _key(digest, config):
    return hashlib.sha256(json.dumps({'image': digest, 'config': config}, sort_keys=True).encode()).hexdigest()


def _plugin_spec(cuda_version):
    try:
        version = tuple(map(int, cuda_version.split('.')[:2]))
    except (AttributeError, ValueError):
        version = ()
    spec = PLUGIN_SPECS.get(str(version[0])) if version else None
    if spec is None or version < spec['minimum']:
        raise RuntimeError(f'Unsupported PyTorch CUDA {cuda_version!r}; requires CUDA 12.8+ or CUDA 13.x')
    return str(version[0]), spec


def _plugin(cache, cuda_version):
    """Download a pinned official binary; no package replacement or manual library copying."""
    major, spec = _plugin_spec(cuda_version)
    cache = Path(cache) / f'cuda{major}'
    library, receipt = cache / 'libonnxruntime_providers_cuda.so', cache / 'plugin.json'
    if library.is_file() and receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved.get('archive_sha256') != spec['sha256'] or saved.get('library_sha256') != sha256(library):
            raise ValueError('CUDA plugin checksum mismatch')
        return library
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache, prefix='.download-') as folder:
        archive_path = Path(folder) / 'plugin.tar.gz'
        with urlopen(PLUGIN_BASE + spec['file'], timeout=120) as incoming, archive_path.open('wb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        if sha256(archive_path) != spec['sha256']:
            raise ValueError('CUDA plugin archive checksum mismatch')
        with tarfile.open(archive_path) as archive:
            members = [m for m in archive if Path(m.name).name == library.name]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError('CUDA archive must contain exactly one regular provider library')
            partial = Path(folder) / library.name
            with archive.extractfile(members[0]) as incoming, partial.open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
        digest = sha256(partial)
        partial.replace(library)
        receipt.write_text(json.dumps({'archive_sha256': spec['sha256'], 'library_sha256': digest}))
    return library


def check_cuda_environment():
    """Fail before source acquisition when this runtime cannot use the pinned plugin."""
    if ort.__version__ != '1.31.0' or sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise RuntimeError('CUDA preparation requires Linux x86_64 and ONNX Runtime 1.31.0')
    import torch  # Loads the runtime's CUDA/cuDNN libraries before registering the matching plugin.
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA device. Select a GPU runtime before starting preparation.')
    _plugin_spec(torch.version.cuda)
    return torch


def cuda_detector(cpu, model_file, cache):
    """Require the pinned runtime, real CUDA devices and CUDA-assigned convolutions."""
    global _REGISTERED
    torch = check_cuda_environment()
    _, spec = _plugin_spec(torch.version.cuda)
    library = _plugin(cache, torch.version.cuda)
    if not _REGISTERED:
        ort.register_execution_provider_library('CUDAExecutionProvider', str(library))
        _REGISTERED = True
    devices = [d for d in ort.get_ep_devices() if d.ep_name == 'CUDAExecutionProvider']
    if not devices:
        raise RuntimeError('CUDA plugin registered no device; CPU fallback is not allowed')
    options = ort.SessionOptions()
    options.add_session_config_entry('session.record_ep_graph_assignment_info', '1')
    options.add_provider_for_devices([devices[0]], OPTIONS)
    gpu = copy(cpu)
    gpu.session = ort.InferenceSession(str(model_file), sess_options=options)
    gpu.session.disable_fallback()
    assignments = list(gpu.session.get_provider_graph_assignment_info())
    convolutions = [group.ep_name for group in assignments for node in group.get_nodes() if 'Conv' in node.op_type]
    if not convolutions or any(ep != 'CUDAExecutionProvider' for ep in convolutions):
        raise RuntimeError('YOLO convolutions are not all assigned to CUDA; bulk screening was not started')
    evidence = {'provider': 'CUDAExecutionProvider', 'plugin_version': PLUGIN_VERSION,
                'plugin_archive_sha256': spec['sha256'], 'library_sha256': sha256(library),
                'options': OPTIONS, 'gpu': torch.cuda.get_device_name(0), 'torch_cuda': torch.version.cuda,
                'graph_nodes': {g.ep_name: sum(len(x.get_nodes()) for x in assignments if x.ep_name == g.ep_name)
                                for g in assignments}}
    return gpu, evidence


def validate_equivalence(rows, cpu, gpu, output, *, confidence=0.25, iou=0.5):
    """Compare deterministic cat/dog samples; differences stop the bulk run and preserve evidence."""
    output = Path(output)
    evidence = {'passed': False, 'sample': [], 'confidence_atol': 1e-4, 'box_atol_pixels': 0.1,
                'limitation': 'Sample agreement is not proof of identical outputs on every image.'}
    times = {'cpu': [], 'gpu': []}
    try:
        for species in ('cat', 'dog'):
            candidates = sorted((r for r in rows if r['species'] == species), key=lambda r: r['sha256'])
            count = 0
            for row in candidates:
                if sha256(row['path']) != row['sha256']:
                    raise ValueError('Image changed before CUDA validation')
                try:
                    with Image.open(row['path']) as image:
                        image = ImageOps.exif_transpose(image).convert('RGB')
                        image.load()
                except (OSError, ValueError, Image.DecompressionBombError):
                    continue
                boxes = {}
                for name, detector in (('cpu', cpu), ('gpu', gpu)):
                    if not times[name]:
                        screening.animal_boxes(detector, image, confidence=confidence, iou=iou)  # Warm up once.
                    started = perf_counter()
                    boxes[name] = screening.animal_boxes(detector, image, confidence=confidence, iou=iou)
                    times[name].append(perf_counter() - started)
                a, b = boxes['cpu'], boxes['gpu']
                matches = (len(a) == len(b) and all(
                    x['class_id'] == y['class_id'] and abs(x['score'] - y['score']) <= 1e-4
                    and np.allclose(x['box'], y['box'], rtol=0, atol=0.1) for x, y in zip(a, b)))
                evidence['sample'].append({'image_sha256': row['sha256'], 'species': species,
                                           'cpu': a, 'gpu': b, 'matches': matches})
                if not matches:
                    raise ValueError('CPU/GPU screening differs on the validation sample; bulk run stopped')
                count += 1
                if count == 16:
                    break
            if candidates and not count:
                raise ValueError(f'No readable {species} images for CUDA validation')
        if not evidence['sample']:
            raise ValueError('No images for CUDA validation')
        evidence['median_seconds'] = {k: float(np.median(v)) for k, v in times.items()}
        evidence['screening_speedup'] = evidence['median_seconds']['cpu'] / max(evidence['median_seconds']['gpu'], 1e-12)
        if evidence['screening_speedup'] <= 1:
            raise RuntimeError('GPU was not faster in the validation sample; bulk run stopped')
        evidence['passed'] = True
    except Exception as exc:
        evidence['error'] = str(exc)
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2))
    return evidence


class _CountingSession:
    def __init__(self, session):
        self.session, self.calls = session, 0

    def run(self, *args, **kwargs):
        self.calls += 1
        return self.session.run(*args, **kwargs)


class _CacheOnlySession:
    def run(self, *args, **kwargs):
        raise RuntimeError('An expected CPU cache entry is missing; refusing to recompute it silently')


def screen_cuda(rows, cpu, cache, model_file, *, detector_sha256, confidence=0.25, iou=0.5, checkpoint=None):
    """Reuse CPU detections read-only; keep new CUDA detections in a separate cache namespace."""
    cache = Path(cache)
    if not rows:
        raise ValueError('No candidate images to screen')
    started = monotonic()
    timings = {'cache_planning': 0, 'validation': 0, 'cached_cpu': 0, 'cuda': 0, 'checkpoints': 0}
    config = cache_config(cpu, detector_sha256, confidence, iou)
    # Version this policy if CUDA computation/options change; logging changes do not invalidate results.
    _, plugin = _plugin_spec(check_cuda_environment().version.cuda)
    namespace = _key('validated-cuda-v1', {**config, 'plugin': plugin['sha256'], 'options': OPTIONS})
    gpu_cache = cache / 'cuda' / namespace
    gpu_cache.mkdir(parents=True, exist_ok=True)
    entries = {p.stem for p in cache.glob('*.json')}
    cuda_entries = {p.stem for p in gpu_cache.glob('*.json')}
    reused, remaining, cuda_cached = [], [], 0
    for row in rows:
        key = _key(row['sha256'], config)
        if key in entries:
            reused.append(row)
        else:
            remaining.append(row)
            cuda_cached += key in cuda_entries
    timings['cache_planning'] = monotonic() - started
    print(f'YOLO: {len(rows)} photos · CPU cache {len(reused)} · CUDA cache {cuda_cached} · '
          f'pending {len(remaining) - cuda_cached}', flush=True)
    validating = monotonic()
    gpu, execution = cuda_detector(cpu, model_file, cache.parent / 'detector-models/cuda-plugin')
    validation = validate_equivalence(rows, cpu, gpu, gpu_cache / 'gpu-validation.json', confidence=confidence, iou=iou)
    timings['validation'] = monotonic() - validating
    print(f"CPU/GPU: {len(validation['sample'])} photos match · measured speedup "
          f"{validation['screening_speedup']:.2f}×", flush=True)
    gpu.session = _CountingSession(gpu.session)
    cached_cpu = copy(cpu)
    cached_cpu.session = _CacheOnlySession()
    exclusions, records, completed = {}, [], 0
    console = sys.stdout
    with (gpu_cache / 'screening.log').open('a') as log:
        for group, detector, directory, origin in ((reused, cached_cpu, cache, 'CPUExecutionProvider'),
                                                  (remaining, gpu, gpu_cache, 'CUDAExecutionProvider')):
            if not group:
                continue
            seen = 0
            group_started, previous_backup_seconds = monotonic(), timings['checkpoints']

            def progress():
                nonlocal seen
                previous = seen
                seen = min(seen + 100, len(group))
                with redirect_stdout(console):
                    backing_up = monotonic()
                    if checkpoint:
                        if isinstance(checkpoint, DriveBackup):
                            # Atomic detection files are immutable after each completed block of 100.
                            paths = [directory / f"{_key(r['sha256'], config)}.json" for r in group[previous:seen]]
                            checkpoint(paths=[*paths, gpu_cache / 'gpu-validation.json'])
                        else:
                            checkpoint()
                    timings['checkpoints'] += monotonic() - backing_up
                    if seen % 1000 == 0 or seen == len(group):
                        print(f'YOLO: {completed + seen}/{len(rows)} checked · new GPU inferences '
                              f'{gpu.session.calls}', flush=True)

            with redirect_stdout(log):
                rejected, result = screening.screen_animals(group, detector, directory, detector_sha256=detector_sha256,
                                                           confidence=confidence, iou=iou, checkpoint=progress)
            label = 'cached_cpu' if origin == 'CPUExecutionProvider' else 'cuda'
            timings[label] = monotonic() - group_started - (timings['checkpoints'] - previous_backup_seconds)
            exclusions.update(rejected)
            records.extend({**r, 'execution_provider': origin} for r in result['records'])
            completed += len(group)
    by_path = {r['path']: r for r in records}
    records = [by_path[r['path']] for r in rows]
    timings['total'] = monotonic() - started
    print(f"YOLO complete: CPU cache {timings['cached_cpu']:.1f}s · CUDA {timings['cuda']:.1f}s · "
          f"backups {timings['checkpoints']:.1f}s", flush=True)
    return exclusions, {'config': config, 'photos': len(rows), 'records': records,
                        'counts': dict(Counter(r['reason'] for r in records)),
                        'execution': execution, 'gpu_validation': validation, 'timings_seconds': timings,
                        'limitation': result['limitation'] + ' ' + validation['limitation']}
