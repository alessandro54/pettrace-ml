"""Compact indexed inputs on CPU before allocating a screening GPU."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import yaml

from pettrace.data.checkpoints import DriveBackup
from pettrace.data.curation import EXTENSIONS
from pettrace.data.lookup import load_candidates
from pettrace.data.preparation import DETECTOR_SHA256, PrepareConfig, _atomic_json, _safe_file
from pettrace.data.review import _read
from pettrace.data.verification import sha256
from pettrace.data.volume import CAPACITY


def indexed_inventory(config):
    """Verify recorded source membership; excluded photo bytes need not be materialized."""
    if not config.candidate_index or config.source_percent not in (20, 50, 100):
        raise ValueError("Indexed input requires a candidate index and source percentage")
    root = Path(config.data_root).resolve()
    index = Path(config.candidate_index)
    load_candidates(index, index / 'review.csv', minimum_photos=config.minimum_photos)
    rows = _read(index / 'review.csv')
    sources, sampled = [], {}
    for source, capacity in CAPACITY.items():
        receipt = json.loads((root / f'{source}-{config.source_percent}-seed{config.seed}.json').read_text())
        expected = {'source_percent': config.source_percent, 'seed': config.seed,
                    'selection': 'sha256_order_full_archive_photos_v1'}
        count = capacity['photos'] * config.source_percent // 100
        if (receipt.get('config') != expected or receipt.get('extraction_pending') is not False
                or any(receipt.get(k) != capacity[k] for k in ('ref', 'version', 'archive_index_sha256'))
                or receipt.get('source_photos') != capacity['photos'] or receipt.get('sampled_photos') != count):
            raise ValueError('Source receipt does not match the pinned volume')
        photos = {}
        for name, digest in receipt['file_sha256'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name:
                raise ValueError('Unsafe source inventory path')
            if relative.suffix.lower() in EXTENSIONS:
                photos[str(root / 'source-cache' / source / relative)] = (source, digest)
        if len(photos) != count:
            raise ValueError('Incomplete source photo inventory')
        sampled.update(photos)
        sources.append(receipt)
    for row in rows:
        if sampled.get(row['path']) != (row['source'], row['sha256']):
            raise ValueError('Review photo differs from the recorded source volume or working directory')
    labelled = {r['path'] for r in rows}
    skipped = [{'path': path, 'source': source, 'reason': 'missing_or_ambiguous_identity'}
               for path, (source, _) in sampled.items() if path not in labelled]
    return rows, {'sources': sources, 'skipped_paths': skipped, 'candidate_files': len(rows)}


def _runtime_config(size):
    from pettrace.data.screening_gpu import cache_config
    return json.loads(json.dumps(cache_config(SimpleNamespace(size=size), DETECTOR_SHA256, 0.25, 0.5)))


def _cache_config(detector):
    spec = yaml.safe_load((Path(detector) / 'pipeline.yaml').read_text())
    steps = [s for s in spec['steps'] if s['op'] == 'detect_crop' and s.get('enabled')]
    if len(steps) != 1 or sha256(_safe_file(Path(detector), steps[0]['file'])) != DETECTOR_SHA256:
        raise ValueError('Expected the frozen E2 detector')
    return _runtime_config(int(steps[0].get('input_size', 640)))


def bundle(job):
    if job.candidate_index is None:
        raise ValueError('Stage requires a verified data_lookup candidate index')
    identity = sha256(Path(job.candidate_index) / 'lookup.json')
    directory = job.drive_root / 'checkpoints/indexed' / identity
    return DriveBackup(job.backup.roots, directory / 'blocks'), directory / 'ready.json', identity


def _read_ready(job):
    backup, path, identity = bundle(job)
    if not path.is_file():
        raise FileNotFoundError('Run preparation.stage() on CPU first; no complete compact input exists')
    ready = json.loads(path.read_text())
    if (ready.get('schema') != 1 or ready.get('candidate_index_sha256') != identity
            or ready.get('work_dir') != str(job.work_dir) or ready.get('source_percent') != job.source_percent):
        raise ValueError('Compact input does not match this index, volume or working directory')
    if _runtime_config(ready['cache_config']['input_size']) != ready['cache_config']:
        raise ValueError('Use the same pinned screening environment as the CPU staging session')
    for name in ready['files']:
        backup._target(name)
    return backup, ready


def restore_staged(job):
    """Read only the completed compact bundle; never fall back to historical raw backups."""
    backup, ready = _read_ready(job)
    backup.restore(files=set(ready['files']))
    if any(sha256(backup._target(name)) != digest for name, digest in ready['files'].items()):
        raise ValueError('Compact input changed after staging')
    detector = backup._target(ready['detector'])
    if _cache_config(detector) != ready['cache_config']:
        raise ValueError('Screening runtime changed; stage with the same pinned environment before using GPU')
    print(f"Entrada compacta: {ready['photos']} candidatas · {ready['bytes'] / 2**30:.2f} GiB", flush=True)
    return backup._target(ready['reference']), detector


def stage(job, *, detector_dir=None):
    """Use local photos first, recover missing candidates, then save a compact Drive input.

    Run on CPU with the pinned screening dependencies. Absolute review paths are
    preserved: both sessions must use the same work_dir (normally /content).
    """
    compact, ready_path, identity = bundle(job)
    if ready_path.is_file():
        _, ready = _read_ready(job)
        catalog = compact.inventory()
        if any(catalog.get(name, {}).get('sha256') != digest for name, digest in ready['files'].items()):
            raise ValueError('Incomplete compact checkpoint inventory')
        print(f"Compact input already prepared: {ready['photos']} candidates. Staging is not repeated.", flush=True)
        return ready
    index = Path(job.candidate_index)
    candidates, _ = load_candidates(index, index / 'review.csv')
    catalog = job.backup.inventory()

    def recover(names, *, optional=False):
        names = set(names)
        if optional:
            names &= catalog.keys()
        missing = {name for name in names if not job.backup._target(name).is_file()}
        if missing:
            job.backup.restore(files=missing)

    receipts = {f'data/{source}-{job.source_percent}-seed42.json' for source in CAPACITY}
    recover(receipts)
    config = PrepareConfig(data_root=str(job.data_root), detector_dir='', source_percent=job.source_percent,
                           candidate_index=job.candidate_index)
    indexed_inventory(config)
    photo_names = {f"data/{Path(r['path']).relative_to(job.data_root).as_posix()}" for r in candidates}
    print(f"CPU: {len(candidates)} candidates; selectively restoring missing photos…", flush=True)
    recover(photo_names)
    for row in candidates:
        if sha256(row['path']) != row['sha256']:
            raise ValueError('Candidate photo changed; preserve evidence and repair its source')

    from pettrace.evaluation.research import PUBLIC_REPO, PUBLIC_COMMIT, PUBLIC_VERSION
    prefix = f"reference/{PUBLIC_REPO.replace('/', '--')}/{PUBLIC_COMMIT}/versions/{PUBLIC_VERSION}/"
    # The reference is small and immutable; exclude every other benchmark/version.
    recover({name for name in catalog if name.startswith(prefix)
             or name.startswith('data/detector-models/')}, optional=True)
    reference, detector = job._reference_and_detector(detector_dir)
    if not Path(detector).resolve().is_relative_to(job.data_root):
        raise ValueError('Stage the detector inside data_root/detector-models')
    cache_config = _cache_config(detector)
    from pettrace.data.screening_gpu import _key, PLUGIN_SHA256, OPTIONS
    namespace = _key('validated-cuda-v1', {**cache_config, 'plugin': PLUGIN_SHA256, 'options': OPTIONS})
    keys = {_key(row['sha256'], cache_config) for row in candidates}
    cache_names = {f'data/animal-screening-cache/{key}.json' for key in keys}
    cache_names |= {f'data/animal-screening-cache/cuda/{namespace}/{key}.json' for key in keys}
    recover(cache_names, optional=True)
    cache_names = {name for name in cache_names if job.backup._target(name).is_file()}
    paths = [job.backup._target(name) for name in photo_names | receipts | cache_names]
    local_index = job.data_root / 'candidate-index' / identity
    local_index.mkdir(parents=True, exist_ok=True)
    index_receipt = json.loads((index / 'lookup.json').read_text())
    for name in {'lookup.json'} | index_receipt['files'].keys():
        target = local_index / name
        shutil.copyfile(index / name, target)
        paths.append(target)
    paths += [p for root in (reference.root, Path(detector)) for p in root.rglob('*') if p.is_file()]
    files = {f'{label}/{p.relative_to(root).as_posix()}': sha256(p)
             for p in paths for label, root in compact.roots.items() if p.is_relative_to(root)}
    # Recover completed packs after an interrupted stage, without repeating their upload.
    existing = compact.inventory().keys() & files.keys()
    if existing:
        compact.restore(files=existing)
    compact(force=True, paths=paths)
    catalog = compact.inventory()
    if any(catalog.get(name, {}).get('sha256') != digest for name, digest in files.items()):
        raise ValueError('Staged files changed; no completed input was published')
    if sha256(index / 'lookup.json') != identity:
        raise ValueError('Candidate index changed during staging')
    load_candidates(index, index / 'review.csv')
    ready = {'schema': 1, 'candidate_index_sha256': identity, 'work_dir': str(job.work_dir),
             'source_percent': job.source_percent, 'photos': len(candidates), 'cache_entries': len(cache_names),
             'cache_config': cache_config, 'files': files, 'bytes': sum(p.stat().st_size for p in paths),
             'reference': f'reference/{reference.root.relative_to(job.reference_root).as_posix()}',
             'index': f'data/{local_index.relative_to(job.data_root).as_posix()}',
             'detector': f'data/{Path(detector).relative_to(job.data_root).as_posix()}'}
    _atomic_json(ready_path, ready)  # Completion is published only after every confirmed pack.
    print(f"CPU: input ready · {ready['photos']} photos · {ready['cache_entries']} cached detections · "
          f"{ready['bytes'] / 2**30:.2f} GiB. The GPU stage can now run.", flush=True)
    return ready
