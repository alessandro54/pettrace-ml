"""Queue the predeclared v4.20/50/100 E3 series after Colab preparation exits.

Run with the existing Colab credentials and mounted Drive; never starts preparation.
Retry with the same output to reuse completed fits and publication receipts.
"""
import csv
from dataclasses import asdict
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import time

DRIVE = Path('/content/drive/MyDrive/PetTrace')
RECEIPTS = DRIVE / 'data-lookup/inventory100'
PERCENTAGES = (20, 50, 100)
RUNTIME_SHA = '906d040f5ea4881e4750b553908afd65cf3bc88a64d8e741951e9f9ffd64aa3a'
SOURCE_SHA = 'a15ddf99952a0d76f6e6b3cd1a8fb2dccafe3c10d09478be86e3bc69ad5b0b26'
PRIVATE_COMMIT = '8b12db0dce41040a821f9792bea5b352342263ce'
BASE_HASHES = {
    'encoder.onnx': '4773e649274ca8774c4a63d45f0438fe598ee7ba67e65def97537aa0eca9a576',
    'detector.onnx': 'da8f9b9ebea5157cdcefa7bea9fd782249ebdf119f917c07d64e5b41942b9c9d',
}


def preparation_ready(receipts, alive):
    """Require the final success receipt, all versions, and no remaining writer."""
    status = json.loads((receipts / 'preparation-status.json').read_text())
    if status['state'] == 'failed':
        raise RuntimeError('Preparation failed; training was not started')
    if status['state'] != 'prepared':
        if not alive:
            raise RuntimeError('Preparation stopped without a successful receipt')
        return None
    series = json.loads((receipts / 'volume-series.json').read_text())
    if set(series) != {str(p) for p in PERCENTAGES}:
        raise ValueError('Incomplete volume-series receipt')
    for percent in PERCENTAGES:
        item = series[str(percent)]
        if (item['version'] != f'v4.{percent}' or not item['fingerprint']
                or status['versions'][str(percent)] != item['fingerprint']):
            raise ValueError('Prepared version/fingerprint differs from the final receipt')
    return None if alive else series


def verify_splits(rows, public_rows, private_hashes):
    def signature(items):
        return sorted((r['animal'], r['species'], r['split'], r['sha256']) for r in items)
    held_out = [r for r in public_rows if r['split'] in {'val', 'test'}]
    if signature(r for r in rows if r['split'] in {'val', 'test'}) != signature(held_out):
        raise ValueError('Public validation/test changed')
    train = [r for r in rows if r['split'] == 'train']
    if ({r['animal'] for r in train} & {r['animal'] for r in held_out}
            or {r['sha256'] for r in train} & {r['sha256'] for r in held_out}
            or {r['sha256'] for r in rows} & private_hashes):
        raise ValueError('Training/benchmark overlap')


def run_series(output, series, record):
    # Import the heavy runtime only once preparation has released its RAM/GPU.
    import torch
    import mlflow
    from huggingface_hub import snapshot_download
    from packaging.requirements import Requirement
    from pettrace import data
    from pettrace.data import benchmark
    from pettrace.data.checkpoints import DriveBackup
    from pettrace.data.curation import sha256
    from pettrace.data.near_copies import configuration
    from pettrace.data.verification import verification_session
    from pettrace.data.versions import load_training
    from pettrace.data.volume import training_volume
    from pettrace.evaluation.core import load_model
    from pettrace.evaluation.research import PUBLIC_REPO, PUBLIC_VERSION, PUBLIC_COMMIT, PUBLIC_FINGERPRINT, public_test_dataset
    from pettrace.training.features import feature_cache
    from pettrace.training.triplet import Config
    from pettrace.tracking import PRODUCTION_URI, production_context, source_identity
    from pettrace.artifact_upload import compact_json_uploads
    import repeat_e3

    runtime = Path('/content/e3-runtime.txt')
    if sha256(runtime) != RUNTIME_SHA or version('pettrace-ml') != '0.2.14':
        raise ValueError('Package or runtime lock changed')
    for line in runtime.read_text().splitlines():
        if line and not line[0].isspace() and not line.startswith('#'):
            req = Requirement(line.rstrip('\\ '))
            if (req.marker is None or req.marker.evaluate()) and version(req.name) not in req.specifier:
                raise ValueError(f'{req.name} differs from the E3 runtime lock')
    source = source_identity()
    if source['source_sha256'] != SOURCE_SHA or not torch.cuda.is_available():
        raise ValueError('Pinned package source and CUDA are required')
    plan = json.loads((output / 'volume-plan.json').read_text())
    if (plan['versions'] != [f'v4.{p}' for p in PERCENTAGES]
            or plan['training_seeds'] != list(repeat_e3.SEEDS)
            or plan['package'] != 'pettrace-ml==0.2.14'):
        raise ValueError('Prospective plan changed')
    mlflow.set_tracking_uri(PRODUCTION_URI)
    mlflow.set_registry_uri(PRODUCTION_URI)
    production_context()
    base = load_model('models:/pettrace-embedder/2')
    if not base.golden_ok or base.sha256 != BASE_HASHES or base.pipeline.detector is None:
        raise ValueError('Frozen E2 model differs')
    private_root = snapshot_download(benchmark.REPO, repo_type='dataset', revision=PRIVATE_COMMIT)
    private = benchmark.from_dir(private_root, repo=benchmark.REPO, version='v4', commit=PRIVATE_COMMIT)
    if private.fingerprint != 'db002d9b5355989f':
        raise ValueError('Private benchmark differs')
    cache = Path('/content/datasets/public/e3-body/feature-cache')
    features_backup = DriveBackup({'features': cache}, DRIVE / 'checkpoints/features')
    database = cache / f'frozen-e2-{repeat_e3.MODEL_KEY}.sqlite3'
    member = f'features/{database.name}'
    # A running session may contain newer embeddings than its last Drive snapshot.
    if not database.exists() and member in features_backup.inventory():
        features_backup.restore(files={member})
    inputs, prepared_versions = {}, {}
    with verification_session():
        public, public_source = load_training(repo_id=PUBLIC_REPO, version=PUBLIC_VERSION,
            revision=PUBLIC_COMMIT, expected_fingerprint=PUBLIC_FINGERPRINT,
            cache_dir='/content/datasets/build/hub-training')
        private_hashes = {sha256(p.path) for p in private.photos}
        if private_hashes & {r['sha256'] for r in public.rows}:
            raise ValueError('Public/private benchmark overlap')
        common = None
        # Validate every input before extracting features or fitting any head.
        for percent in PERCENTAGES:
            item = series[str(percent)]
            prepared, provenance = data.load(item['version'],
                cache_dir='/content/datasets/build/pettrace-training', drive_root=DRIVE)
            receipt = json.loads((prepared.root / 'prepared.json').read_text())
            index = Path(item['candidate_index'])
            if (prepared.fingerprint != item['fingerprint']
                    or sha256(index / 'lookup.json') != item['candidate_index_sha256']
                    or receipt['inputs']['candidate_index_sha256'] != item['candidate_index_sha256']):
                raise ValueError('Refined version or candidate index changed')
            policy = json.loads((index / 'lookup.json').read_text())['inputs'].copy()
            policy.pop('review_sha256')
            if ('identity_review_sha256' in policy or policy['minimum_photos'] != 4
                    or policy['near_copies'] != configuration()):
                raise ValueError('Automatic curation policy changed')
            if common is not None and policy != common:
                raise ValueError('Curation differs between volumes')
            common = policy
            with (index / 'review.csv').open() as stream:
                if any(row['decision'] for row in csv.DictReader(stream)):
                    raise ValueError('Manual review is outside this series')
            verify_splits(prepared.rows, public.rows, private_hashes)
            folder = output / 'inputs' / item['version']
            folder.mkdir(parents=True, exist_ok=True)
            for filename in ('manifest.csv', 'review.csv', 'prepared.json', 'audit.json', 'summary.json',
                             'excluded.csv', 'sources.json', 'animal-screening.json'):
                shutil.copy2(prepared.root / filename, folder / filename)
            shutil.copy2(index / 'lookup.json', folder / 'lookup.json')
            shutil.copy2(runtime, folder / 'requirements-e3-runtime.txt')
            metrics, params = training_volume(prepared, percent, folder)
            prepared_versions[percent] = (prepared, provenance, folder, metrics, params)
        with features_backup:
            for percent, (prepared, provenance, folder, metrics, params) in prepared_versions.items():
                record('features', source_percent=percent)
                rows = [r for r in prepared.rows if r['split'] in {'train', 'val'}]
                values, digest = feature_cache(base.pipeline, rows, cache, base.dir, checkpoint=features_backup)
                del values
                config = {'experiment': 'E3', 'source_sample_pct': percent, 'dataset_coverage': metrics,
                          'coverage_params': params, 'seed': 42, 'training': asdict(Config()),
                          'prepared_dataset_fingerprint': prepared.fingerprint, 'training_dataset_source': provenance,
                          'dataset': private.name, 'dataset_commit': private.commit, 'dataset_fingerprint': private.fingerprint,
                          'base_model': base.uri, 'base_model_sha256': base.sha256, 'cache_sha256': digest,
                          'manifest_sha256': sha256(folder / 'manifest.csv'), 'review_sha256': sha256(folder / 'review.csv'),
                          'animal_screening_sha256': sha256(folder / 'animal-screening.json'), 'source': source,
                          'runtime_lock_sha256': RUNTIME_SHA, 'log_to_mlflow': True, 'register_model': True}
                repeat_e3.write_json(folder / 'training-config.json', config)
                inputs[f'v4.{percent}'] = folder
        record('training_and_publication')
        context = {'base_model': base, 'public_ds': public_test_dataset(public, public_source), 'ds': private,
                   'FEATURE_CACHE_ROOT': cache, 'REGISTER_E3': True, 'LOG_TO_MLFLOW': True,
                   'launcher': {'sha256': sha256(__file__), 'volume_plan_sha256': sha256(output / 'volume-plan.json')}}
        with compact_json_uploads():
            repeat_e3.run(context, output, prepared_inputs=inputs)
    publications = {}
    for percent in PERCENTAGES:
        for seed in repeat_e3.SEEDS:
            name = f'v4.{percent}-seed{seed}'
            result = json.loads((output / name / 'publication.json').read_text())
            if not result['published'] or set(result['runs']) != {'public', 'private'} or not result['registry_version']:
                raise RuntimeError('Incomplete publication')
            publications[name] = result
    record('published', publications=publications)


def main(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    def record(state, **details):
        value = {'state': state, 'updated_unix': time.time(), 'output': str(output), **details}
        for path in (output / 'status.json', RECEIPTS / 'experiment-status.json'):
            partial = path.with_suffix('.tmp')
            partial.write_text(json.dumps(value, indent=2) + '\n')
            partial.replace(path)
        print(f'E3 series: {state}', flush=True)

    # One launcher for the shared feature cache, including launches with another output.
    with Path('/content/e3-volume-runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for filename in ('run_e3_volume.py', 'repeat_e3.py', 'volume-plan.json'):
                source = Path(__file__).with_name(filename)
                target = output / filename
                if target.exists() and target.read_bytes() != source.read_bytes():
                    raise ValueError('Launcher or plan changed; preserve existing run evidence')
                shutil.copy2(source, target)
            record('waiting_preparation')
            while True:
                pid = int(Path('/content/volume-preparation.pid').read_text())
                process = Path(f'/proc/{pid}/cmdline')
                try:
                    alive = b'continue_volume_preparation.py' in process.read_bytes()
                except FileNotFoundError:
                    alive = False
                series = preparation_ready(RECEIPTS, alive)
                if series is not None:
                    break
                time.sleep(30)
            record('verifying')
            (output / 'volume-series.json').write_text(json.dumps(series, indent=2) + '\n')
            run_series(output, series, record)
        except BaseException as error:
            record('failed', error_type=type(error).__name__)
            raise


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python run_e3_volume.py /content/reports/e3-volume-RUN')
    main(sys.argv[1])
