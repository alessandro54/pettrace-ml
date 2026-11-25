"""DINO study evidence, component selection and mandatory training publication."""

import hashlib
import json
from pathlib import Path

import numpy as np

from pettrace.data.verification import sha256
from pettrace.tracking import production_context


def frozen_features(pipeline, rows, output, identity):
    """Checkpoint extraction batches; cache identity includes ordered photo hashes."""
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    signature = dict(identity=identity, rows=[{k: r[k] for k in ('sha256', 'animal', 'species', 'split')} for r in rows])
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    (folder / 'identity.json').write_text(json.dumps(signature, indent=2))
    parts = []
    from PIL import Image, ImageOps
    for offset in range(0, len(rows), 32):
        path = folder / f'{key}-{offset}.npy'
        checksum = path.with_suffix('.sha256')
        if path.exists() and checksum.exists():
            if sha256(path) != checksum.read_text():
                raise ValueError('Corrupt DINO feature cache')
            values = np.load(path, allow_pickle=False)
        else:
            images = []
            for row in rows[offset:offset + 32]:
                if sha256(row['path']) != row['sha256']:
                    raise ValueError('Photo changed since preparation')
                with Image.open(row['path']) as source:
                    images.append(ImageOps.exif_transpose(source).convert('RGB'))
            values = pipeline.embed(images)
            np.save(path, values)
            checksum.write_text(sha256(path))
        if values.shape != (min(32, len(rows) - offset), 1024) or not np.isfinite(values).all():
            raise ValueError('Invalid DINO features')
        parts.append(values)
        if offset % 3200 == 0:
            print(f'DINO features: {offset + len(values)}/{len(rows)}', flush=True)
    return np.concatenate(parts)


def select_family(candidates, control):
    """All three predeclared seeds per family, with the frozen control eligible."""
    families = {}
    for mode in ('projection', 'lora', 'combined'):
        runs = [c for c in candidates if c['mode'] == mode]
        if sorted(c['seed'] for c in runs) != [45, 46, 47]:
            raise ValueError('Every family requires exactly seeds 45, 46, 47')
        scores = np.array([[c['validation'][k] for k in ('mAP', 'rank1')] for c in runs])
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite validation scores')
        mean = scores.mean(axis=0)
        families[mode] = dict(mAP=float(mean[0]), rank1=float(mean[1]))
    eligible = [m for m, v in families.items() if v['mAP'] > control['mAP'] and v['rank1'] >= control['rank1']]
    if not eligible:
        return dict(mode='frozen', seed=None, family_means=families)
    mode = min(eligible, key=lambda m: (-families[m]['mAP'], m))
    winner = min((c for c in candidates if c['mode'] == mode), key=lambda c: (-c['validation']['mAP'], c['seed']))
    return dict(mode=mode, seed=winner['seed'], family_means=families)


def publish_training(folder, design, metrics):
    """Publish/retry saved evidence, never train as part of publication."""
    folder = Path(folder)
    client, experiment = production_context()
    path = folder / 'training-publication.json'
    if path.exists():
        receipt = json.loads(path.read_text())
        run_id = receipt['run_id']
    else:
        run_id = client.create_run(experiment.experiment_id, tags={
            'experiment': 'DINO', 'purpose': 'training',
            'mlflow.runName': f"DINO {design['mode']} seed {design['seed']}"}).info.run_id
        receipt = dict(run_id=run_id, published=False)
        path.write_text(json.dumps(receipt, indent=2))
    (folder / 'design.json').write_text(json.dumps(design, indent=2))
    for k, v in metrics.items():
        client.log_metric(run_id, 'val.' + k, float(v))
    for artifact in sorted(folder.iterdir()):
        if artifact.is_file():
            client.log_artifact(run_id, str(artifact), 'training')
    client.set_terminated(run_id, 'FINISHED')
    downloaded = Path(client.download_artifacts(run_id, 'training/design.json', str(folder / 'verified')))
    if sha256(downloaded) != sha256(folder / 'design.json') or client.get_run(run_id).info.status != 'FINISHED':
        raise RuntimeError('Training publication verification failed')
    receipt.update(published=True, design_sha256=sha256(downloaded))
    path.write_text(json.dumps(receipt, indent=2))
    client.log_artifact(run_id, str(path), 'training')
    return run_id
