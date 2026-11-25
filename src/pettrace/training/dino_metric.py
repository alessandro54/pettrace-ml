"""D1: identity-initialized metric on frozen DINO CLS; train/val only."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pettrace.evaluation.metrics import retrieval
from pettrace.training.triplet import batch_hard_triplet, pk_batch, seed_all


class Metric(nn.Module):
    def __init__(self, dimension=1024):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dimension))

    def forward(self, x):
        return F.normalize(F.linear(x, self.weight), dim=-1)


def scores(model, features, rows):
    with torch.no_grad():
        embeddings = model(torch.as_tensor(features, dtype=torch.float32)).numpy()
    result = retrieval(embeddings, [r['animal'] for r in rows], [r['species'] for r in rows])
    return {key: float(np.mean([result[s][key] for s in ('cat', 'dog')])) for key in ('mAP', 'rank1')}


def fit(features, rows, output, seed, *, epochs=30, steps=100):
    """Keep epoch zero eligible; never accept test rows or overlapping identities."""
    if features.shape != (len(rows), 1024) or not np.isfinite(features).all():
        raise ValueError('Expected finite N x 1024 features')
    if not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-3):
        raise ValueError('Features must be normalized')
    if any(r['split'] not in ('train', 'val') for r in rows):
        raise ValueError('Only train/val allowed')
    ti = [i for i, r in enumerate(rows) if r['split'] == 'train']
    vi = [i for i, r in enumerate(rows) if r['split'] == 'val']
    train, val = [rows[i] for i in ti], [rows[i] for i in vi]
    if not train or not val or {r['animal'] for r in train} & {r['animal'] for r in val}:
        raise ValueError('Separate train/val identities required')
    if any({r['species'] for r in subset} != {'cat', 'dog'} for subset in (train, val)):
        raise ValueError('Both species required')
    if len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image bytes')
    seed_all(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = Metric()
    initial = scores(model, features[vi], val)
    best, best_epoch, state, stale = initial, 0, copy.deepcopy(model.state_dict()), 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ids = {a: i for i, a in enumerate(sorted({r['animal'] for r in train}))}
    labels = np.array([ids[r['animal']] for r in train])
    species = [r['species'] for r in train]
    rng = np.random.default_rng(seed)
    x, y = torch.tensor(features[ti]), torch.tensor(labels)
    history, seen = [], set()
    for epoch in range(1, epochs + 1):
        losses = []
        for step in range(steps):
            idx = pk_batch(labels, species, 8, 4, rng, ('cat', 'dog')[step % 2])
            loss, _ = batch_hard_triplet(model(x[idx]), y[idx], margin=0.2)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            seen.update(idx.tolist())
            losses.append(float(loss.detach()))
        current = scores(model, features[vi], val)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), **current,
                            unique_train_photos=len(seen), steps=epoch * steps))
        if current['mAP'] > best['mAP']:
            best, best_epoch, state, stale = current, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        torch.save(state, output / 'metric.pt')
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        print(seed, history[-1], flush=True)
        if stale >= 7:
            break
    result = dict(seed=seed, initial=initial, best=best, best_epoch=best_epoch, history=history)
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    model.load_state_dict(state)
    return model.eval(), result
