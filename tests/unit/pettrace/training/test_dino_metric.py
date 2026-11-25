import numpy as np
import pytest
import torch

from pettrace.training.dino_metric import Metric, fit


def test_identity_and_reject_test_rows(tmp_path):
    x = torch.nn.functional.normalize(torch.randn(4, 1024), dim=-1)
    assert torch.allclose(Metric()(x), x, atol=1e-6)
    rows = [dict(split='test') for _ in x]
    with pytest.raises(ValueError, match='train/val'):
        fit(x.numpy(), rows, tmp_path, 45)


def test_training_keeps_best_validation_and_checkpoint(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(1)
    rows=[]
    for split, count in [('train', 8), ('val', 2)]:
        for species in ('cat', 'dog'):
            for animal in range(count):
                for photo in range(2):
                    rows.append(dict(split=split, species=species, animal=f'{split}-{species}-{animal}',
                                     sha256=f'{split}-{species}-{animal}-{photo}'))
    x = rng.normal(size=(len(rows), 1024)).astype('float32')
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    model, result = fit(x, rows, tmp_path, 45, epochs=2, steps=2)
    assert result['best']['mAP'] >= result['initial']['mAP']
    assert result['best_epoch'] in (0, 1, 2)
    saved = torch.load(tmp_path/'metric.pt', weights_only=True)
    assert torch.equal(saved['weight'], model.weight)
    assert (tmp_path/'result.json').exists()
