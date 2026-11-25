"""The E3 API must preserve a caller-selected width across training and restore."""
import numpy as np
import pytest
import torch

from pettrace.training.triplet import Config, Head, fit


@pytest.mark.parametrize('width', [256, 512])
def test_dino_width_survives_fit_and_checkpoint(tmp_path, width):
    torch.set_num_threads(2)
    rows = [dict(animal=f'{split}-{sp}-{i}', species=sp, split=split)
            for split in ('train', 'val') for sp in ('cat', 'dog')
            for i in range(2) for _ in range(2)]
    x = np.random.default_rng(1).normal(size=(len(rows), 1024)).astype('float32')
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    config = Config(hidden=width, epochs=1, steps_per_epoch=2,
                    identities_per_batch=2, images_per_identity=2)
    head, _ = fit(x, rows, tmp_path, config, device='cpu',
                  callback=lambda _: None, dimension=1024)
    saved = torch.load(tmp_path/'head.pt', map_location='cpu', weights_only=True)
    restored = Head(saved['config']['hidden'], saved['dimension'])
    restored.load_state_dict(saved['state_dict'])
    assert saved['config']['hidden'] == width and saved['dimension'] == 1024
    assert sum(p.numel() for p in head.parameters()) == 2*1024*width + width + 1024
    with torch.no_grad():
        expected = head(torch.from_numpy(x))
        actual = restored(torch.from_numpy(x))
    assert actual.shape == (len(rows), 1024)
    assert torch.equal(expected, actual)
    assert torch.allclose(actual.norm(dim=1), torch.ones(len(rows)), atol=1e-6)
