import copy

import numpy as np
import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from pettrace.training.dinov2 import Config, Encoder, fit
from pettrace.training.lora import load_adapter


def backbone():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    return Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=1,
                                  num_attention_heads=4, image_size=28, patch_size=14))


@pytest.mark.parametrize('mode', ['frozen', 'projection', 'lora', 'combined'])
def test_initial_equivalence_and_trainable_sets(mode):
    base = backbone()
    control = Encoder(copy.deepcopy(base), 'frozen', Config()).eval()
    model = Encoder(base, mode, Config()).eval()
    pixels = torch.randn(4, 3, 28, 28)
    assert torch.allclose(model(pixels), control(pixels), atol=1e-6)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert ('head.weight' in names) == (mode in ('projection', 'combined'))
    assert any(n.endswith('.a') for n in names) == (mode in ('lora', 'combined'))
    assert all(n == 'head.weight' or n.endswith(('.a', '.b')) for n in names)
    if names:
        model(pixels)[:, 0].sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_epoch_zero_checkpoint_and_restore(tmp_path):
    config = Config(epochs=1, steps_per_epoch=1, identities_per_batch=2,
                    images_per_identity=2, learning_rate=0)
    model = Encoder(backbone(), 'combined', config).eval()
    rows = [dict(animal=f'{split}-{sp}-{i}', species=sp, split=split, sha256=f'{split}-{sp}-{i}-{j}')
            for split in ('train', 'val') for sp in ('cat', 'dog') for i in range(2) for j in range(2)]
    pixels = np.random.default_rng(2).normal(size=(len(rows), 3, 28, 28)).astype('float32')
    indices = {r['sha256']: i for i, r in enumerate(rows)}

    class Preprocessor:
        def pixels(self, subset):
            return pixels[[indices[r['sha256']] for r in subset]]

    model, result = fit(model, rows, Preprocessor(), tmp_path, config)
    assert result['best_epoch'] == 0
    before = model(torch.tensor(pixels[:2])).detach()
    fresh = Encoder(backbone(), 'combined', config).eval()
    load_adapter(fresh, tmp_path/'adapter.pt')
    assert torch.allclose(before, fresh(torch.tensor(pixels[:2])), atol=1e-6)
    with pytest.raises(FileExistsError):
        fit(model, rows, Preprocessor(), tmp_path, config)
    with pytest.raises(ValueError, match='train/val'):
        fit(model, [{**rows[0], 'split': 'test'}], Preprocessor(), tmp_path, config)


def test_sequential_lora_preserves_parent_and_roundtrips(tmp_path):
    from pettrace.training.triplet import Head
    from pettrace.training.lora import _adapter_state
    head = Head(hidden=16, dimension=32)
    base = backbone()
    parent = Encoder(copy.deepcopy(base), 'mlp', Config(), copy.deepcopy(head)).eval()
    model = Encoder(base, 'mlp_lora', Config(), copy.deepcopy(head)).eval()
    pixels = torch.randn(4, 3, 28, 28)
    assert torch.allclose(parent(pixels), model(pixels), atol=1e-6)
    before = copy.deepcopy(model.head.state_dict())
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all(n.endswith(('.a', '.b')) for n in names)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model(pixels)[:, 0].sum().backward()
    optimizer.step()
    assert all(torch.equal(v, model.head.state_dict()[k]) for k, v in before.items())
    checkpoint = tmp_path/'adapter.pt'
    torch.save({'state_dict': _adapter_state(model)}, checkpoint)
    fresh = Encoder(backbone(), 'mlp_lora', Config(), copy.deepcopy(head)).eval()
    load_adapter(fresh, checkpoint)
    assert torch.allclose(model(pixels), fresh(pixels), atol=1e-6)
    with pytest.raises(ValueError, match='trained E3'):
        Encoder(backbone(), 'mlp_lora', Config())


def test_shared_mlp_trainer_supports_dino_dimension(tmp_path):
    from pettrace.training import triplet
    rows = [dict(animal=f'{split}-{sp}-{i}', species=sp, split=split)
            for split in ('train', 'val') for sp in ('cat', 'dog') for i in range(2) for j in range(2)]
    x = np.random.default_rng(4).normal(size=(len(rows), 1024)).astype('float32')
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    config = triplet.Config(epochs=1, steps_per_epoch=1, identities_per_batch=2, images_per_identity=2)
    head, result = triplet.fit(x, rows, tmp_path, config, 'cpu', callback=lambda _: None, dimension=1024)
    saved = torch.load(tmp_path/'head.pt', weights_only=True)
    assert saved['dimension'] == 1024 and result['best_epoch'] == 1
    fresh = triplet.Head(dimension=1024)
    fresh.load_state_dict(saved['state_dict'])
    assert torch.allclose(head(torch.tensor(x)), fresh(torch.tensor(x)))
    assert head.layers[0].out_features == 256
    with pytest.raises(ValueError, match='test data'):
        triplet.fit(x, [{**r, 'split': 'test'} for r in rows], tmp_path, config, dimension=1024)
