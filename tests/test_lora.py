import hashlib

import pytest

# ruff: noqa: E402 -- skip this optional training suite before importing torch-dependent modules.
torch = pytest.importorskip("torch")
from torch import nn

from pettrace.training.lora import (
    Config,
    LoRALinear,
    PreparedPreprocessor,
    fit,
    inject_lora,
    load_frozen_e3_head,
    merge_lora,
    trainable_parameters,
)
from pettrace.training.triplet import Head


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)


def test_lora_starts_equivalent_and_only_targets_are_trainable():
    model = Attention().eval()
    inputs = torch.randn(3, 4)
    expected = model(inputs).detach()
    assert inject_lora(model) == ["q_proj", "v_proj"]
    torch.testing.assert_close(model(inputs), expected)
    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.v_proj, LoRALinear)
    assert not any(parameter.requires_grad for parameter in model.k_proj.parameters())
    assert trainable_parameters(model) == 2 * (8 * 4 + 4 * 8)


def test_merge_preserves_adapted_output():
    model = Attention().eval()
    inject_lora(model, rank=2, alpha=4)
    with torch.no_grad():
        model.q_proj.b.normal_()
        model.v_proj.b.normal_()
    inputs = torch.randn(3, 4)
    expected = model(inputs).detach()
    assert merge_lora(model) == ["q_proj", "v_proj"]
    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    torch.testing.assert_close(model(inputs), expected, rtol=1e-5, atol=1e-6)


def test_e3_head_hash_is_a_required_boundary(tmp_path):
    checkpoint = tmp_path / "head.pt"
    torch.save({"state_dict": Head().state_dict(), "config": {"hidden": 256}}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    head = load_frozen_e3_head(checkpoint, digest)
    assert not head.training
    assert not any(parameter.requires_grad for parameter in head.parameters())
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_e3_head(checkpoint, "0" * 64)


def test_e4_protocol_defaults_are_fixed():
    assert Config().learning_rate == 1e-4
    assert (Config().rank, Config().alpha) == (8, 16)
    assert (Config().identities_per_batch, Config().images_per_identity) == (8, 4)


def test_fit_rejects_test_photos_before_training():
    with pytest.raises(ValueError, match="test data is forbidden"):
        fit(nn.Linear(1, 1), [{"split": "test"}], None, ".", device="cpu")


def test_preprocessor_requires_matching_detector(tmp_path):
    model = tmp_path / "detector.onnx"
    model.write_bytes(b"detector")
    detector = type("Detector", (), {"session": type("Session", (), {"_model_path": str(model)})(),
                                      "margin": 0.1})()
    pipeline = type("Pipeline", (), {"detector": detector,
                                      "spec": {"steps": [{"op": "detect_crop", "file": "detector.onnx"}]}})()
    with pytest.raises(ValueError, match="different detector"):
        PreparedPreprocessor(pipeline, {"config": {"detector_sha256": "0" * 64}})
