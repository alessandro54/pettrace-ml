"""LoRA adapters for E4 encoder adaptation (Hu et al., 2021)."""

import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from pettrace.training.triplet import Head


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank update."""

    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1:
            raise ValueError("LoRA requires a linear layer and rank >= 1")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, inputs):
        return self.base(inputs) + (inputs @ self.a.T @ self.b.T) * self.scale

    def merge(self):
        with torch.no_grad():
            self.base.weight.add_((self.b @ self.a) * self.scale)
        return self.base


def inject_lora(model, rank=8, alpha=16, targets=("q_proj", "v_proj")):
    """Freeze ``model`` and replace every named target projection with LoRA."""
    model.requires_grad_(False)
    replaced = []

    def visit(module, prefix=""):
        for name, child in list(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha))
                replaced.append(path)
            else:
                visit(child, path)

    visit(model)
    if not replaced:
        raise ValueError(f"No linear projections matched {tuple(targets)}")
    return replaced


def merge_lora(model):
    """Merge every adapter into its frozen linear layer for dependency-free export."""
    merged = []

    def visit(module, prefix=""):
        for name, child in list(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merge())
                merged.append(path)
            else:
                visit(child, path)

    visit(model)
    return merged


def load_frozen_e3_head(checkpoint, expected_sha256):
    """Load the fixed E3 control only after verifying its MLflow artifact hash."""
    checkpoint = Path(checkpoint)
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"E3 head SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    head = Head(saved.get("config", {}).get("hidden", 256))
    head.load_state_dict(saved["state_dict"])
    return head.requires_grad_(False).eval()


def trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
