"""DINOv2 adaptation, keeping frozen/metric/LoRA interventions explicit.

Methods: Oquab et al. (2024), Hu et al. (2022), Hermans et al. (2017).
Identity projection, component controls and selection are study design choices.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pettrace.training import lora
from pettrace.training.dino_metric import Metric
from pettrace.training.triplet import seed_all

REPOSITORY = "facebook/dinov2-large"
REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
MODES = ("frozen", "projection", "lora", "combined", "mlp", "mlp_lora")


@dataclass
class Config(lora.Config):
    learning_rate: float = 1e-5


class Encoder(nn.Module):
    def __init__(self, backbone, mode, config, head=None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown DINO intervention: {mode}")
        self.backbone = backbone.requires_grad_(False)
        self.mode = mode
        self.head = Metric(backbone.config.hidden_size) if mode in ("projection", "combined") else nn.Identity()
        if mode in ("mlp", "mlp_lora"):
            if head is None or head.layers[0].in_features != backbone.config.hidden_size:
                raise ValueError("A dimension-matched, trained E3 MLP is required")
            self.head = head.requires_grad_(False).eval()
        self.targets = []
        if mode in ("lora", "combined", "mlp_lora"):
            self.targets = lora.inject_lora(backbone, config.rank, config.alpha, targets=("query", "value"))
            if len(self.targets) != 2 * backbone.config.num_hidden_layers:
                raise ValueError("DINO attention target count changed")

    def forward(self, pixels):
        embedding = F.normalize(self.backbone(pixel_values=pixels).last_hidden_state[:, 0], dim=-1)
        return F.normalize(self.head(embedding), dim=-1)


def build(mode, device="cpu", config=None, head=None):
    from transformers import AutoImageProcessor, AutoModel

    config = config or Config()
    seed_all(config.seed)
    backbone = AutoModel.from_pretrained(REPOSITORY, revision=REVISION, trust_remote_code=False)
    processor = AutoImageProcessor.from_pretrained(REPOSITORY, revision=REVISION, use_fast=False)
    return Encoder(backbone, mode, config, head).to(device).eval(), processor


class Preprocessor:
    def __init__(self, processor, detector, prepared_crop=None):
        self.processor, self.detector, self.prepared_crop = processor, detector, prepared_crop

    def pixels(self, rows):
        from PIL import Image, ImageOps

        images = []
        for row in rows:
            if self.prepared_crop is not None:
                images.append(self.prepared_crop.image(row))
            else:
                with Image.open(row["path"]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                images.append(self.detector.crop(image) if self.detector is not None else image)
        return self.processor(images=images, return_tensors="np")["pixel_values"].astype(np.float32)


class Pipeline:
    """Same embed interface as the shared evaluator, using the official processor."""

    def __init__(self, encoder, processor, detector, batch=8):
        if batch < 1:
            raise ValueError("Positive inference batch required")
        self.encoder, self.processor, self.detector, self.batch = encoder, processor, detector, batch

    def embed(self, images):
        device = next(self.encoder.parameters()).device
        self.encoder.eval()
        output = []
        for offset in range(0, len(images), self.batch):
            crops = [self.detector.crop(x.convert("RGB")) if self.detector else x.convert("RGB")
                     for x in images[offset:offset + self.batch]]
            pixels = self.processor(images=crops, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                output.append(self.encoder(pixels).cpu().numpy())
        return np.concatenate(output) if output else np.empty((0, self.encoder.backbone.config.hidden_size))


def fit(encoder, rows, preprocessor, output, config=None, device="cpu"):
    """Reuse the audited trainer and include the initial frozen condition in selection."""
    config = config or Config()
    if encoder.mode in ("frozen", "mlp"):
        raise ValueError("The frozen control cannot be trained")
    if any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("Only train/val allowed")
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if not train or not val or {r['animal'] for r in train} & {r['animal'] for r in val}:
        raise ValueError("Separate training/validation identities required")
    if len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError("Duplicate image bytes")
    if any({r['species'] for r in subset} != {'cat', 'dog'} for subset in (train, val)):
        raise ValueError("Both species required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'adapter.pt').exists():
        raise FileExistsError("Preserve the existing run; load its result or choose a new directory")
    initial_state = lora._adapter_state(encoder)
    initial_score, initial_metrics = lora._validation_score(encoder, val, preprocessor.pixels(val), device)
    initial = dict(state_dict=initial_state, config=asdict(config), epoch=0, validation_macro_mAP=initial_score)
    torch.save(initial, output / 'initial.pt')
    encoder, result = lora.fit(encoder, rows, preprocessor, output, config, device)
    if initial_score >= result['best_val_mAP']:
        torch.save(initial, output / 'adapter.pt')
        lora.load_adapter(encoder, output / 'adapter.pt')
        result.update(best_epoch=0, best_val_mAP=initial_score)
    result.update(mode=encoder.mode, initial_validation=initial_metrics,
                  trainable_parameters=lora.trainable_parameters(encoder), targets=encoder.targets,
                  model_repo=REPOSITORY, model_revision=REVISION)
    (output / 'training-result.json').write_text(json.dumps(result, indent=2))
    return encoder.eval(), result
