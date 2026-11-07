"""LoRA adapters for E4 encoder adaptation (Hu et al., 2021)."""

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pettrace.training.triplet import Head, batch_hard_triplet, pk_batch, seed_all
from pettrace.ui import EpochProgress


@dataclass
class Config:
    seed: int = 42
    epochs: int = 30
    steps_per_epoch: int = 100
    identities_per_batch: int = 8
    images_per_identity: int = 4
    margin: float = 0.2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 7
    rank: int = 8
    alpha: int = 16


class Encoder(nn.Module):
    """CLIP vision encoder, projection and the frozen E3 head."""

    def __init__(self, vision_model, visual_projection, head):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.head = head

    def forward(self, pixel_values):
        pooled = self.vision_model(pixel_values=pixel_values).pooler_output
        return self.head(F.normalize(self.visual_projection(pooled), p=2, dim=-1))


class PreparedPreprocessor:
    """Reuse prepared YOLO boxes, then apply the exact serving CLIP preprocessing."""

    def __init__(self, pipeline, screening):
        self.pipeline = pipeline
        self.clip = copy.copy(pipeline)
        self.clip.detector = None
        if pipeline.detector is None:
            raise ValueError("E4 requires the E2 detector pipeline")
        detector_sha256 = getattr(screening, "detector_sha256", None)
        if detector_sha256 is None:
            detector_sha256 = screening.get("config", {}).get("detector_sha256")
        if detector_sha256 != _detector_hash(pipeline):
            raise ValueError("Prepared YOLO boxes use a different detector")
        self.boxes = {} if hasattr(screening, "detector_sha256") else {
            record["image_sha256"]: record["boxes"][0]["box"]
            for record in screening.get("records", [])
            if record.get("reason") == "single_expected_animal" and len(record.get("boxes", [])) == 1
        }

    def image(self, row):
        from PIL import Image, ImageOps

        with Image.open(row["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        box = row.get("box") or self.boxes.get(row["sha256"])
        if box is None:
            if row["split"] == "train":
                raise ValueError(f"Missing prepared YOLO box for training photo {row['sha256']}")
            return self.pipeline.detector.crop(image)
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        margin = self.pipeline.detector.margin
        crop = (max(0, x1 - margin * width), max(0, y1 - margin * height),
                min(image.width, x2 + margin * width), min(image.height, y2 + margin * height))
        return image.crop(crop) if crop[2] - crop[0] >= 2 and crop[3] - crop[1] >= 2 else image

    def pixels(self, rows):
        return np.stack([self.clip.preprocess(self.image(row)) for row in rows]).astype(np.float32)

    def validate(self, rows, atol=1e-6):
        checked = []
        for row in rows:
            from PIL import Image, ImageOps

            with Image.open(row["path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            expected = self.pipeline.preprocess(image)
            actual = self.clip.preprocess(self.image(row))
            if not np.allclose(actual, expected, atol=atol):
                raise ValueError(f"Cached YOLO crop differs from E2 for {row['sha256']}")
            checked.append(row["sha256"])
        return checked


def _detector_hash(pipeline):
    next(step for step in pipeline.spec["steps"] if step["op"] == "detect_crop")
    return hashlib.sha256((Path(pipeline.detector.session._model_path)).read_bytes()).hexdigest()


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank update."""

    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1:
            raise ValueError("LoRA requires a linear layer and rank >= 1")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.a = nn.Parameter(base.weight.new_empty((rank, base.in_features)))
        self.b = nn.Parameter(base.weight.new_zeros((base.out_features, rank)))
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


def build_encoder(head, revision, device="cuda", config=None):
    """Load pinned CLIP, freeze it and add only the planned E4 adapters."""
    from transformers import CLIPModel

    config = config or Config()
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", revision=revision)
    encoder = Encoder(clip.vision_model, clip.visual_projection, head).to(device).eval()
    replaced = inject_lora(encoder.vision_model, config.rank, config.alpha)
    if trainable_parameters(encoder) <= 0:
        raise RuntimeError("No trainable LoRA parameters")
    return encoder, replaced


def control_equivalence(encoder, pixels, e2_embeddings, atol=1e-3):
    """Prove that zero-initialized adapters reproduce E3 on identical E2 pixels."""
    device = next(encoder.parameters()).device
    encoder.eval()
    with torch.no_grad():
        actual = encoder(torch.as_tensor(pixels, dtype=torch.float32, device=device)).cpu().numpy()
        expected = encoder.head(torch.as_tensor(e2_embeddings, dtype=torch.float32, device=device)).cpu().numpy()
    cosine = np.sum(actual * expected, axis=1) / (
        np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1))
    maximum_error = float(np.abs(actual - expected).max())
    minimum_cosine = float(cosine.min())
    if minimum_cosine < 0.9999 or maximum_error > atol:
        raise RuntimeError(f"E3 control parity failed: cosine={minimum_cosine}, max_abs={maximum_error}")
    return {"minimum_cosine": minimum_cosine, "maximum_absolute_error": maximum_error,
            "photos": len(pixels)}


def _embed(encoder, pixels, device, batch=64):
    encoder.eval()
    values = []
    with torch.no_grad():
        for offset in range(0, len(pixels), batch):
            values.append(encoder(torch.as_tensor(pixels[offset:offset + batch], device=device)).cpu().numpy())
    return np.concatenate(values)


def _validation_score(encoder, rows, pixels, device):
    from pettrace.evaluation.metrics import retrieval

    result = retrieval(_embed(encoder, pixels, device),
                       [row["animal"] for row in rows], [row["species"] for row in rows])
    species = sorted({row["species"] for row in rows})
    if any(name not in result for name in species):
        raise ValueError("Validation requires valid retrieval galleries for every species")
    return float(np.mean([result[name]["mAP"] for name in species])), result


def _adapter_state(encoder):
    return {name: parameter.detach().cpu().clone() for name, parameter in encoder.named_parameters()
            if parameter.requires_grad}


def load_adapter(encoder, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    parameters = dict(encoder.named_parameters())
    if set(saved["state_dict"]) != {name for name, value in parameters.items() if value.requires_grad}:
        raise ValueError("Adapter checkpoint does not match this E4 encoder")
    with torch.no_grad():
        for name, value in saved["state_dict"].items():
            parameters[name].copy_(value.to(parameters[name].device))
    return saved


def fit(encoder, rows, preprocessor, output, config=None, device="cuda", callback=None):
    """Train only LoRA on prepared photos; public validation selects the checkpoint."""
    config = config or Config()
    seed_all(config.seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E4 requires a CUDA runtime")
    if any(row["split"] not in {"train", "val"} for row in rows):
        raise ValueError("fit() accepts train/val only; test data is forbidden")
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    if not train or not val or {row["animal"] for row in train} & {row["animal"] for row in val}:
        raise ValueError("Separate train and validation identities are required")
    if {row["species"] for row in train} != {row["species"] for row in val}:
        raise ValueError("Training and validation must cover the same species")
    ids = {animal: index for index, animal in enumerate(sorted({row["animal"] for row in train}))}
    labels = np.asarray([ids[row["animal"]] for row in train])
    species = [row["species"] for row in train]
    target = torch.as_tensor(labels, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in encoder.parameters() if parameter.requires_grad],
        lr=config.learning_rate, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    selected = sorted(set(species))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history, best, best_epoch, stale = [], -1.0, 0, 0
    optimizer_steps, examples_seen, unique_seen = 0, 0, set()
    encoder.to(device)
    val_pixels = preprocessor.pixels(val)  # Fixed validation preprocessing is reused across epochs.
    progress = None if callback else EpochProgress(config.epochs, config.seed)
    try:
        for epoch in range(1, config.epochs + 1):
            encoder.train()
            losses, active = [], []
            for step in range(config.steps_per_epoch):
                indices = pk_batch(labels, species, config.identities_per_batch, config.images_per_identity,
                                   rng, selected[((epoch - 1) * config.steps_per_epoch + step) % len(selected)])
                pixels = preprocessor.pixels([train[index] for index in indices])
                index = torch.as_tensor(indices, dtype=torch.long, device=device)
                embedding = encoder(torch.as_tensor(pixels, dtype=torch.float32, device=device))
                loss, fraction = batch_hard_triplet(embedding, target[index], config.margin)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite E4 training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in encoder.parameters() if p.requires_grad], 5.0)
                optimizer.step()
                optimizer_steps += 1
                examples_seen += len(indices)
                unique_seen.update(int(index) for index in indices)
                losses.append(float(loss.detach()))
                active.append(fraction)
            score, _ = _validation_score(encoder, val, val_pixels, device)
            record = {"epoch": epoch, "train.loss": float(np.mean(losses)),
                      "train.active_triplets": float(np.mean(active)), "val.macro_species_mAP": score,
                      "training.optimizer_steps": optimizer_steps, "training.examples_seen": examples_seen,
                      "training.unique_photos_seen": len(unique_seen)}
            history.append(record)
            if score > best:
                best, best_epoch, stale = score, epoch, 0
                torch.save({"state_dict": _adapter_state(encoder), "config": asdict(config), "epoch": epoch,
                            "validation_macro_mAP": score}, output / "adapter.pt")
            else:
                stale += 1
            (output / "history.json").write_text(json.dumps(history, indent=2))
            (callback or progress.update)(record)
            if stale >= config.patience:
                break
    finally:
        if progress:
            progress.close()
    load_adapter(encoder, output / "adapter.pt")
    return encoder.eval(), {"best_epoch": best_epoch, "best_val_mAP": best, "history": history,
                            "config": asdict(config), "device": device}


def export_encoder(base_model_dir, encoder, output_dir, model_version, verification_pixels):
    """Merge adapters and export the complete E4 image encoder for serving."""
    import shutil

    import onnxruntime as ort
    import yaml

    from pettrace.models.golden import compare, fingerprint, golden_images
    from pettrace.models.pipeline import Pipeline

    base_dir, output = Path(base_model_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merge_lora(encoder)
    encoder.eval().cpu()
    sample = torch.as_tensor(verification_pixels, dtype=torch.float32)
    expected = encoder(sample).detach().numpy()
    torch.onnx.export(encoder, sample, str(output / "encoder.onnx"), input_names=["pixel_values"],
                      output_names=["embedding"], dynamic_axes={"pixel_values": {0: "batch"},
                      "embedding": {0: "batch"}}, opset_version=17, dynamo=False)
    actual = ort.InferenceSession(str(output / "encoder.onnx"), providers=["CPUExecutionProvider"]).run(
        None, {"pixel_values": verification_pixels.astype(np.float32)})[0]
    if not np.allclose(expected, actual, atol=1e-3):
        raise RuntimeError("E4 PyTorch/ONNX parity failed")
    spec = yaml.safe_load((base_dir / "pipeline.yaml").read_text())
    onnx_step = next(step for step in spec["steps"] if step["op"] == "onnx")
    onnx_step.update(file="encoder.onnx", input="pixel_values", output="embedding")
    for step in spec["steps"]:
        if step["op"] == "detect_crop" and step.get("enabled"):
            shutil.copy2(base_dir / step["file"], output / step["file"])
    spec.update(experiment="E4", model_version=model_version,
                adaptation={"method": "LoRA", "targets": ["q_proj", "v_proj"],
                            "base": "frozen E2 CLIP and frozen E3-100 head"})
    (output / "pipeline.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    images = golden_images()
    base_pipeline = Pipeline(str(base_dir))
    golden_pixels = np.stack([base_pipeline.preprocess(image) for image in images]).astype(np.float32)
    expected_golden = encoder(torch.as_tensor(golden_pixels)).detach().numpy()
    actual_golden = Pipeline(str(output)).embed(images)
    ok, minimum_cosine, maximum_error = compare(expected_golden, actual_golden)
    if not ok:
        raise RuntimeError(f"E4 composed encoder parity failed: {minimum_cosine}, {maximum_error}")
    np.save(output / "golden.npy", actual_golden)
    (output / "golden.json").write_text(json.dumps({"fingerprint": fingerprint(actual_golden),
        "parity_min_cosine": minimum_cosine, "parity_max_abs": maximum_error}, indent=2))
    (output / "thresholds.json").write_text(json.dumps({"calibrated": False,
        "note": "Calibrate on public val; never on either test benchmark."}))
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.glob("*.onnx")}
