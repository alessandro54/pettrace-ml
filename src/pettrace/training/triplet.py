"""E3: frozen E2 embeddings -> 512/256/512 MLP -> L2, batch-hard triplet loss.

Only the head is trained. Public validation identities select checkpoints; test is never
read by fit(). The exported encoder composes the exact E2 ONNX graph with the learned head.
"""

import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    seed: int = 42
    epochs: int = 30
    steps_per_epoch: int = 100
    identities_per_batch: int = 8
    images_per_identity: int = 4
    hidden: int = 256
    margin: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7


class Head(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(512, hidden), nn.ReLU(), nn.Linear(hidden, 512))

    def forward(self, embedding):
        return F.normalize(self.layers(embedding), p=2, dim=-1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def batch_hard_triplet(embedding, labels, margin=0.2):
    distance = torch.cdist(embedding, embedding, p=2)
    same = labels[:, None].eq(labels[None, :])
    positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    negative = ~same
    if not (positive.any(1).all() and negative.any(1).all()):
        raise ValueError("Every anchor needs another same-ID image and a different-ID negative")
    hardest_positive = distance.masked_fill(~positive, -torch.inf).max(1).values
    hardest_negative = distance.masked_fill(~negative, torch.inf).min(1).values
    losses = F.relu(hardest_positive - hardest_negative + margin)
    return losses.mean(), float((losses.detach() > 0).float().mean())


def pk_batch(labels, species, p, k, rng, selected_species=None):
    if p < 2 or k < 2:
        raise ValueError("Triplet batches require P >= 2 and K >= 2")
    groups = {}
    for i, (label, sp) in enumerate(zip(labels, species)):
        groups.setdefault(sp, {}).setdefault(label, []).append(i)
    eligible = {sp: {a: idx for a, idx in ids.items() if len(idx) >= 2} for sp, ids in groups.items()}
    if any(len(ids) < p for ids in eligible.values()):
        raise ValueError(f"Each training species needs >= {p} identities with >= 2 distinct photos")
    sp = selected_species or rng.choice(sorted(eligible))
    identities = rng.choice(sorted(eligible[sp]), p, replace=False)
    indices = []
    for animal in identities:
        choices = eligible[sp][animal]
        # Include distinct images before replacement, even for identities with fewer than K.
        chosen = list(rng.choice(choices, min(k, len(choices)), replace=False))
        if len(chosen) < k:
            chosen.extend(rng.choice(choices, k - len(chosen), replace=True))
        rng.shuffle(chosen)
        indices.extend(chosen)
    return np.asarray(indices, dtype=np.int64)


def project(head, features, device="cpu", batch=512):
    head.eval()
    with torch.no_grad():
        return np.concatenate([
            head(torch.as_tensor(features[i:i + batch], dtype=torch.float32, device=device)).cpu().numpy()
            for i in range(0, len(features), batch)
        ])


def validation_score(head, features, rows, device):
    from pettrace.evaluation.metrics import retrieval

    embeddings = project(head, features, device)
    result = retrieval(embeddings, [r["animal"] for r in rows], [r["species"] for r in rows])
    # Macro species mean: a larger cat dataset must not determine selection alone.
    scores = [result[sp]["mAP"] for sp in sorted({r["species"] for r in rows}) if sp in result]
    if len(scores) != len({r["species"] for r in rows}):
        raise ValueError("Validation requires >= 2 identities, each with >= 2 images, per species")
    return float(np.mean(scores)), result


def fast_eer(same, diff):
    """Compatibility entry point for training callers requiring both pair classes."""
    from pettrace.evaluation.metrics import eer

    if not len(same) or not len(diff):
        raise ValueError("EER calibration needs positive and negative pairs")
    return eer(same, diff)


def fit(features, rows, output, config=None, device=None, callback=None):
    config = config or Config()
    seed_all(config.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if features.shape != (len(rows), 512) or not np.isfinite(features).all():
        raise ValueError("Expected finite frozen E2 features shaped N x 512")
    if not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-3):
        raise ValueError("Frozen E2 features must be L2-normalized")
    if any(r["split"] not in {"train", "val"} for r in rows):
        raise ValueError("fit() accepts train/val only; test data is forbidden")
    train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    val = [i for i, r in enumerate(rows) if r["split"] == "val"]
    tr = [rows[i] for i in train]
    vr = [rows[i] for i in val]
    if not tr or not vr:
        raise ValueError("Separate train and val identities are required")
    if {r["animal"] for r in tr} & {r["animal"] for r in vr}:
        raise ValueError("Identity leakage between train and val")
    if {r["species"] for r in tr} != {r["species"] for r in vr}:
        raise ValueError("Training and validation must cover the same species")
    ids = {a: i for i, a in enumerate(sorted({r["animal"] for r in tr}))}
    labels = np.asarray([ids[r["animal"]] for r in tr])
    species = [r["species"] for r in tr]
    x = torch.as_tensor(features[train], dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    head = Head(config.hidden).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    history, best, best_epoch, stale = [], -1.0, 0, 0
    selected = sorted(set(species))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.epochs + 1):
        head.train()
        losses, active = [], []
        for step in range(config.steps_per_epoch):
            idx = pk_batch(labels, species, config.identities_per_batch, config.images_per_identity,
                           rng, selected[(epoch * config.steps_per_epoch + step) % len(selected)])
            index = torch.as_tensor(idx, dtype=torch.long, device=device)
            loss, fraction = batch_hard_triplet(head(x[index]), y[index], config.margin)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            active.append(fraction)
        score, _ = validation_score(head, features[val], vr, device)
        record = {"epoch": epoch, "train.loss": float(np.mean(losses)),
                  "train.active_triplets": float(np.mean(active)), "val.macro_species_mAP": score}
        history.append(record)
        if callback:
            callback(record)
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"state_dict": copy.deepcopy(head.cpu().state_dict()), "config": asdict(config),
                        "epoch": epoch, "validation_macro_mAP": score}, output / "head.pt")
            head.to(device)
        else:
            stale += 1
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(record, flush=True)
        if stale >= config.patience:
            break
    checkpoint = torch.load(output / "head.pt", map_location="cpu", weights_only=True)
    head = Head(config.hidden)
    head.load_state_dict(checkpoint["state_dict"])
    return head.eval(), {"best_epoch": best_epoch, "best_val_mAP": best, "history": history,
                         "config": asdict(config), "device": device}


def export_encoder(base_model_dir, head, output_dir, model_version):
    import shutil

    import onnx
    import onnxruntime as ort
    import yaml
    from onnx import compose

    from pettrace.models.golden import compare, fingerprint, golden_images
    from pettrace.models.pipeline import Pipeline

    base_dir, output = Path(base_model_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load((base_dir / "pipeline.yaml").read_text())
    step = next(s for s in spec["steps"] if s["op"] == "onnx")
    base = onnx.load(base_dir / step["file"])
    head.eval().cpu()
    sample = F.normalize(torch.randn(3, 512), dim=-1)
    torch.onnx.export(head, sample, str(output / "head.onnx"), input_names=["base_embedding"],
                      output_names=["embedding"], dynamic_axes={"base_embedding": {0: "batch"},
                      "embedding": {0: "batch"}}, opset_version=17, dynamo=False)
    added = onnx.load(output / "head.onnx")
    added.ir_version = base.ir_version
    merged = compose.merge_models(base, added, [(step["output"], "base_embedding")],
                                  prefix1="clip/", prefix2="head/")
    onnx.checker.check_model(merged)
    onnx.save(merged, output / "encoder.onnx")
    expected = head(sample).detach().numpy()
    actual = ort.InferenceSession(str(output / "head.onnx"), providers=["CPUExecutionProvider"]).run(
        None, {"base_embedding": sample.numpy()})[0]
    if not np.allclose(expected, actual, atol=1e-5):
        raise RuntimeError("Head PyTorch/ONNX parity failed")
    for s in spec["steps"]:
        if s["op"] == "detect_crop" and s.get("enabled"):
            shutil.copy2(base_dir / s["file"], output / s["file"])
    step.update(file="encoder.onnx", input=merged.graph.input[0].name, output=merged.graph.output[0].name)
    spec.update(experiment="E3", model_version=model_version, head={"architecture": f"512-{head.layers[0].out_features}-512 ReLU L2",
                "loss": "batch-hard triplet", "base": "frozen E2, no image augmentation"})
    (output / "pipeline.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    images = golden_images()
    base_embeddings = Pipeline(str(base_dir)).embed(images)
    expected_golden = project(head, base_embeddings)
    actual_golden = Pipeline(str(output)).embed(images)
    ok, minimum_cosine, maximum_error = compare(expected_golden, actual_golden)
    if not ok:
        raise RuntimeError(f"Composed encoder parity failed: {minimum_cosine}, {maximum_error}")
    np.save(output / "golden.npy", actual_golden)
    (output / "golden.json").write_text(json.dumps({"fingerprint": fingerprint(actual_golden),
        "parity_min_cosine": minimum_cosine, "parity_max_abs": maximum_error}, indent=2))
    (output / "thresholds.json").write_text(json.dumps({"calibrated": False,
        "note": "Calibrate on public val; never on local test."}))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.onnx")}
