"""Frozen E4 visual control for the progressive E5 experiment."""

import json
from pathlib import Path

from pettrace.evaluation.core import load_model

CONTROL_RUNS = {
    "public": "2302b302772b4349894e1d1b58d438ed",
    "private": "5bc216d03afc4a31a15c53fc3bfbc747",
}
CONTROL_MODEL_VERSION = "19"
CONTROL_MODEL_LABEL = "clip-vit-b32-crop-lora-e3-v4.100@v2"
CONTROL_MODEL_SHA256 = {
    "detector.onnx": "da8f9b9ebea5157cdcefa7bea9fd782249ebdf119f917c07d64e5b41942b9c9d",
    "encoder.onnx": "0d6bbd7b5156d3630e8b4a2d2369ea110d78f5a94eaf9481f1bbd23b1d6b66cb",
}
CONTROL_SEED = "46"
CONTROL_EPOCH = 9
CONTROL_VALIDATION_MAP = 0.8368148223597829


def load_frozen_e4(model_uri, output):
    """Load the exact E4 winner from a local or configured URI and verify its immutable identity."""
    model = load_model(str(model_uri))
    if (not model.golden_ok or model.model_version != CONTROL_MODEL_LABEL
            or model.sha256 != CONTROL_MODEL_SHA256):
        raise ValueError("E4 control failed identity or golden verification")

    receipt = {
        "experiment": "E4",
        "status": "frozen",
        "runs": CONTROL_RUNS,
        "model_name": "pettrace-embedder",
        "model_version": CONTROL_MODEL_VERSION,
        "model_uri": str(model_uri),
        "model_label": CONTROL_MODEL_LABEL,
        "model_sha256": CONTROL_MODEL_SHA256,
        "seed": int(CONTROL_SEED),
        "selected_epoch": CONTROL_EPOCH,
        "validation_macro_species_mAP": CONTROL_VALIDATION_MAP,
        "inference": {"representation": "single", "identity_aggregation": "max"},
        "alias_promoted": False,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "e4-control.json").write_text(json.dumps(receipt, indent=2))
    return model, receipt
