"""Training correctness checks; synthetic fixtures are not experimental results."""
# ruff: noqa: E402 -- skip the optional training suite before importing its dependencies.

import json

import numpy as np
import pytest
import yaml
torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from torch import nn
from torch.nn import functional as F

from pettrace.training.triplet import Config, Head, batch_hard_triplet, export_encoder, fast_eer, fit, pk_batch
from pettrace.evaluation.metrics import eer
from pettrace.data.curation import audit_and_split, discover


def test_pk_batch_has_same_species_unique_identities_and_distinct_positives():
    labels = [0, 0, 1, 1, 2, 2, 3, 3]
    species = ["cat"] * 4 + ["dog"] * 4
    batch = pk_batch(labels, species, 2, 4, np.random.default_rng(8), "dog")
    assert len(batch) == 8
    assert {species[i] for i in batch} == {"dog"}
    assert {labels[i] for i in batch} == {2, 3}
    assert len({i for i in batch if labels[i] == 2}) == 2
    assert len({i for i in batch if labels[i] == 3}) == 2


def test_triplet_has_no_self_positive_and_finite_gradients():
    x = torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.1, .9]], requires_grad=True)
    loss, _ = batch_hard_triplet(x, torch.tensor([0, 0, 1, 1]))
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError, match="Every anchor"):
        batch_hard_triplet(x, torch.arange(4))


def fixture_features():
    rng = np.random.default_rng(7)
    rows, features = [], []
    for split in ["train", "val"]:
        for species in ["cat", "dog"]:
            for identity in range(3):
                centre = rng.normal(size=512)
                for photo in range(3):
                    x = centre + rng.normal(scale=.4, size=512)
                    features.append(x / np.linalg.norm(x))
                    rows.append({"animal": f"{split}:{species}:{identity}", "species": species,
                                 "split": split, "path": f"{split}/{species}/{identity}/{photo}.jpg"})
    return np.asarray(features, dtype=np.float32), rows


def test_fit_checkpoints_trained_weights_and_rejects_test_and_identity_leakage(tmp_path):
    features, rows = fixture_features()
    config = Config(epochs=2, steps_per_epoch=2, identities_per_batch=2, images_per_identity=2)
    trained, result = fit(features, rows, tmp_path, config, device="cpu")
    assert result["best_epoch"] >= 1
    assert (tmp_path / "head.pt").exists()
    assert len(json.loads((tmp_path / "history.json").read_text())) == 2
    assert torch.allclose(trained(torch.from_numpy(features)).norm(dim=1), torch.ones(len(rows)), atol=1e-5)
    invalid = [dict(r) for r in rows]
    invalid[-1]["split"] = "test"
    with pytest.raises(ValueError, match="test data is forbidden"):
        fit(features, invalid, tmp_path / "forbidden", config)
    invalid = [dict(r) for r in rows]
    invalid[-1]["animal"] = rows[0]["animal"]
    with pytest.raises(ValueError, match="Identity leakage"):
        fit(features, invalid, tmp_path / "leak", config)


def test_fast_eer_matches_existing_threshold_including_ties():
    rng = np.random.default_rng(3)
    same = np.round(rng.random(100), 1)
    diff = np.round(rng.random(100), 1)
    assert fast_eer(same, diff) == eer(same, diff)


def test_public_import_deduplicates_without_mixing_identities(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(4)
    cats = tmp_path / "cats"
    for identity in range(12):
        directory = cats / "cat_individuals_dataset" / str(identity)
        directory.mkdir(parents=True)
        for photo in range(3):
            Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(directory / f"{photo}.jpg")
    original = cats / "cat_individuals_dataset/0/0.jpg"
    (original.parent / "duplicate.jpg").write_bytes(original.read_bytes())
    rows, skipped = discover(cats, "cat-individuals", "cat")
    assert not skipped
    result, report = audit_and_split(rows, tmp_path / "manifest")
    assert len(result) == 36
    assert len(report["removed_duplicates"]) == 1
    splits = {}
    for r in result:
        splits.setdefault(r["animal"], set()).add(r["split"])
    assert all(len(s) == 1 for s in splits.values())
    conflict = original.parent.parent / "1/conflict.jpg"
    conflict.write_bytes(original.read_bytes())
    rows, _ = discover(cats, "cat-individuals", "cat")
    with pytest.raises(ValueError, match="conflicting identities"):
        audit_and_split(rows, tmp_path / "conflict")


def test_composed_encoder_export_is_compatible_with_unchanged_serving_pipeline(tmp_path):
    import onnx

    class TinyFrozenEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 512)

        def forward(self, pixels):
            return F.normalize(self.projection(pixels.mean(dim=(2, 3))), dim=-1)

    base = tmp_path / "base"
    base.mkdir()
    encoder = TinyFrozenEncoder().eval()
    torch.onnx.export(encoder, torch.randn(2, 3, 8, 8), str(base / "encoder.onnx"),
                      input_names=["pixel_values"], output_names=["embedding"],
                      dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17)
    spec = {"steps": [
        {"op": "clip_preprocess", "resize_shortest_edge": 8, "center_crop": [8, 8],
         "resample": "bicubic", "rescale_factor": 1 / 255, "mean": [0, 0, 0], "std": [1, 1, 1]},
        {"op": "onnx", "file": "encoder.onnx", "input": "pixel_values", "output": "embedding"}]}
    (base / "pipeline.yaml").write_text(yaml.safe_dump(spec))
    hashes = export_encoder(base, Head(), tmp_path / "composed", "e3-test")
    onnx.checker.check_model(str(tmp_path / "composed/encoder.onnx"))
    assert "encoder.onnx" in hashes
    assert (tmp_path / "composed/golden.npy").exists()


def test_nested_archives_extract_once_and_reject_path_traversal(tmp_path):
    import io
    import zipfile

    from pettrace.data.acquisition import extract

    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("photos/a.jpg", b"test fixture")
    wrapper = io.BytesIO()
    with zipfile.ZipFile(wrapper, "w") as archive:
        archive.writestr("MPDD/inner.zip", nested.getvalue())
    extract(io.BytesIO(wrapper.getvalue()), tmp_path / "extracted")
    assert (tmp_path / "extracted/MPDD/photos/a.jpg").read_bytes() == b"test fixture"
    unsafe = io.BytesIO()
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape.txt", b"invalid")
    with pytest.raises(ValueError, match="Unsafe archive"):
        extract(io.BytesIO(unsafe.getvalue()), tmp_path / "unsafe")


def test_mpdd_never_uses_original_test_for_checkpoint_selection(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(13)
    root = tmp_path / "mpdd"
    for identity in range(14):
        category = "gallery" if identity >= 10 else "train"
        directory = root / "MPDD/pytorch" / category
        directory.mkdir(parents=True, exist_ok=True)
        for photo in range(2):
            Image.fromarray(rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)).save(
                directory / f"{identity}_c1s1_{photo}.jpg")
    rows, _ = discover(root, "mpdd", "dog")
    curated, _ = audit_and_split(rows, tmp_path / "curated")
    assert all(r["split"] == "test" for r in curated if int(r["animal"].split(":")[-1]) >= 10)
    assert {r["split"] for r in curated if int(r["animal"].split(":")[-1]) < 10} == {"train", "val"}
