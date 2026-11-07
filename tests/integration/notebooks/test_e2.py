"""E2 displays the actual detector intervention and preserves paired comparisons."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import pytest

from pettrace.models.pipeline import Pipeline
from tests import ROOT


def source(cell_id):
    notebook = json.loads((ROOT / "notebooks/e2_yolo_crop_zero_shot.ipynb").read_text())
    return "".join(next(c for c in notebook["cells"] if c["id"] == cell_id)["source"])


def test_detector_audit_keeps_failures_and_species_mismatches(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (100, 100)).save(path)
    detections = iter([None, (.8, 15, (20, 20, 40, 40)), (.9, 16, (30, 30, 30, 30))])
    detector = SimpleNamespace(margin=.1, box=lambda image: next(detections))
    dataset = SimpleNamespace(root=tmp_path, photos=[SimpleNamespace(path=path, animal="a", species="dog")] * 3)
    tree = ast.parse(source("e2-detector-audit"))
    namespace = {"model": SimpleNamespace(pipeline=SimpleNamespace(detector=detector)),
                 "PILImage": Image, "ImageOps": ImageOps, "Path": Path}
    exec(compile(ast.Module(body=[tree.body[0]], type_ignores=[]), "detector-audit", "exec"), namespace)
    findings = namespace["audit_detector"](dataset)
    assert [row["status"] for row in findings] == ["no_detection", "crop", "invalid_box"]
    assert findings[1]["species_mismatch"]
    assert findings[1]["box"] == [18, 18, 42, 42]
    assert findings[1]["retained_area"] == 24 * 24 / 10000
    assert findings[0]["retained_area"] == findings[2]["retained_area"] == 1


def test_visual_shows_actual_crop_and_clip_tensor_without_double_cropping(tmp_path):
    pixels = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    path = tmp_path / "photo.png"
    Image.fromarray(pixels).save(path)
    pipeline = object.__new__(Pipeline)

    def crop(image):
        assert image.size == (16, 12), "Detector must receive the original, not an already cropped image"
        return image.crop((2, 2, 14, 10))

    pipeline.detector = SimpleNamespace(crop=crop)
    pipeline.pre = {"resize_shortest_edge": 8, "center_crop": [6, 6], "resample": "bicubic",
                    "rescale_factor": 1 / 255, "mean": [.48, .45, .40], "std": [.26, .25, .27]}
    displayed = []
    namespace = {"public_ds": SimpleNamespace(commit="pinned", photos=[SimpleNamespace(path=path, animal="a")]),
                 "public_detector_findings": [{"query": 0, "species": "cat", "status": "crop", "box": [2, 2, 14, 10]}],
                 "model": SimpleNamespace(pipeline=pipeline), "PUBLIC_REPORT": tmp_path,
                 "PILImage": Image, "ImageOps": ImageOps, "np": np, "json": json, "display": displayed.append}
    exec(source("e2-preprocess-visual"), namespace)
    assert len(displayed) == 1  # No fabricated fallback example.
    axes = displayed[0].axes
    np.testing.assert_array_equal(axes[0].images[0].get_array(), pixels)
    np.testing.assert_array_equal(axes[1].images[0].get_array(), pixels[2:10, 2:14])
    np.testing.assert_allclose(axes[2].images[0].get_array(), pixels[3:9, 5:11] / 255, atol=1e-7)
    manifest = json.loads((tmp_path / "preprocessing-examples.json").read_text())
    assert manifest["dataset_commit"] == "pinned"
    assert (tmp_path / manifest["examples"][0]["figure"]).is_file()


def test_public_comparison_pairs_queries_and_checks_fallback(tmp_path):
    from copy import deepcopy

    public = tmp_path / "public_test"
    public.mkdir()
    ds = SimpleNamespace(commit="public")

    def evaluation(ranks, commit):
        rows = [{"query": i, "photo": f"{i}.png", "animal": str(i), "species": "cat",
                 "gallery_animals": 2, "rank": rank, "ap": 1/rank} for i, rank in enumerate(ranks)]
        metrics = {group: {"rank1": .5, "rank5": 1., "mAP": .75} for group in ("global", "cat", "dog")}
        metrics["per_query"] = rows
        return SimpleNamespace(protocol="zero-shot", dataset=SimpleNamespace(commit=commit),
                               photos=[SimpleNamespace(path="0.png"), SimpleNamespace(path="1.png")],
                               embeddings=np.eye(2), result={"test": metrics, "val": {"eer": .25}})

    first, second = evaluation([2, 1], "public"), evaluation([1, 2], "public")
    namespace = {"public_ds": ds, "public_ev": second, "baseline_model": SimpleNamespace(pipeline=object()),
                 "evaluate": lambda dataset, pipeline, **kwargs: first,
                 "query_signature": lambda row: tuple(row[k] for k in ("query", "photo", "animal", "species", "gallery_animals")),
                 "public_detector_findings": [{"query": 1, "status": "no_detection"}],
                 "save": lambda *args: None, "save_provenance": lambda *args: None,
                 "baseline_ev": evaluation([1, 1], "private"), "ev": evaluation([1, 1], "private"),
                 "REPORT": tmp_path, "PUBLIC_REPORT": public, "PROTOCOL": "zero-shot",
                 "np": np, "json": json, "plt": plt, "Markdown": str, "display": lambda *args: None}
    exec(source("e2-public-baseline"), namespace)
    result = json.loads((public / "e1-e2-comparison.json").read_text())
    assert result["corrected"] == result["lost"] == 1
    assert result["dataset_commit"] == "public"
    assert json.loads((tmp_path / "e1-e2-comparison.json").read_text())["dataset_commit"] == "private"
    changed = deepcopy(first)
    changed.result["test"]["per_query"][0]["gallery_animals"] = 3
    namespace["evaluate"] = lambda *args, **kwargs: changed
    with pytest.raises(AssertionError, match="coverage differs"):
        exec(source("e2-public-baseline"), namespace)
    namespace["evaluate"] = lambda *args, **kwargs: first
    first.embeddings[1] = [1, 0]
    with pytest.raises(AssertionError):
        exec(source("e2-public-baseline"), namespace)
