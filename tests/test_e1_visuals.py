"""E1 presentation must use the actual preprocessing and retrieval protocol."""
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image, ImageOps

from pettrace.evaluation.metrics import retrieval
from pettrace.models.pipeline import Pipeline


def source(cell_id):
    path = Path(__file__).parents[1] / "notebooks/e1_clip_zero_shot.ipynb"
    notebook = json.loads(path.read_text())
    return "".join(next(c for c in notebook["cells"] if c["id"] == cell_id)["source"])


def test_visual_preprocessing_is_inverse_of_actual_tensor(tmp_path):
    # Rectangular gradient reveals a wrong crop or normalization, unlike a solid image.
    pixels = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    path = tmp_path / "sample.png"
    Image.fromarray(pixels).save(path)
    photos = [SimpleNamespace(path=str(path), animal=sp, species=sp) for sp in ("cat", "dog")]
    pipeline = object.__new__(Pipeline)
    pipeline.detector = None
    pipeline.pre = {"resize_shortest_edge": 8, "center_crop": [6, 6], "resample": "bicubic",
                    "rescale_factor": 1 / 255, "mean": [.48, .45, .40], "std": [.26, .25, .27]}
    displayed = []
    namespace = {"public_ds": SimpleNamespace(photos=photos, commit="pinned"),
                 "model": SimpleNamespace(pipeline=pipeline), "PUBLIC_REPORT": tmp_path,
                 "PILImage": Image, "ImageOps": ImageOps, "np": np, "Path": Path,
                 "json": json, "display": displayed.append}
    exec(source("e1-preprocess-visual"), namespace)
    assert len(displayed) == 2
    for fig in displayed:
        np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), pixels)
        np.testing.assert_allclose(fig.axes[1].images[0].get_array(), pixels[1:7, 3:9] / 255, atol=1e-7)
    manifest = json.loads((tmp_path / "preprocessing-examples.json").read_text())
    assert manifest["dataset_commit"] == "pinned"
    assert [r["query"] for r in manifest["examples"]] == [0, 1]
    assert (tmp_path / "preprocessing-cat.png").is_file()


def test_visual_retrieval_handles_ties_singletons_and_separate_species(tmp_path):
    import matplotlib.pyplot as plt

    photos, vectors = [], []
    for sp in ("cat", "dog"):
        # Ties, successful queries, failed queries and a gallery-only singleton.
        for animal, vector in zip(("a", "a", "b", "b", "solo"),
                                  ([1, 0], [1, 0], [1, 0], [-1, 0], [0, 1])):
            path = tmp_path / f"{len(photos)}.png"
            Image.new("RGB", (12, 8), (len(photos) * 20, 80, 120)).save(path)
            photos.append(SimpleNamespace(path=str(path), animal=sp + animal, species=sp))
            vectors.append(vector)
    embeddings = np.array(vectors, dtype=float)
    metrics = retrieval(embeddings, [p.animal for p in photos], [p.species for p in photos])
    # A reversed metrics list must not change the first-query selection rule.
    metrics["per_query"].reverse()
    ev = SimpleNamespace(photos=photos, embeddings=embeddings, result={"test": metrics})
    namespace = {"public_ev": ev, "public_ds": SimpleNamespace(commit="pinned"), "PUBLIC_REPORT": tmp_path,
                 "PILImage": Image, "ImageOps": ImageOps, "json": json, "plt": plt, "display": lambda fig: None}
    exec(source("e1-retrieval-visual"), namespace)
    manifest = json.loads((tmp_path / "retrieval-examples.json").read_text())
    examples = manifest["examples"]
    assert [e["query"] for e in examples] == [0, 2, 5, 7]
    for example in examples:
        q = example["query"]
        top = example["top5"]
        assert len(top) == len({p["animal"] for p in top}) == 3
        assert q not in [p["photo"] for p in top]
        assert all(photos[p["photo"]].species == photos[q].species for p in top)
        assert any(p["animal"].endswith("solo") for p in top)
        assert (tmp_path / example["figure"]).is_file()
    # No errors is a valid result; never fabricate a failure panel.
    ev.result["test"]["per_query"] = [r for r in metrics["per_query"] if r["rank"] == 1]
    exec(source("e1-retrieval-visual"), namespace)
    manifest = json.loads((tmp_path / "retrieval-examples.json").read_text())
    assert len(manifest["examples"]) == 2


def test_photo_upload_previews_fit_limit_preserve_originals_and_retry(tmp_path):
    import ast

    report = tmp_path / "run"
    public = report / "public_test"
    public.mkdir(parents=True)
    # Noise exercises the size bound instead of relying on compressible solid fixtures.
    pixels = np.random.default_rng(7).integers(0, 256, (1200, 3600, 3), dtype=np.uint8)
    panel = public / "retrieval-cat-error.png"
    Image.fromarray(pixels).save(panel)
    original = panel.read_bytes()
    assert len(original) > 1024 * 1024
    manifest = public / "retrieval-examples.json"
    manifest.write_text(json.dumps({"examples": [{"figure": panel.name, "rank": 3}]}))
    metrics = public / "summary.json"
    metrics.write_text('{"mAP": 0.5}')
    chart = public / "cmc.png"
    Image.new("RGB", (12, 12), "white").save(chart)
    chart_bytes = chart.read_bytes()
    tree = ast.parse(source("38c87fe3"))
    stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "experiment_design" for t in node.targets))
    code = compile(ast.Module(body=tree.body[:stop], type_ignores=[]), "publication-preview", "exec")
    namespace = {"LOG_TO_MLFLOW": True, "REPORT": report, "PUBLIC_REPORT": public,
                 "PILImage": Image, "json": json}
    exec(code, namespace)
    preview = public / "retrieval-cat-error.jpg"
    assert preview.stat().st_size <= 512 * 1024
    assert not panel.exists()
    assert (tmp_path / "run-original-figures" / panel.name).read_bytes() == original
    assert json.loads(manifest.read_text())["examples"] == [{"figure": preview.name, "rank": 3}]
    assert chart.read_bytes() == chart_bytes
    assert metrics.read_text() == '{"mAP": 0.5}'
    first = preview.read_bytes()
    exec(code, namespace)
    assert preview.read_bytes() == first
