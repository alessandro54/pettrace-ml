"""E1 presentation must use the actual preprocessing and retrieval protocol."""
import json
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from PIL import Image

from pettrace.evaluation.figures import retrieval_examples
from pettrace.evaluation.metrics import retrieval
from pettrace.experiments.zero_shot import _prepare_photo_panels


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
    figures, _ = retrieval_examples(ev, tmp_path, "pinned")
    for figure in figures:
        plt.close(figure)
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
    figures, _ = retrieval_examples(ev, tmp_path, "pinned")
    for figure in figures:
        plt.close(figure)
    manifest = json.loads((tmp_path / "retrieval-examples.json").read_text())
    assert len(manifest["examples"]) == 2


@pytest.mark.parametrize("experiment,panel_name,manifest_name", [("e1", "retrieval-cat-error", "retrieval-examples"), ("e2", "retrieval-cat-error", "retrieval-examples"), ("e2", "preprocessing-cat-crop", "preprocessing-examples")])
def test_photo_upload_previews_fit_limit_preserve_originals_and_retry(tmp_path, experiment, panel_name, manifest_name):
    report = tmp_path / "run"
    public = report / "public_test"
    public.mkdir(parents=True)
    # Noise exercises the size bound instead of relying on compressible solid fixtures.
    pixels = np.random.default_rng(7).integers(0, 256, (1200, 3600, 3), dtype=np.uint8)
    panel = public / f"{panel_name}.png"
    Image.fromarray(pixels).save(panel)
    original = panel.read_bytes()
    assert len(original) > 1024 * 1024
    manifest = public / f"{manifest_name}.json"
    manifest.write_text(json.dumps({"examples": [{"figure": panel.name, "rank": 3}]}))
    metrics = public / "summary.json"
    metrics.write_text('{"mAP": 0.5}')
    chart = public / "cmc.png"
    Image.new("RGB", (12, 12), "white").save(chart)
    chart_bytes = chart.read_bytes()
    _prepare_photo_panels(report, public)
    preview = public / f"{panel_name}.jpg"
    assert preview.stat().st_size <= 512 * 1024
    assert not panel.exists()
    assert (tmp_path / "run-original-figures" / panel.name).read_bytes() == original
    assert json.loads(manifest.read_text())["examples"] == [{"figure": preview.name, "rank": 3}]
    assert chart.read_bytes() == chart_bytes
    assert metrics.read_text() == '{"mAP": 0.5}'
    first = preview.read_bytes()
    _prepare_photo_panels(report, public)
    assert preview.read_bytes() == first
