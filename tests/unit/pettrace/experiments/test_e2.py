"""E2 detector audit and paired intervention checks."""

from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytest

from pettrace.experiments import e2


def test_detector_audit_keeps_fallback_invalid_boxes_and_species_mismatches(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (100, 100)).save(path)
    detections = iter([None, (.8, 15, (20, 20, 40, 40)), (.9, 16, (30, 30, 30, 30))])
    detector = SimpleNamespace(margin=.1, box=lambda image: next(detections))
    photo = SimpleNamespace(path=path, animal="a", species="dog")
    dataset = SimpleNamespace(root=tmp_path, photos=[photo, photo, photo])

    findings = e2.audit_detector(dataset, detector)

    assert [row["status"] for row in findings] == ["no_detection", "crop", "invalid_box"]
    assert findings[1]["species_mismatch"]
    assert findings[1]["box"] == [18, 18, 42, 42]
    assert findings[1]["retained_area"] == 24 * 24 / 10000
    assert findings[0]["retained_area"] == findings[2]["retained_area"] == 1


def test_condition_comparison_requires_identical_queries_and_fallback_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr("pettrace.evaluation.figures.comparison_figure",
                        lambda *args: plt.figure())
    dataset = SimpleNamespace(commit="pinned")

    def evaluation(gallery=2):
        rows = [{"query": index, "photo": f"{index}.png", "animal": str(index),
                 "species": "cat", "gallery_animals": gallery, "rank": rank}
                for index, rank in enumerate((2, 1))]
        metrics = {group: {metric: .5 for metric in ("rank1", "rank2", "rank5", "mAP")}
                   for group in ("global", "cat", "dog")}
        metrics["per_query"] = rows
        return SimpleNamespace(dataset=dataset, result={"test": metrics}, embeddings=np.eye(2))

    baseline, localized = evaluation(), evaluation()
    result = e2._save_condition_comparison(baseline, localized, tmp_path, "public", [1])
    assert result["paired_queries"] == 2
    assert (tmp_path / "e1-e2-comparison.json").is_file()

    with pytest.raises(ValueError, match="coverage differs"):
        e2._save_condition_comparison(evaluation(3), localized, tmp_path, "public")
    localized.embeddings[1] = [1, 0]
    with pytest.raises(ValueError, match="fallback embeddings differ"):
        e2._save_condition_comparison(baseline, localized, tmp_path, "public", [1])
