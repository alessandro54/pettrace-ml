from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pettrace.e3_detection import animal_boxes, screen_animals
from pettrace.pipeline import Detector
from pettrace.public_data import sha256


def prediction(items):
    raw = np.zeros((1, 84, len(items)), dtype=np.float32)
    for index, (cls, score, box) in enumerate(items):
        raw[0, :4, index] = box
        raw[0, 4 + cls, index] = score
    return raw


class Session:
    def __init__(self, outputs):
        self.outputs, self.calls = iter(outputs), 0

    def run(self, _, inputs):
        self.calls += 1
        return [next(self.outputs)]


def detector(outputs):
    value = Detector.__new__(Detector)
    value.session, value.input_name = Session(outputs), "images"
    value.size, value.classes = 640, [15, 16]
    value.conf, value.margin = 0.25, 0.10
    return value


def test_nms_counts_distinct_animals_not_overlapping_anchors_or_cross_class_duplicates():
    image = Image.new("RGB", (640, 320))
    raw = prediction([(16, .95, (320, 320, 200, 160)), (16, .9, (322, 320, 200, 160)),
                      (15, .8, (320, 320, 200, 160)), (0, .99, (600, 200, 50, 80))])
    boxes = animal_boxes(detector([raw]), image)
    assert len(boxes) == 1 and boxes[0]["class_id"] == 16
    assert boxes[0]["box"] == pytest.approx([220, 80, 420, 240])
    pair = prediction([(16, .9, (100, 320, 100, 100)), (16, .8, (500, 320, 100, 100))])
    assert len(animal_boxes(detector([pair]), image)) == 2
    assert len(animal_boxes(detector([pair]), image, confidence=.85)) == 1


def test_screening_reasons_and_per_image_cache_preserve_source_images(tmp_path):
    rows = []
    for index in range(5):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (100, 100), (index, 0, 0)).save(path)
        rows.append({"path": str(path), "animal": f"dogs-world:dog:{index}", "species": "dog", "source": "dogs-world"})
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"unreadable source file")
    rows.append({**rows[0], "path": str(bad)})
    raw = [prediction([]), prediction([(15, .9, (320, 320, 200, 200))]),
           prediction([(16, .9, (320, 320, 200, 200))]),
           prediction([(16, .9, (100, 320, 100, 100)), (16, .8, (500, 320, 100, 100))]),
           prediction([(16, .9, (100, 320, 100, 100)), (15, .8, (500, 320, 100, 100))])]
    model = detector(raw)
    before = {row["path"]: sha256(row["path"]) for row in rows}
    excluded, report = screen_animals(rows, model, tmp_path / "cache", detector_sha256="a" * 64)
    assert report["counts"] == {"no_animal_detected": 1, "unexpected_species": 1, "single_expected_animal": 1,
                                "multiple_animals": 2, "unreadable": 1}
    assert len(excluded) == 5 and rows[2]["path"] not in excluded
    assert model.session.calls == 5
    assert {row["path"]: sha256(row["path"]) for row in rows} == before
    cached_excluded, _ = screen_animals(rows, model, tmp_path / "cache", detector_sha256="a" * 64)
    assert model.session.calls == 5 and cached_excluded == excluded
    model.session = Session([raw[2]])
    screen_animals([rows[2]], model, tmp_path / "cache", detector_sha256="b" * 64)
    assert model.session.calls == 1
    model.session = Session([raw[2]])
    screen_animals([rows[2]], model, tmp_path / "cache", detector_sha256="b" * 64, confidence=.5)
    assert model.session.calls == 1


def test_screening_fails_on_changed_input_or_model_errors_instead_of_silent_exclusion(tmp_path):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (100, 100)).save(image)
    row = {"path": str(image), "animal": "lcw:cat:1", "species": "cat", "source": "lcw", "sha256": "0" * 64}
    with pytest.raises(ValueError, match="changed"):
        screen_animals([row], detector([]), tmp_path / "cache", detector_sha256="a" * 64)
    row["sha256"] = sha256(image)

    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")

    model = detector([])
    model.session = SimpleNamespace(run=fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        screen_animals([row], model, tmp_path / "cache", detector_sha256="a" * 64)
    assert not list((tmp_path / "cache").glob("*.json"))
    with pytest.raises(ValueError, match="threshold"):
        animal_boxes(model, Image.new("RGB", (100, 100)), confidence=0)
