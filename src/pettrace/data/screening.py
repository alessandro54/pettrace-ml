"""Count animals with the frozen E2 detector; never infer individual identity."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import PIL
from PIL import Image, ImageOps

from pettrace.models import pipeline
from pettrace.data.curation import sha256

SPECIES = {15: "cat", 16: "dog"}


def animal_boxes(detector, image, *, confidence=0.25, iou=0.5):
    """YOLOv8 raw output → class-agnostic NMS boxes in oriented image coordinates.

    Class-agnostic suppression avoids counting overlapping cat/dog predictions for
    the same animal twice. Heavy occlusion can still merge/miss real animals.
    This is separate from Detector.box/crop, preserving E2 inference exactly.
    """
    if not 0 < confidence < 1 or not 0 < iou < 1:
        raise ValueError("Detection thresholds must lie strictly between zero and one")
    if not set(SPECIES) <= set(detector.classes):
        raise ValueError("Screening requires both COCO cat=15 and dog=16")
    tensor, scale, left, top = detector.letterbox(image)
    output = np.asarray(detector.session.run(None, {detector.input_name: tensor})[0])
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] < 21:
        raise ValueError("Expected raw YOLOv8 [1, 4 + classes, anchors] output")
    raw = output[0]
    if raw.shape[1] == 0:
        return []
    scores = raw[[4 + cls for cls in SPECIES]]
    if not np.isfinite(raw[:4]).all() or not np.isfinite(scores).all():
        raise ValueError("Non-finite detector output")
    class_indices = np.argmax(scores, axis=0)
    best = scores[class_indices, np.arange(scores.shape[1])]
    chosen = np.flatnonzero(best >= confidence)
    if not len(chosen):
        return []
    cx, cy, width, height = raw[:4, chosen]
    boxes = np.stack(((cx - width / 2 - left) / scale, (cy - height / 2 - top) / scale,
                      (cx + width / 2 - left) / scale, (cy + height / 2 - top) / scale), axis=1)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image.width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image.height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, chosen = boxes[valid], chosen[valid]
    order = np.argsort(-best[chosen], kind="stable")
    kept = []
    while len(order):
        index = order[0]
        anchor = chosen[index]
        kept.append({"score": float(best[anchor]), "class_id": list(SPECIES)[class_indices[anchor]],
                     "box": [float(value) for value in boxes[index]]})
        others = order[1:]
        low = np.maximum(boxes[index, :2], boxes[others, :2])
        high = np.minimum(boxes[index, 2:], boxes[others, 2:])
        intersection = np.prod(np.maximum(high - low, 0), axis=1)
        area = np.prod(boxes[index, 2:] - boxes[index, :2])
        areas = np.prod(boxes[others, 2:] - boxes[others, :2], axis=1)
        overlap = intersection / np.maximum(area + areas - intersection, 1e-12)
        order = others[overlap <= iou]
    return kept


def screen_animals(rows, detector, cache, *, detector_sha256, confidence=0.25, iou=0.5):
    """Keep exactly one detected animal of the expected species; cache by bytes/model/code.

    Caller verifies the loaded detector's artifact hash. Original photos, labels and
    manual decisions remain unchanged; return path exclusions for pre-split curation.
    Model/runtime errors propagate, rather than masquerading as unreadable images.
    """
    if not 0 < confidence < 1 or not 0 < iou < 1:
        raise ValueError("Detection thresholds must lie strictly between zero and one")
    if not set(SPECIES) <= set(detector.classes):
        raise ValueError("Screening requires both COCO cat=15 and dog=16")
    if not rows:
        raise ValueError("No candidate images to screen")
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    config = {"detector_sha256": detector_sha256, "confidence": confidence, "nms_iou": iou,
              "nms": "class-agnostic", "classes": SPECIES, "input_size": detector.size,
              "implementation_sha256": sha256(__file__), "pipeline_sha256": sha256(pipeline.__file__),
              "onnxruntime": ort.__version__, "numpy": np.__version__, "pillow": PIL.__version__}
    exclusions, records = {}, []
    for number, row in enumerate(rows, 1):
        if row["species"] not in SPECIES.values():
            raise ValueError("Expected cat or dog source labels")
        digest = sha256(row["path"])
        if row.get("sha256", digest) != digest:
            raise ValueError("Image changed before animal screening")
        key = hashlib.sha256(json.dumps({"image": digest, "config": config}, sort_keys=True).encode()).hexdigest()
        saved = cache / f"{key}.json"
        if saved.exists():
            result = json.loads(saved.read_text())
            if result.get("fingerprint") != key or result.get("image_sha256") != digest:
                raise ValueError("Animal detection cache mismatch")
        else:
            result = {"fingerprint": key, "image_sha256": digest, "boxes": [], "decode_error": None}
            try:
                with Image.open(row["path"]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    image.load()
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                result["decode_error"] = f"{type(exc).__name__}: {exc}"
            else:
                result["boxes"] = animal_boxes(detector, image, confidence=confidence, iou=iou)
            partial = saved.with_suffix(".partial")
            partial.write_text(json.dumps(result))
            partial.replace(saved)
        boxes = result["boxes"]
        if result["decode_error"]:
            reason = "unreadable"
        elif len(boxes) > 1:
            reason = "multiple_animals"
        elif not boxes:
            reason = "no_animal_detected"
        elif SPECIES[boxes[0]["class_id"]] != row["species"]:
            reason = "unexpected_species"
        else:
            reason = "single_expected_animal"
        if reason != "single_expected_animal":
            exclusions[row["path"]] = reason
        records.append({**{field: row[field] for field in ("path", "source", "species", "animal")},
                        **result, "reason": reason})
        if number % 100 == 0 or number == len(rows):
            print(f"YOLO animal screening: {number}/{len(rows)}", flush=True)
    report = {"config": config, "photos": len(rows), "counts": dict(Counter(r["reason"] for r in records)),
              "records": records,
              "limitation": "A single detection does not verify identity. Occlusion, missed animals and false "
                            "detections remain possible; these thresholds have not been calibrated on this pilot."}
    return exclusions, report
