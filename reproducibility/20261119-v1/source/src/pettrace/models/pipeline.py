# Mirror of apps/vision/src/vision/ml/pipeline.py in the PetTrace app repo (what the vision
# service serves). Keep both identical: registration and evaluation embed with this copy, vision
# with its own, and the golden fingerprint stored on every registered version (checked by vision at
# startup and by `make verify-model`) fails if they ever diverge.

"""Run a `pettrace-embedder` version from its artifacts: pipeline.yaml + ONNX.

Preprocessing is implemented here from the spec (numpy + PIL), not delegated to Hugging Face, so a
version is fully described by its own files. This is what the vision service serves with, and what
the registration/verification scripts use: no torch or transformers at runtime.

Steps (pipeline.yaml): decode → detect_crop (E2+, optional) → clip_preprocess → onnx encoder.
"""

import os

import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image

_RESAMPLE = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR}


class Detector:
    """Animal crop (E2): YOLOv8 exported to ONNX, COCO classes. The highest-confidence box of an
    allowed class (cat, dog) is cropped with a margin; no box above `conf` → the full image.

    Output of the ONNX model: [1, 4 + num_classes, anchors] with (cx, cy, w, h) in the 640×640
    letterboxed frame followed by per-class scores. Only the best box is used, so no NMS is needed
    (NMS never removes the single highest-scoring box).
    """

    def __init__(self, model_dir: str, step: dict):
        self.session = ort.InferenceSession(
            os.path.join(model_dir, step["file"]), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.size = int(step.get("input_size", 640))
        self.conf = float(step.get("conf", 0.25))
        self.classes = [int(c) for c in step.get("classes", [15, 16])]
        self.margin = float(step.get("margin", 0.10))

    def letterbox(self, img: Image.Image) -> tuple[np.ndarray, float, float, float]:
        """Ultralytics-style letterbox: keep aspect, pad with gray 114 to a square."""
        w, h = img.size
        r = min(self.size / w, self.size / h)
        nw, nh = round(w * r), round(h * r)
        resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        left, top = round((self.size - nw) / 2 - 0.1), round((self.size - nh) / 2 - 0.1)
        canvas.paste(resized, (left, top))
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return x, r, left, top

    def box(self, img: Image.Image) -> tuple[float, int, tuple[float, float, float, float]] | None:
        """Best (score, class, x1y1x2y2 in image pixels, without margin) or None."""
        x, r, left, top = self.letterbox(img)
        out = self.session.run(None, {self.input_name: x})[0][0]  # [4 + C, anchors]
        scores = out[4:][self.classes]  # [len(classes), anchors]
        cls_i, anchor = np.unravel_index(int(np.argmax(scores)), scores.shape)
        score = float(scores[cls_i, anchor])
        if score < self.conf:
            return None
        cx, cy, bw, bh = (float(v) for v in out[:4, anchor])
        x1, y1 = (cx - bw / 2 - left) / r, (cy - bh / 2 - top) / r
        x2, y2 = (cx + bw / 2 - left) / r, (cy + bh / 2 - top) / r
        return score, self.classes[cls_i], (x1, y1, x2, y2)

    def crop(self, img: Image.Image) -> Image.Image:
        found = self.box(img)
        if found is None:
            return img
        _, _, (x1, y1, x2, y2) = found
        w, h = x2 - x1, y2 - y1
        box = (
            max(0, x1 - self.margin * w),
            max(0, y1 - self.margin * h),
            min(img.width, x2 + self.margin * w),
            min(img.height, y2 + self.margin * h),
        )
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            return img
        return img.crop(box)


class Pipeline:
    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, "pipeline.yaml")) as f:
            self.spec = yaml.safe_load(f)
        steps = {s["op"]: s for s in self.spec["steps"]}
        detect = steps.get("detect_crop", {})
        self.detector = Detector(model_dir, detect) if detect.get("enabled") else None
        self.pre = steps["clip_preprocess"]
        onnx_step = steps["onnx"]
        self.session = ort.InferenceSession(
            os.path.join(model_dir, onnx_step["file"]), providers=["CPUExecutionProvider"]
        )
        self.input_name, self.output_name = onnx_step["input"], onnx_step["output"]

    def preprocess(self, img: Image.Image) -> np.ndarray:
        p = self.pre
        img = img.convert("RGB")
        if self.detector is not None:
            img = self.detector.crop(img)
        short = p["resize_shortest_edge"]
        w, h = img.size
        if w <= h:
            size = (short, int(short * h / w))
        else:
            size = (int(short * w / h), short)
        img = img.resize(size, _RESAMPLE[p["resample"]])
        ch, cw = p["center_crop"]
        left, top = (img.width - cw) // 2, (img.height - ch) // 2
        img = img.crop((left, top, left + cw, top + ch))
        x = np.asarray(img, dtype=np.float32) * p["rescale_factor"]
        x = (x - np.array(p["mean"], dtype=np.float32)) / np.array(p["std"], dtype=np.float32)
        return x.transpose(2, 0, 1)  # HWC → CHW

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        batch = np.stack([self.preprocess(i) for i in images]).astype(np.float32)
        return self.session.run([self.output_name], {self.input_name: batch})[0]
