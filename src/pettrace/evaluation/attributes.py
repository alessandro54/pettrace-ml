"""Deterministic, identity-blind morphology extraction for E5."""

from hashlib import sha256
from copy import deepcopy
import json
from pathlib import Path
import re

from PIL import Image, ImageOps

from pettrace.data.verification import sha256 as file_sha256

MODEL_4B = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_2B = "Qwen/Qwen3-VL-2B-Instruct"

VOCABULARY = {
    "primary_color": ("black", "white", "brown", "cream", "orange", "gray", "tricolor", "unknown"),
    "secondary_color": ("none", "white", "black", "brown", "cream", "orange", "gray", "unknown"),
    "coat_pattern": ("solid", "bicolor", "tabby", "calico", "tortoiseshell", "spotted", "merle", "brindle", "unknown"),
    "coat_length": ("short", "medium", "long", "unknown"),
    "ear_shape": ("erect", "floppy", "semi_erect", "folded", "not_visible"),
    "white_markings": ("none", "chest", "paws", "muzzle", "forehead_blaze", "tail_tip", "neck_ring"),
    "distinctive_features": ("none", "collar", "dark_facial_mask", "heterochromia", "bobtail"),
}

PROMPT = """Inspect only the visible cat or dog in this cropped image. Return exactly one JSON
object with these keys and only values from the supplied closed vocabulary. Do not infer identity,
breed, name, source, or hidden traits. Use unknown/not_visible when the crop does not support a
decision. white_markings and distinctive_features are JSON arrays; use [\"none\"] when absent.
Vocabulary: {vocabulary}""".format(vocabulary=json.dumps(VOCABULARY, separators=(",", ":")))
PROMPT_SHA256 = sha256(PROMPT.encode()).hexdigest()
SCHEMA_VERSION = 2
STATUSES = {"ok", "repaired", "fallback"}
UNKNOWN_ATTRIBUTES = {
    "primary_color": "unknown",
    "secondary_color": "unknown",
    "coat_pattern": "unknown",
    "coat_length": "unknown",
    "ear_shape": "not_visible",
    "white_markings": ["none"],
    "distinctive_features": ["none"],
}


def validate_attributes(value):
    """Return a canonical closed-vocabulary record, normalizing common VLM list drift."""
    if not isinstance(value, dict) or set(value) != set(VOCABULARY):
        raise ValueError("Attribute output must contain exactly the fixed E5 schema")
    primary = value["primary_color"]
    if hasattr(primary, "tolist"):
        primary = primary.tolist()
    primary_candidates = list(primary) if isinstance(primary, (list, tuple)) else [primary]
    primary_candidates = [item for item in primary_candidates if item in VOCABULARY["primary_color"]]
    result = {}
    for field, allowed in VOCABULARY.items():
        item = value[field]
        if field in {"white_markings", "distinctive_features"}:
            if hasattr(item, "tolist"):
                item = item.tolist()
            if isinstance(item, tuple):
                item = list(item)
            if not isinstance(item, list) or not item or len(item) != len(set(item)):
                raise ValueError(f"{field} must be a nonempty list without duplicates")
            if any(entry not in allowed for entry in item) or ("none" in item and len(item) != 1):
                raise ValueError(f"Invalid {field} value")
            result[field] = sorted(item, key=allowed.index)
        else:
            if hasattr(item, "tolist"):
                item = item.tolist()
            if isinstance(item, (list, tuple)):
                choices = [entry for entry in item if entry in allowed]
                if field == "secondary_color":
                    item = next((entry for entry in choices if entry != result["primary_color"]), None)
                else:
                    item = choices[0] if choices else None
            if (field == "secondary_color" and item in {"none", "unknown"}
                    and len(primary_candidates) > 1):
                item = next((entry for entry in primary_candidates[1:] if entry in allowed), item)
            if item not in allowed:
                raise ValueError(f"Invalid {field} value: {item}")
            result[field] = item
    return result


def parse_attributes(text):
    """Parse the single JSON object emitted by the deterministic VLM."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            return validate_attributes(value)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError("The VLM did not return a valid E5 JSON object")


def yolo_crop(photo, pipeline):
    """Return exactly the crop seen by E4, before CLIP resize/normalization."""
    image = ImageOps.exif_transpose(Image.open(photo.path)).convert("RGB")
    return pipeline.detector.crop(image) if pipeline.detector is not None else image


class QwenAttributeExtractor:
    """Greedy Qwen3-VL inference whose model input is only a PIL crop and the fixed prompt."""

    def __init__(self, model_id=None, revision=None, token=None):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("E5 attribute extraction requires a CUDA runtime")
        memory_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        self.model_id = model_id or (MODEL_4B if memory_gib >= 20 else MODEL_2B)
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Qwen3-VL revision must be an immutable 40-character commit SHA")
        self.revision = revision
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.processor = AutoProcessor.from_pretrained(self.model_id, revision=revision, token=token)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id, revision=revision, token=token, dtype=self.dtype, device_map="auto")
        self.model.eval()
        self.generation_config = deepcopy(self.model.generation_config)
        self.generation_config.do_sample = False
        self.generation_config.temperature = None
        self.generation_config.top_p = None
        self.generation_config.top_k = None
        self.last_status = "ok"

    def _generate(self, messages):
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        generated = self.model.generate(
            **inputs, max_new_tokens=180, generation_config=self.generation_config)
        return self.processor.decode(
            generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    def __call__(self, crop):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": crop}, {"type": "text", "text": PROMPT}]}]
        answer = self._generate(messages)
        try:
            self.last_status = "ok"
            return parse_attributes(answer)
        except ValueError:
            repair = [{"role": "user", "content": [{"type": "text", "text":
                f"Convert this attempted answer to exactly the JSON schema requested below. "
                f"Return JSON only.\n{PROMPT}\nAttempted answer:\n{answer}"}]}]
            try:
                result = parse_attributes(self._generate(repair))
                self.last_status = "repaired"
                return result
            except ValueError:
                self.last_status = "fallback"
                return dict(UNKNOWN_ATTRIBUTES)


def save_attribute_cache(records, path, *, model_id, revision, datasets):
    """Write a versioned Parquet cache plus a content-addressed receipt."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    required = {"photo_sha256", "dataset", "split", "attribute_status", *VOCABULARY}
    if not len(frame) or set(frame) != required or frame["photo_sha256"].duplicated().any():
        raise ValueError("Attribute cache rows are empty, duplicated, or have the wrong schema")
    if not set(frame["attribute_status"]) <= STATUSES:
        raise ValueError("Attribute cache contains an invalid extraction status")
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.sort_values(["dataset", "photo_sha256"]).to_parquet(temporary, index=False)
    temporary.replace(path)
    digest = sha256(path.read_bytes()).hexdigest()
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "model": model_id,
        "revision": revision,
        "decoding": {"do_sample": False},
        "prompt_sha256": PROMPT_SHA256,
        "datasets": datasets,
        "rows": len(frame),
        "status_counts": {
            status: int((frame["attribute_status"] == status).sum()) for status in sorted(STATUSES)
        },
        "parquet": path.name,
        "sha256": digest,
        "identity_metadata_passed_to_model": False,
    }
    path.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def extract_attribute_cache(groups, pipeline, extractor, path, datasets):
    """Extract once, checkpoint as JSONL, and finish as an immutable Parquet artifact.

    ``groups`` maps a public dataset label to photos. Identity labels are deliberately absent from
    both the checkpoint and the VLM call.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = path.with_suffix(".jsonl")
    checkpoint_receipt = path.with_suffix(".checkpoint.json")
    checkpoint_identity = {
        "schema_version": SCHEMA_VERSION,
        "model": extractor.model_id,
        "revision": extractor.revision,
        "prompt_sha256": PROMPT_SHA256,
    }
    if checkpoint.is_file():
        if (not checkpoint_receipt.is_file()
                or json.loads(checkpoint_receipt.read_text()) != checkpoint_identity):
            raise ValueError("Attribute checkpoint belongs to a different model, revision, or prompt")
    else:
        checkpoint_receipt.write_text(json.dumps(checkpoint_identity, indent=2) + "\n")
    completed = {}
    if checkpoint.is_file():
        for line in checkpoint.read_text().splitlines():
            record = json.loads(line)
            completed[(record["dataset"], record["photo_sha256"])] = record
    total = sum(len(photos) for photos in groups.values())
    done = 0
    with checkpoint.open("a") as stream:
        for dataset, photos in groups.items():
            for photo in photos:
                digest = file_sha256(photo.path)
                key = dataset, digest
                if key not in completed:
                    record = {
                        "photo_sha256": digest,
                        "dataset": dataset,
                        "split": photo.split,
                        **extractor(yolo_crop(photo, pipeline)),
                        "attribute_status": getattr(extractor, "last_status", "ok"),
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    stream.flush()
                    completed[key] = record
                done += 1
                if done == total or done % 100 == 0:
                    print(f"VLM attributes: {done}/{total}", flush=True)
    receipt = save_attribute_cache(
        list(completed.values()), path, model_id=extractor.model_id,
        revision=extractor.revision, datasets=datasets)
    checkpoint.unlink()
    checkpoint_receipt.unlink()
    return receipt


def load_attribute_cache(path, expected_sha256=None):
    """Load and validate a previously extracted E5 cache."""
    import pandas as pd

    path = Path(path)
    digest = sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Attribute cache SHA-256 differs from the frozen receipt")
    rows = pd.read_parquet(path).to_dict("records")
    for row in rows:
        if row.get("attribute_status") not in STATUSES:
            raise ValueError("Attribute cache contains an invalid extraction status")
        validate_attributes({field: row[field] for field in VOCABULARY})
    return rows, digest


def align_attributes(photos, records):
    """Align a cache to evaluation photos by content hash, never by identity or path."""
    by_hash = {record["photo_sha256"]: validate_attributes(
        {field: record[field] for field in VOCABULARY}) for record in records}
    aligned = []
    for photo in photos:
        digest = file_sha256(photo.path)
        if digest not in by_hash:
            raise ValueError(f"Missing cached attributes for photo SHA-256 {digest}")
        aligned.append(by_hash[digest])
    return aligned
