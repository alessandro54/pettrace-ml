import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pettrace.evaluation.attributes import (
    QwenAttributeExtractor,
    UNKNOWN_ATTRIBUTES,
    attribute_cache_identity,
    extract_attribute_cache,
    load_attribute_cache,
    parse_attributes,
    validate_attributes,
)


def record(**overrides):
    value = {
        "primary_color": "black",
        "secondary_color": "white",
        "coat_pattern": "bicolor",
        "coat_length": "short",
        "ear_shape": "erect",
        "white_markings": ["chest", "paws"],
        "distinctive_features": ["none"],
    }
    value.update(overrides)
    return value


def test_closed_vocabulary_output_is_canonical_and_accepts_parquet_arrays():
    value = record(white_markings=np.array(["paws", "chest"]))

    assert validate_attributes(value)["white_markings"] == ["chest", "paws"]
    assert parse_attributes(f"```json\n{json.dumps(record())}\n```") == record()


def test_scalar_lists_are_canonicalized_without_losing_the_second_color():
    value = record(primary_color=["black", "white"], secondary_color=["black", "white"])

    assert validate_attributes(value)["primary_color"] == "black"
    assert validate_attributes(value)["secondary_color"] == "white"


def test_extractor_repairs_invalid_format_then_falls_back_to_unknown():
    extractor = QwenAttributeExtractor.__new__(QwenAttributeExtractor)
    answers = iter(["not json", json.dumps(record())])
    extractor._generate = lambda messages: next(answers)

    assert extractor(None) == record()
    assert extractor.last_status == "repaired"

    extractor._generate = lambda messages: "still not json"
    assert extractor(None) == UNKNOWN_ATTRIBUTES
    assert extractor.last_status == "fallback"


def test_extractor_batches_visual_generation_and_repairs_only_invalid_rows():
    extractor = QwenAttributeExtractor.__new__(QwenAttributeExtractor)
    extractor._generate_many = lambda messages: [json.dumps(record()), "not json"]
    extractor._generate = lambda messages: json.dumps(record(primary_color="brown"))

    results = extractor.extract_many([object(), object()])

    assert results == [(record(), "ok"), (record(primary_color="brown"), "repaired")]


def test_extractor_forces_left_padding_and_disables_model_sampling(monkeypatch):
    processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))
    model = SimpleNamespace(
        generation_config=SimpleNamespace(do_sample=True, temperature=0.7, top_p=0.8, top_k=20),
        eval=lambda: None,
    )
    torch = SimpleNamespace(
        bfloat16="bfloat16", float16="float16",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: False,
            get_device_properties=lambda _: SimpleNamespace(total_memory=16 * 2**30),
        ),
    )
    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
        Qwen3VLForConditionalGeneration=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: model),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    extractor = QwenAttributeExtractor(model_id="Qwen/example", revision="a" * 40)

    assert extractor.processor.tokenizer.padding_side == "left"
    assert extractor.generation_config.do_sample is False
    assert extractor.generation_config.temperature is None

    class Inputs(dict):
        def to(self, device):
            return self

    calls = {}
    inputs = Inputs(input_ids=np.array([[1, 2]]))
    extractor.processor = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: inputs,
        batch_decode=lambda values, **kwargs: ["decoded"],
    )
    extractor.model = SimpleNamespace(
        device="cuda", generate=lambda **kwargs: calls.update(kwargs) or np.array([[1, 2, 3]]))

    assert extractor._generate_many([[{"role": "user"}]]) == ["decoded"]
    assert calls["do_sample"] is False
    assert calls["use_model_defaults"] is False


@pytest.mark.parametrize("bad", [
    record(primary_color="purple"),
    record(white_markings=[]),
    record(distinctive_features=["none", "collar"]),
    {"primary_color": "black"},
])
def test_schema_drift_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_attributes(bad)


def test_resume_rejects_a_checkpoint_from_another_vlm_revision(tmp_path):
    path = tmp_path / "attributes-v1.parquet"
    path.with_suffix(".jsonl").write_text("")
    path.with_suffix(".checkpoint.json").write_text(json.dumps({
        "model": "Qwen/example", "revision": "old", "prompt_sha256": "old",
    }))
    extractor = SimpleNamespace(model_id="Qwen/example", revision="new")

    with pytest.raises(ValueError, match="different model, prompt, or dataset"):
        extract_attribute_cache({}, None, extractor, path, {})


def test_interrupted_extraction_resumes_without_repeating_finished_rows(tmp_path, monkeypatch):
    paths = []
    for index in range(3):
        path = tmp_path / f"photo-{index}.jpg"
        path.write_bytes(f"photo-{index}".encode())
        paths.append(SimpleNamespace(path=path, split="val"))
    monkeypatch.setattr(
        "pettrace.evaluation.attributes.yolo_crop", lambda photo, pipeline: photo.path.name)
    saved = {}
    monkeypatch.setattr(
        "pettrace.evaluation.attributes.save_attribute_cache",
        lambda records, path, **kwargs: saved.update(records=records) or {"sha256": "saved"})
    cache = tmp_path / "attributes.parquet"
    datasets = {"public": {"version": "v2", "commit": "a" * 40}}

    class Extractor:
        model_id = "Qwen/example"
        revision = "b" * 40
        batch_size = 2

        def __init__(self, fail=False):
            self.fail = fail
            self.calls = []

        def extract_many(self, crops):
            self.calls.append(list(crops))
            if self.fail and len(self.calls) == 2:
                raise RuntimeError("runtime stopped")
            return [(record(), "ok") for _ in crops]

    interrupted = Extractor(fail=True)
    with pytest.raises(RuntimeError, match="runtime stopped"):
        extract_attribute_cache({"validation": paths}, None, interrupted, cache, datasets)
    assert len(cache.with_suffix(".jsonl").read_text().splitlines()) == 2

    resumed = Extractor()
    extract_attribute_cache({"validation": paths}, None, resumed, cache, datasets)

    assert resumed.calls == [["photo-2.jpg"]]
    assert len(saved["records"]) == 3
    assert not cache.with_suffix(".jsonl").exists()


def test_finished_cache_rejects_different_dataset_identity(tmp_path):
    path = tmp_path / "attributes.parquet"
    datasets = {"public": {"version": "v2", "commit": "a" * 40}}
    path.write_bytes(b"parquet placeholder")
    path.with_suffix(".receipt.json").write_text(json.dumps(
        attribute_cache_identity("Qwen/example", "b" * 40, datasets)))
    expected_identity = attribute_cache_identity(
        "Qwen/example", "b" * 40,
        {"public": {"version": "v3", "commit": "c" * 40}})

    with pytest.raises(ValueError, match="different model, prompt, or dataset"):
        load_attribute_cache(path, expected_identity=expected_identity)
