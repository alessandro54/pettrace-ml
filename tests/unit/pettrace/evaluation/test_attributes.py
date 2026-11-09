import json
from types import SimpleNamespace

import numpy as np
import pytest

from pettrace.evaluation.attributes import (
    QwenAttributeExtractor,
    UNKNOWN_ATTRIBUTES,
    extract_attribute_cache,
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

    with pytest.raises(ValueError, match="different model, revision, or prompt"):
        extract_attribute_cache({}, None, extractor, path, {})
