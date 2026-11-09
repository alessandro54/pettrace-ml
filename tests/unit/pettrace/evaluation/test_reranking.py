import numpy as np
import pytest
from types import SimpleNamespace

from pettrace.evaluation.reranking import (
    attribute_similarity,
    calibrate,
    gated_topk_rerank,
    rerank_evaluation,
)


def rerank(visual, text, **overrides):
    options = {
        "visual_margin_threshold": 0.1,
        "text_advantage_threshold": 0.2,
        "text_weight": 0.2,
        "top_k": min(5, len(visual)),
    }
    options.update(overrides)
    return gated_topk_rerank(visual, text, **options)


def test_closed_visual_gate_preserves_ranking():
    order, audit = rerank([0.9, 0.7, 0.6], [0.0, 1.0, 0.0])

    assert order.tolist() == [0, 1, 2]
    assert audit["gate_open"] is False
    assert audit["changed"] is False


def test_text_gate_rejects_weak_advantage():
    order, audit = rerank([0.9, 0.86, 0.7], [0.1, 0.25, 0.0])

    assert order.tolist() == [0, 1, 2]
    assert audit["gate_open"] is True
    assert audit["changed"] is False


def test_double_gate_can_rescue_ambiguous_top1():
    order, audit = rerank([0.9, 0.86, 0.7], [0.1, 0.5, 0.0])

    assert order.tolist() == [1, 0, 2]
    assert audit["changed"] is True
    assert audit["reranked_top1"] == 1


def test_candidate_outside_topk_cannot_be_promoted():
    order, audit = rerank(
        [0.9, 0.89, 0.88, 0.87, 0.86, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
        top_k=5,
    )

    assert order.tolist() == [0, 1, 2, 3, 4, 5]
    assert audit["changed"] is False


@pytest.mark.parametrize(
    ("visual", "text", "overrides"),
    [
        ([0.9], [0.1], {}),
        ([0.9, 0.8], [0.1], {}),
        ([0.9, np.nan], [0.1, 0.2], {}),
        ([0.9, 0.8], [0.1, 0.2], {"text_weight": -1}),
        ([0.9, 0.8], [0.1, 0.2], {"top_k": 1}),
    ],
)
def test_invalid_inputs_are_rejected(visual, text, overrides):
    with pytest.raises(ValueError):
        rerank(visual, text, **overrides)


def attributes(color):
    return {
        "primary_color": color, "secondary_color": "none", "coat_pattern": "solid",
        "coat_length": "short", "ear_shape": "erect", "white_markings": ["none"],
        "distinctive_features": ["none"],
    }


def test_attribute_score_ignores_unknown_and_penalizes_contradictions():
    black = attributes("black")
    unknown = {**black, "primary_color": "unknown"}
    white = {**black, "primary_color": "white", "coat_pattern": "bicolor"}

    assert attribute_similarity(black, black) == 1.0
    assert attribute_similarity(black, unknown) == 1.0
    assert attribute_similarity(black, white) < attribute_similarity(black, black)


def test_unknown_fallback_cannot_create_textual_advantage():
    unknown = attributes("unknown") | {
        "secondary_color": "unknown", "coat_pattern": "unknown",
        "coat_length": "unknown", "ear_shape": "not_visible",
    }

    assert attribute_similarity(unknown, unknown) == 0.0


def test_reranker_can_rescue_visual_twin_without_changing_gallery_protocol():
    embeddings = np.array([[1.0, 0.0], [0.8, 0.6], [0.99, 0.1], [0.0, 1.0]])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    photos = [SimpleNamespace(animal=animal, species="cat") for animal in ("a", "a", "b", "b")]
    evaluation = SimpleNamespace(photos=photos, embeddings=embeddings)
    records = [attributes(color) for color in ("black", "black", "white", "white")]

    result = rerank_evaluation(
        evaluation, records, text_weight=0.5, gated=True,
        visual_margin_threshold=0.3, text_advantage_threshold=0.2, top_k=2)

    assert result["global"]["queries"] == 4
    assert result["audit"]["rescued_queries"] >= 1
    assert all(row["gallery_animals"] == 2 for row in result["per_query"])


def test_reranker_preserves_gallery_order_for_exact_visual_ties():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    photos = [SimpleNamespace(animal=animal, species="cat") for animal in ("a", "b", "a")]
    evaluation = SimpleNamespace(photos=photos, embeddings=embeddings)

    result = rerank_evaluation(
        evaluation, [attributes("black")] * 3, text_weight=0.1, gated=True,
        visual_margin_threshold=0.0, text_advantage_threshold=0.1, top_k=2)

    query = next(row for row in result["per_query"] if row["query"] == 0)
    assert query["baseline_top1"] == "b"


def test_calibration_prefers_zero_damage_before_higher_rank1(monkeypatch):
    def result(*_args, text_weight, **_kwargs):
        damaged = int(text_weight == 0.2)
        return {
            "global": {"rank1": 0.9 if damaged else 0.8, "mAP": 0.7},
            "audit": {"damaged_queries": damaged, "gate_activation_rate": 0.1,
                      "rescued_queries": 1, "changed_queries": 1, "queries": 10},
        }

    monkeypatch.setattr("pettrace.evaluation.reranking.rerank_evaluation", result)
    selected, _ = calibrate(None, None, [(0.1, 0.2, 0.1), (0.1, 0.1, 0.1)])

    assert selected["lambda"] == 0.1


def test_calibration_minimizes_damage_when_no_candidate_is_damage_free(monkeypatch):
    def result(*_args, text_weight, **_kwargs):
        damaged = 2 if text_weight == 0.2 else 1
        return {
            "global": {"rank1": 0.9 if damaged == 2 else 0.8, "mAP": 0.7},
            "audit": {"damaged_queries": damaged, "gate_activation_rate": 0.1,
                      "rescued_queries": 1, "changed_queries": 1, "queries": 10},
        }

    monkeypatch.setattr("pettrace.evaluation.reranking.rerank_evaluation", result)
    selected, _ = calibrate(None, None, [(0.1, 0.2, 0.1), (0.1, 0.1, 0.1)])

    assert selected["lambda"] == 0.1
