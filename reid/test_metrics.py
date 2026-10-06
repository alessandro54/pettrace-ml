"""Unit tests for the evaluation protocol: `python -m pytest ml/reid -q`."""

import numpy as np

from reid import metrics


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_perfect_separation():
    # 3 animals × 2 photos, each animal along its own axis.
    E = unit([[1, 0, 0], [1, 0.05, 0], [0, 1, 0], [0.05, 1, 0], [0, 0, 1], [0, 0.05, 1]])
    animals = ["a", "a", "b", "b", "c", "c"]
    r = metrics.retrieval(E, animals, ["cat"] * 6)
    assert r["global"]["rank1"] == 1.0 and r["global"]["mAP"] == 1.0
    assert r["global"]["gallery_animals"] == 3


def test_lookalike_confusion_counts_as_rank2():
    # b's second photo sits right next to a: query b2 finds a first.
    E = unit([[1, 0], [0.98, 0.2], [0.2, 1], [0.99, 0.1]])
    r = metrics.retrieval(E, ["a", "a", "b", "b"], ["cat"] * 4)
    q = {row["query"]: row for row in r["per_query"]}
    assert q[3]["rank"] == 2 and q[3]["rival"] == "a"
    assert r["global"]["rank5"] == 1.0  # only 2 animals: rank-5 is always a hit


def test_gallery_is_same_species_only():
    # The dog photo is identical to the cat query but must not compete with it.
    E = unit([[1, 0, 0], [0.9, 0.4, 0], [0, 0.6, 1], [0, 0.5, 1],
              [1, 0, 0], [0, 0, 1], [0.1, 1, 0], [0, 1, 0.1]])
    animals = ["cat1", "cat1", "cat2", "cat2", "dog1", "dog1", "dog2", "dog2"]
    r = metrics.retrieval(E, animals, ["cat"] * 4 + ["dog"] * 4)
    assert r["cat"]["rank1"] == 1.0 and r["cat"]["gallery_animals"] == 2
    assert set(r) == {"global", "cat", "dog", "per_query"}


def test_singleton_animals_are_skipped():
    E = unit([[1, 0], [0.9, 0.1], [0, 1]])
    r = metrics.retrieval(E, ["a", "a", "solo"], ["cat"] * 3)
    assert r["global"]["queries"] == 2


def test_eer_and_threshold():
    same = np.array([0.9, 0.85, 0.8, 0.7])
    diff = np.array([0.6, 0.65, 0.75, 0.5])
    e, t = metrics.eer(same, diff)
    assert 0.0 <= e <= 0.5 and 0.6 <= t <= 0.85
    rates = metrics.at_threshold(same, diff, t)
    assert abs(rates["far"] - rates["frr"]) <= 0.25


def test_eer_needs_both_kinds_of_pairs():
    e, t = metrics.eer(np.array([0.9]), np.array([]))
    assert np.isnan(e) and np.isnan(t)


def test_single_animal_species_is_skipped():
    # Two cats plus one dog: the dog has no rival, so it must not count as a 100 % hit.
    E = unit([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0.1, 0.9, 0], [0, 0, 1], [0, 0.1, 0.9]])
    r = metrics.retrieval(E, ["a", "a", "b", "b", "d", "d"], ["cat"] * 4 + ["dog"] * 2)
    assert "dog" not in r
    assert r["global"]["queries"] == 4


def test_pairs_skip_single_animal_species():
    E = unit([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [1, 1], [0.9, 1]])
    same, diff = metrics.pair_scores(E, ["a", "a", "b", "b", "d", "d"], ["cat"] * 4 + ["dog"] * 2)
    assert len(same) == 2 and len(diff) == 4  # the lone dog contributes nothing
