"""Gallery-only animals must remain visible in retrieval diagnostics."""

from types import SimpleNamespace

import numpy as np
import pytest

from pettrace import figures, metrics


def test_singleton_rival_in_summary_and_confusion():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    animals = ["a", "a", "b", "b", "solo"]
    species = ["cat"] * len(animals)
    embeddings = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]], dtype=float)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    result = metrics.retrieval(embeddings, animals, species)
    same, diff = metrics.pair_scores(embeddings, animals, species)
    ev = SimpleNamespace(
        photos=[SimpleNamespace(animal=a, species=s) for a, s in zip(animals, species)],
        embeddings=embeddings, same=same, diff=diff,
        result={"test": result, "val": {"eer": 0.5, "threshold": 0.5}},
        dataset=SimpleNamespace(name="synthetic", commit="test"), protocol="zero-shot",
    )
    assert "solo" not in {r["animal"] for r in result["per_query"]}
    assert any(r["rival"] == "solo" and r["rank"] > 1 for r in result["per_query"])
    summary = figures.summary(ev)
    assert summary["per_animal"]["a"]["rivals"]["solo"] == 2
    fig = figures.confusion(ev)
    try:
        assert [tick.get_text() for tick in fig.axes[0].get_xticklabels()] == ["a", "b", "solo"]
        assert fig.axes[0].images[0].get_array()[0, 2] == 1.0
    finally:
        plt.close(fig)
