"""Public/private benchmark isolation and reproducible domain comparison."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pettrace.evaluation import metrics
from pettrace.evaluation.research import public_test_dataset, save_benchmark_comparison


def prepared(tmp_path):
    rows = []
    for species in ("cat", "dog"):
        for split in ("train", "val", "test"):
            for animal in range(2):
                for photo in range(2):
                    identity = f"{species}-{split}-{animal}"
                    rows.append({"animal": identity, "species": species, "split": split,
                                 "source": "lcw" if species == "cat" else "dogs-world",
                                 "path": str(tmp_path / f"{identity}-{photo}.jpg"),
                                 "sha256": f"{identity}-{photo}"})
    return SimpleNamespace(root=tmp_path, fingerprint="prepared", rows=rows)


def receipt():
    return {"repo": "test/public", "version": "v1", "commit": "a"*40, "fingerprint": "prepared"}


def test_public_view_only_contains_identity_disjoint_test_and_keeps_version(tmp_path):
    original = prepared(tmp_path)
    ds = public_test_dataset(original, receipt())
    assert len(ds.photos) == 8 and len(ds.animals) == 4
    assert all(p.split == "test" and "-test-" in p.animal for p in ds.photos)
    assert ds.commit == "a"*40 and ds.version == "v1"
    # Location does not influence the scientific selection fingerprint.
    moved = prepared(tmp_path / "moved")
    assert public_test_dataset(moved, receipt()).fingerprint == ds.fingerprint
    assert len(original.rows) == 24


def test_public_view_rejects_leakage_and_unpinned_or_changed_receipt(tmp_path):
    original = prepared(tmp_path)
    for bad in (None, {**receipt(), "commit": ""}, {**receipt(), "fingerprint": "other"}):
        with pytest.raises(ValueError, match="receipt"):
            public_test_dataset(original, bad)
    test = next(r for r in original.rows if r["split"] == "test")
    original.rows.append({**test, "split": "train"})
    with pytest.raises(ValueError, match="identity-disjoint"):
        public_test_dataset(original, receipt())


def test_comparison_keeps_each_gallery_size_and_rank2(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    def evaluation(count):
        animals = [f"a{i}" for i in range(count) for _ in range(2)]
        embeddings = np.repeat(np.eye(count), 2, axis=0)
        result = metrics.retrieval(embeddings, animals, ["cat"]*len(animals))
        return SimpleNamespace(result={"test": result}, dataset=SimpleNamespace(
            name=f"set-{count}", commit=f"commit-{count}", fingerprint=f"digest-{count}"))
    public, private = evaluation(3), evaluation(2)
    table, figure = save_benchmark_comparison(public, private, tmp_path, "E1")
    try:
        records = json.loads((tmp_path / "benchmark-comparison.json").read_text())["records"]
        assert {r["gallery_animals"] for r in records} == {2, 3}
        assert all(r["rank1"] == r["rank2"] == r["mAP"] == 1.0 for r in records)
        assert "Rank-2" in table and (tmp_path / "benchmark-comparison.png").is_file()
    finally:
        plt.close(figure)


def test_long_identity_plot_labels_are_unique_reversible_and_species_specific():
    from pettrace.evaluation.figures import identity_labels

    photos = [SimpleNamespace(animal=name, species=sp) for name, sp in [
        ("cat-001", "cat"), ("lcw:cat:123456789012345678901234", "cat"),
        ("dogs-world:dog:123456789012345678901234", "dog")]]
    labels = identity_labels(SimpleNamespace(photos=photos))
    assert set(labels) == {p.animal for p in photos}
    assert len(set(labels.values())) == len(photos)
    assert max(map(len, labels.values())) <= 18
    assert labels[photos[0].animal].startswith("cat-")
    assert labels[photos[-1].animal].startswith("dog-")
