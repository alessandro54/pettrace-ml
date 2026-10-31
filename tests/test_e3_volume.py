"""Source acquisition percentages must not be confused with effective training volume."""
import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def cell(cell_id, pct=20):
    notebook = json.loads((ROOT / f"notebooks/e3_mlp_head_triplet_{pct}.ipynb").read_text())
    return "".join(next(c for c in notebook["cells"] if c["id"] == cell_id)["source"])


def coverage_functions():
    from pettrace.data.volume import source_sample_metrics, dataset_coverage
    return source_sample_metrics, dataset_coverage


def sample_receipts(pct):
    capacities = {source: {"ref": source, "version": 1, "species": species,
                          "photos": 21, "archive_index_sha256": source}
                  for source, species in (("dogs-world", "dog"), ("lcw", "cat"))}
    sources = [{**reference, "extraction_pending": False,
                "file_sha256": {(f"images/{i}.jpg" if source == "dogs-world" else f"CatFullDataset/1/{i}.jpg"): f"{i:064x}"
                                for i in range(21 * pct // 100)}}
               for source, reference in capacities.items()]
    return capacities, sources


@pytest.mark.parametrize("pct", [20, 50, 100])
def test_sample_quota_is_per_original_source_and_not_train_fraction(pct):
    sample, coverage = coverage_functions()
    capacities, sources = sample_receipts(pct)
    sources[0]["file_sha256"]["metadata/extra.json"] = "a" * 64
    metrics = sample(sources, capacities, pct)
    assert metrics["dataset.sample_requested_pct"] == pct
    assert metrics["dataset.sampled_photos"] == 2 * (21 * pct // 100)
    rows = [{"source": source, "species": reference["species"], "animal": f"{source}-{split}", "split": split}
            for source, reference in capacities.items() for split in ("train", "val", "test")]
    train, _ = coverage(rows, sources, capacities)
    assert train["dataset.train_photos"] == train["dataset.train_identities"] == 2
    assert train["dataset.train_pct"] == 100 * 2 / 42
    assert train["dataset.train_val_photos"] == 4
    assert metrics["dataset.sampled_pct"] != train["dataset.train_pct"]
    # A matching total cannot hide a shortfall in one source.
    sources[0]["file_sha256"].pop("images/0.jpg")
    sources[1]["file_sha256"]["CatFullDataset/1/extra.jpg"] = "f" * 64
    with pytest.raises(ValueError, match="requires"):
        sample(sources, capacities, pct)


def test_sample_rejects_pilot_unverified_receipts_and_bad_hashes():
    sample, _ = coverage_functions()
    capacities, sources = sample_receipts(20)
    with pytest.raises(ValueError, match="requires"):
        sample(sources, capacities, 100)
    sources[0]["extraction_pending"] = True
    with pytest.raises(ValueError, match="completed acquisition"):
        sample(sources, capacities, 20)
    sources[0]["extraction_pending"] = False
    sources[0]["file_sha256"]["images/0.jpg"] = "unverified"
    with pytest.raises(ValueError, match="hash"):
        sample(sources, capacities, 20)
    with pytest.raises(ValueError, match="one receipt"):
        sample(sources + [sources[0]], capacities, 20)


def test_three_variants_have_identical_science_and_distinct_report_paths():
    notebooks = []
    for pct in (20, 50, 100):
        config = ast.parse(cell("e3-05", pct))
        assignment = next(n for n in config.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "SOURCE_SAMPLE_PCT" for t in n.targets))
        assert ast.literal_eval(assignment.value) == pct
        assert 'f"e3_{SOURCE_SAMPLE_PCT}_{DATASET}"' in cell("e3-05", pct)
        assert 'REGISTER_E3 = True' in cell("e3-05", pct)
        notebook = json.loads((ROOT / f"notebooks/e3_mlp_head_triplet_{pct}.ipynb").read_text())
        notebooks.append(notebook)
        if pct != 20:
            assert all(not c.get("outputs") and c.get("execution_count") is None for c in notebook["cells"])
    assert len(list((ROOT / "notebooks").glob("e3_mlp_head_triplet*.ipynb"))) == 3
    assert cell("e3-17", 20) == cell("e3-17", 50) == cell("e3-17", 100)
    assert cell("e3-28", 20) == cell("e3-28", 50) == cell("e3-28", 100)


def test_expanded_training_rejects_test_leakage_and_changed_validation():
    # Exercise guards before any public/private evaluation or optimization.
    tree = ast.parse(cell("e3-public-data"))
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "public_test_rows" for t in n.targets))
    end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "public_ds" for t in n.targets))
    code = compile(ast.Module(body=tree.body[start:end], type_ignores=[]), "held-out", "exec")
    val = {"animal": "v", "species": "cat", "sha256": "v", "split": "val"}
    test = {"animal": "t", "species": "cat", "sha256": "t", "split": "test"}
    train = {"animal": "a", "species": "cat", "sha256": "a", "split": "train"}
    namespace = {"public_prepared": SimpleNamespace(rows=[val, test]), "manifest_rows": [train, val]}
    exec(code, namespace)
    for changed in ({**train, "animal": "t"}, {**train, "sha256": "t"}):
        namespace["manifest_rows"] = [changed, val]
        with pytest.raises(ValueError, match="overlaps"):
            exec(code, namespace)
    namespace["manifest_rows"] = [train, {**val, "sha256": "new"}]
    with pytest.raises(ValueError, match="validation"):
        exec(code, namespace)


def test_both_domains_receive_training_volume_and_final_receipts(tmp_path, monkeypatch):
    import mlflow

    @dataclass
    class Config:
        seed: int = 42

    batches, tags, artifacts = {}, {}, {}

    def log_batch(run_id, metrics, params):
        batches[run_id] = ({m.key: m.value for m in metrics}, {p.key: p.value for p in params})

    client = SimpleNamespace(log_batch=log_batch,
        set_tag=lambda run, key, value: tags.update({(run, key): value}),
        get_run=lambda run: SimpleNamespace(info=SimpleNamespace(status="FINISHED")),
        log_artifact=lambda run, path, **kwargs: artifacts.update({run: json.loads(Path(path).read_text())}))
    monkeypatch.setattr(mlflow, "MlflowClient", lambda: client)
    public = tmp_path / "public_test"
    public.mkdir()
    (public / "publication.json").write_text(json.dumps({"published": True, "run_id": "public"}))
    source = cell("e3-28").split("# Dataset coverage is published with every E3 run.")[1]
    source = source.split("if REGISTER_E3:")[0]
    exec(source, {"mlflow": mlflow, "json": json, "asdict": asdict, "config": Config(),
        "train_summary": {"history": [{"epoch": 1, "train.loss": .1, "train.active_triplets": .2, "val.macro_species_mAP": .3, "training.optimizer_steps": 1, "training.examples_seen": 32, "training.unique_photos_seen": 32}], "best_epoch": 1, "best_val_mAP": .3},
        "prepared": SimpleNamespace(fingerprint="prepared"), "training_source": {"commit": "pinned"},
        "coverage_metrics": {"dataset.sample_requested_pct": 20, "dataset.sampled_pct": 19.999,
                             "dataset.train_pct": 12, "dataset.train_photos": 100, "dataset.train_identities": 10},
        "coverage_params": {"dataset.reference.scope": "original_archives_before_filtering"},
        "SOURCE_SAMPLE_PCT": 20, "REPORT": tmp_path, "PUBLIC_REPORT": public,
        "receipt": tmp_path / "publication.json", "run_id": "private", "public_run_id": "public"})
    assert batches["public"] == batches["private"]
    assert batches["public"][0]["dataset.train_photos"] == 100
    assert batches["public"][0]["dataset.sample_requested_pct"] == 20
    assert tags["public", "source_sample_pct"] == tags["private", "source_sample_pct"] == "20"
    assert artifacts["public"]["published"] and artifacts["private"]["published"]
    assert artifacts["private"]["public_run_id"] == "public"
