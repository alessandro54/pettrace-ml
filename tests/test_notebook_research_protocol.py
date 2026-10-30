"""Verify the notebooks' research protocol without remote access or retraining."""
import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
COMMIT = "8b12db0dce41040a821f9792bea5b352342263ce"


def cells(path):
    return json.loads(path.read_text())["cells"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_shared_research_structure_and_fixed_benchmark(path):
    contents = cells(path)
    headings = [line for c in contents if c["cell_type"] == "markdown"
                for line in "".join(c["source"]).splitlines() if line.startswith("## ")]
    assert [int(re.match(r"## Fase (\d+)\.", h).group(1)) for h in headings] == [1, 2, 3]
    assignments = [node for c in contents if c["cell_type"] == "code"
                   and "DATASET_COMMIT =" in "".join(c["source"])
                   for node in ast.parse("".join(c["source"])).body if isinstance(node, ast.Assign)]
    commit = next(n for n in assignments if any(isinstance(t, ast.Name) and t.id == "DATASET_COMMIT"
                                                for t in n.targets))
    assert ast.literal_eval(commit.value) == COMMIT
    publication = next("".join(c["source"]) for c in contents
                       if c["cell_type"] == "code" and "run_id = publish_evaluation(" in "".join(c["source"]))
    assert publication.index('"experiment-design.json"') < publication.index("run_id = publish_evaluation(")


@pytest.mark.parametrize("name", ["e1_clip_zero_shot.ipynb", "e2_yolo_crop_zero_shot.ipynb"])
def test_mutated_tag_stops_before_any_download(name):
    code = next("".join(c["source"]) for c in cells(ROOT / "notebooks" / name)
                if c["cell_type"] == "code" and "resolved_commit =" in "".join(c["source"]))
    downloads = []
    namespace = {"HfApi": lambda: SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(sha="changed")),
                 "data": SimpleNamespace(REPO="test/benchmark"), "DATASET": "v4", "DATASET_COMMIT": COMMIT,
                 "snapshot_download": lambda *a, **k: downloads.append(k)}
    with pytest.raises(AssertionError, match="pinned"):
        exec(code, namespace)
    assert downloads == []


def test_e3_identity_macro_does_not_weight_by_photo_count(tmp_path):
    code = next("".join(c["source"]) for c in cells(ROOT / "notebooks/e3_mlp_head_triplet.ipynb")
                if c["id"] == "e3-identity-findings")

    def evaluation(ranks):
        rows = [{"animal": animal, "rank": rank, "ap": 1.0 / rank}
                for animal, rank in zip(["a", "a", "b"], ranks)]
        return SimpleNamespace(result={"test": {"per_query": rows}})

    namespace = {"np": np, "json": json, "REPORT": tmp_path,
                 "baseline_evaluations": {"E1": evaluation([1, 2, 1]), "E2": evaluation([2, 2, 1])},
                 "ev": evaluation([1, 1, 2]), "ds": SimpleNamespace(name="benchmark", commit=COMMIT,
                      fingerprint="digest", animals={"a": {"species": "cat"}, "b": {"species": "dog"}}),
                 "PROTOCOL": "zero-shot", "comparison": {}, "train_summary": {},
                 "prepared": SimpleNamespace(fingerprint="public"), "coverage_metrics": {},
                 "display": lambda *a: None, "Markdown": lambda s: s}
    exec(code, namespace)
    analysis = json.loads((tmp_path / "analysis.json").read_text())
    assert analysis["macro_rank1"]["E3"] == 0.5
    assert analysis["per_animal"][0]["rank1_delta_e3_e2"] == 1.0
    assert analysis["per_animal"][1]["rank1_delta_e3_e2"] == -1.0
