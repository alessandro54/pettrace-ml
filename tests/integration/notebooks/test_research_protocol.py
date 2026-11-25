"""Verify the notebooks' research protocol without remote access or retraining."""
import ast
import json
import re
from types import SimpleNamespace

import pytest

from tests import ROOT

NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
PRIVATE_COMMITS = {
    "dino_lora_test.ipynb": "52bfe26384c4230008fe4707aa0cb145988766a8",
    "dino_mlp_width_ablation.ipynb": "52bfe26384c4230008fe4707aa0cb145988766a8",
    "e5_luna_hard_case.ipynb": "52bfe26384c4230008fe4707aa0cb145988766a8",
    "dino_sequential_pipeline.ipynb": "52bfe26384c4230008fe4707aa0cb145988766a8",
    "dino_projection_test.ipynb": "52bfe26384c4230008fe4707aa0cb145988766a8",
    "dino_pipeline.ipynb": "52bfe26384c4230008fe4707aa0cb145988766a8",
    "c1_dinov2_reference.ipynb": "52bfe26384c4230008fe4707aa0cb145988766a8",
    "e4_clip_lora.ipynb": "52bfe26384c4230008fe4707aa0cb145988766a8",
    "e4_sampling_multiseed.ipynb": "52bfe26384c4230008fe4707aa0cb145988766a8",
    "e4_component_ablation.ipynb": "52bfe26384c4230008fe4707aa0cb145988766a8",
    "e5_language_reranking.ipynb": "52bfe26384c4230008fe4707aa0cb145988766a8",
}
DEFAULT_PRIVATE_COMMIT = "1777734dbffda0df0bf43305233809ee73157d94"


def cells(path):
    return json.loads(path.read_text())["cells"]


def python_source(cell):
    """Return executable Python while ignoring supported IPython magic lines."""
    return "".join(line for line in cell["source"] if not line.lstrip().startswith("%"))


@pytest.mark.parametrize("path", [p for p in NOTEBOOKS if not p.name.startswith("e1_")], ids=lambda p: p.stem)
@pytest.mark.parametrize("version", [(3, 10), (3, 11), (3, 12), (3, 13), (3, 14)])
def test_notebook_python_support_matches_dependency_profile(path, version):
    source = next("".join(c["source"]) for c in cells(path)
                  if c["cell_type"] == "code" and "sys.version_info" in "".join(c["source"]))
    source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
    guard = next(n for n in ast.parse(source).body
                 if isinstance(n, ast.If) and "version_info" in ast.unparse(n.test))
    code = compile(ast.Module(body=[guard], type_ignores=[]), str(path), "exec")
    namespace = {"sys": SimpleNamespace(version_info=version, version=".".join(map(str, version)))}
    supported = (3, 11) <= version < (3, 14)
    namespace["sys"].platform = "linux"
    if supported:
        exec(code, namespace)
    else:
        with pytest.raises(RuntimeError, match="Python"):
            exec(code, namespace)


def test_current_e3_notebook_installs_the_published_package_directly():
    contents = cells(ROOT / "notebooks/e3_mlp_head_triplet.ipynb")
    install = next("".join(c["source"]) for c in contents if c["id"] == "e3-setup")
    notebook = "\n".join("".join(c["source"]) for c in contents)
    assert '"pettrace-ml[training,notebooks,tracking]==0.2.37"' in install
    assert "requirements-e3-runtime" not in notebook
    assert "e3-runtime.txt" not in notebook
    assert "drive_root" not in notebook
    assert "SOURCE_SAMPLE_PCT, SEEDS = 100, (45, 46, 47, 48, 49)" in notebook


def test_e1_notebook_installs_the_complete_headless_toolkit():
    contents = cells(ROOT / "notebooks/e1_clip_zero_shot.ipynb")
    install = next("".join(c["source"]) for c in contents if c["id"].endswith("setup"))
    notebook = "\n".join("".join(c["source"]) for c in contents)
    assert '"pettrace-ml[notebooks,tracking]==0.2.33"' in install
    assert "import mlflow" in install
    assert "requirements-e1-runtime" not in notebook
    assert "publication = run(REPORT)" in notebook
    assert "snapshot_download" not in notebook
    assert "publish_results" not in notebook


def test_e2_notebook_installs_the_published_package_directly():
    contents = cells(ROOT / "notebooks/e2_yolo_crop_zero_shot.ipynb")
    install = next("".join(c["source"]) for c in contents if c["id"].endswith("setup"))
    notebook = "\n".join("".join(c["source"]) for c in contents)
    assert '"pettrace-ml[notebooks,tracking]==0.2.35"' in install
    assert "publication = run(REPORT)" in notebook
    assert "snapshot_download" not in notebook
    assert "publish_results" not in notebook


def test_e4_installs_complete_published_training_package():
    contents = cells(ROOT / "notebooks/e4_clip_lora.ipynb")
    install = next("".join(c["source"]) for c in contents if c["id"] == "e4-setup")
    notebook = "\n".join("".join(c["source"]) for c in contents)
    assert '"pettrace-ml[training,notebooks,tracking]==0.2.39"' in install
    assert "requirements" not in install
    assert "LOG_TO_MLFLOW = True" in notebook
    assert "training(version=TRAINING_DATASET_VERSION" in notebook
    assert "drive_root" not in notebook
    assert 'TRAINING_DATASET_VERSION, SOURCE_SAMPLE_PCT, SEED = "v2", 100, 46' in notebook


def test_e4_sampling_selects_family_mean_before_best_seed():
    import numpy as np
    from pettrace.experiments.e3 import select_validation_candidate

    contents = cells(ROOT / "notebooks/e4_sampling_multiseed.ipynb")
    train = next("".join(c["source"]) for c in contents if c["id"] == "e4-study-train")
    assert train.index("seed_all(seed)") < train.index("build_encoder(")
    candidates = [{"p": p, "seed": seed, "best_val_mAP": score}
                  for p, scores in ((8, (0.8, 0.81, 0.82)), (16, (0.7, 0.7, 0.99)))
                  for seed, score in zip((45, 46, 47), scores)]
    assignments = [node for node in ast.parse(train).body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in {"family_scores", "selected_p", "selected"}
                           for t in node.targets)]
    namespace = {"candidates": candidates, "BATCHES": ((8, 4), (16, 2)), "np": np,
                 "select_validation_candidate": select_validation_candidate}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "selection", "exec"), namespace)
    assert namespace["selected_p"] == 8
    assert namespace["selected"]["seed"] == 47


@pytest.mark.parametrize("arm", ["lora_only", "projection_only"])
def test_e4_component_ablation_trains_only_the_declared_component(arm):
    torch = pytest.importorskip("torch")
    from torch import nn
    from pettrace.training.lora import LoRALinear

    train = next("".join(c["source"]) for c in cells(ROOT / "notebooks/e4_component_ablation.ipynb")
                 if c["id"] == "e4-components-train")
    freeze = next(node for node in ast.walk(ast.parse(train))
                  if isinstance(node, ast.If) and "requires_grad_" in ast.unparse(node))
    torch.manual_seed(46)
    encoder = nn.Module()
    encoder.vision_model = LoRALinear(nn.Linear(4, 4), rank=2)
    encoder.visual_projection = nn.Linear(4, 4, bias=False)
    encoder.head = nn.Linear(4, 2).requires_grad_(False)
    exec(compile(ast.Module(body=[freeze], type_ignores=[]), "component freeze", "exec"),
         {"encoder": encoder, "arm": arm})
    expected = ({"vision_model.a", "vision_model.b"} if arm == "lora_only"
                else {"visual_projection.weight"})
    assert {name for name, p in encoder.named_parameters() if p.requires_grad} == expected
    output = encoder.head(encoder.visual_projection(encoder.vision_model(torch.randn(3, 4))))
    output.square().sum().backward()
    assert {name for name, p in encoder.named_parameters() if p.grad is not None} == expected
    assert any(p.grad.abs().sum() > 0 for p in encoder.parameters() if p.grad is not None)


def test_e5_installs_the_vlm_profile_and_freezes_before_test():
    contents = cells(ROOT / "notebooks/e5_language_reranking.ipynb")
    install = next("".join(c["source"]) for c in contents if c["id"] == "e5-setup")
    calibration_index = next(i for i, c in enumerate(contents) if c["id"] == "e5-calibrate")
    evaluation_index = next(i for i, c in enumerate(contents) if c["id"] == "e5-evaluate")
    calibration = "".join(contents[calibration_index]["source"])
    evaluation = "".join(contents[evaluation_index]["source"])
    notebook = "\n".join("".join(c["source"]) for c in contents)

    assert '"pettrace-ml[vlm,notebooks,tracking]==0.2.46"' in install
    assert calibration_index < evaluation_index
    assert "select_calibration(" in calibration
    assert 'public_split_dataset(prepared, public_source, "val")' in notebook
    assert "load_benchmarks(" in notebook
    assert "evaluate(benchmarks.public" in evaluation
    assert "PetTrace v6" in notebook
    assert "publication = publish_results(" in notebook


def test_dinov2_reference_uses_normalized_cls_and_preserves_batch_order():
    torch = pytest.importorskip("torch")
    import numpy as np
    from torch import nn
    from PIL import Image

    source = next("".join(c["source"]) for c in cells(ROOT / "notebooks/c1_dinov2_reference.ipynb")
                  if "class ReferencePipeline" in "".join(c["source"]))
    definition = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    namespace = {"torch":torch,"np":np}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "reference pipeline", "exec"), namespace)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        def forward(self, pixel_values):
            cls = torch.stack((pixel_values, torch.ones_like(pixel_values)),dim=-1)
            return SimpleNamespace(last_hidden_state=torch.stack((cls, torch.zeros_like(cls)),dim=1))

    batches = []
    def process(images, return_tensors):
        batches.append(len(images))
        return {"pixel_values":torch.tensor([im.getpixel((0,0))[0] for im in images],dtype=torch.float32)}
    pipeline = namespace["ReferencePipeline"](Backbone(),process,SimpleNamespace(crop=lambda im:im))
    actual = pipeline.embed([Image.new("RGB",(1,1),(i,0,0)) for i in range(11)])
    expected = np.stack((np.arange(11),np.ones(11)),axis=1)
    expected /= np.linalg.norm(expected,axis=1,keepdims=True)
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
    assert batches == [8,3]


def test_e5_uses_colab_secrets_resumable_cache_and_fixed_vlm():
    contents = cells(ROOT / "notebooks/e5_language_reranking.ipynb")
    config = next("".join(c["source"]) for c in contents if c["id"] == "e5-config")
    calibration = next("".join(c["source"]) for c in contents if c["id"] == "e5-calibrate")

    assert "from google.colab import drive, userdata" in config
    assert "userdata.get(name)" in config
    assert 'MLFLOW_TRACKING_URI' in config
    assert "/content/drive/MyDrive/PetTrace/checkpoints/e5/attributes-v3.parquet" in config
    assert "attribute_model = MODEL_2B" in calibration
    assert "MODEL_4B" not in calibration
    assert calibration.index("QwenAttributeExtractor(") > calibration.index("else:")
    assert "expected_identity=expected_identity" in calibration


@pytest.mark.parametrize(
    "path", [path for path in NOTEBOOKS if path.name != "e5_language_reranking.ipynb"],
    ids=lambda p: p.stem,
)
def test_experiment_notebooks_do_not_use_google_drive(path):
    source = "\n".join("".join(cell["source"]) for cell in cells(path))
    assert "DriveBackup" not in source
    assert "/content/drive" not in source
    assert "google.colab import drive" not in source


def test_e4_loads_colab_secrets_before_dataset_access():
    contents = cells(ROOT / "notebooks/e4_clip_lora.ipynb")
    config = next("".join(c["source"]) for c in contents if c["id"] == "e4-config")
    load_index = next(i for i, c in enumerate(contents) if c["id"] == "e4-load-data")
    config_index = next(i for i, c in enumerate(contents) if c["id"] == "e4-config")
    assert config_index < load_index
    assert "from google.colab import userdata" in config
    assert 'userdata.get(name)' in config
    assert "isinstance(value, str)" in config
    assert "configure_access" not in config


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_shared_research_structure_and_fixed_benchmark(path):
    contents = cells(path)
    headings = [line for c in contents if c["cell_type"] == "markdown"
                for line in "".join(c["source"]).splitlines() if line.startswith("## ")]
    assert [int(re.match(r"## Fase (\d+)\.", h).group(1)) for h in headings] == [1, 2, 3]
    if path.name in {"d1_dinov2_pilot.ipynb", "dino_pipeline_smoke.ipynb"}:
        source = "\n".join(python_source(c) for c in contents if c["cell_type"] == "code")
        assert "test_evaluated=False" in source
        assert re.search(r"LOG_TO_MLFLOW\s*=\s*True", source)
        assert "client.set_terminated(run_id,'FINISHED')" in source
        assert "client.log_artifacts(run_id,str(folder)" in source
        assert "DATASET_COMMIT" not in source  # This pilot has no private evaluation.
        return
    if path.name in {"distractor_sensitivity.ipynb", "external_distractors.ipynb"}:
        # These analyses reuse recorded inputs rather than downloading a dataset tag.
        source = "\n".join(python_source(c) for c in contents if c["cell_type"] == "code")
        assert "277" in source and "502" in source
        assert "sha256" in source
        assert "log_artifact(" in source
        assert "FINISHED" in source
        if path.name == "external_distractors.ipynb":
            assert "089695c834a7deb60505b7cc506672db1c31a6aa" in source
            assert "assert sha256(ROOT / r['path']) == r['sha256']" in source
        else:
            assert "assert signature==expected" in source
        return
    assignments = [node for c in contents if c["cell_type"] == "code"
                   and "DATASET_COMMIT =" in "".join(c["source"])
                   for node in ast.parse("".join(c["source"])).body if isinstance(node, ast.Assign)]
    commit = next(n for n in assignments if any(isinstance(t, ast.Name) and t.id == "DATASET_COMMIT"
                                                for t in n.targets))
    expected_commit = PRIVATE_COMMITS.get(path.name, DEFAULT_PRIVATE_COMMIT)
    assert ast.literal_eval(commit.value) == expected_commit
    publication = "\n".join("".join(c["source"]) for c in contents if c["cell_type"] == "code")
    if path.name in {"e1_clip_zero_shot.ipynb", "e2_yolo_crop_zero_shot.ipynb"}:
        assert "publication = run(REPORT)" in publication
        experiment = path.stem.split("_")[0]
        helper = (ROOT / f"src/pettrace/experiments/{experiment}.py").read_text()
        assert helper.index("_dataset_selection(") < helper.index("publication = publish_results(")
    elif path.name.startswith(("e3_", "e4_")):
        assert "publication = publish_results(" in publication
        helper = (ROOT / "src/pettrace/experiments/e3.py").read_text()
        assert helper.index('"experiment-design.json"') < helper.index("public_run_id = publish_evaluation(")
    elif path.name == "e5_luna_hard_case.ipynb":
        assert 'client.log_artifact(run_id' in publication
        assert 'client.set_terminated(run_id, status="FINISHED")' in publication
    elif path.name in {"dino_lora_test.ipynb", "dino_mlp_width_ablation.ipynb"}:
        assert 'run=publish_evaluation(' in publication
        assert 'save_publication_input(' in publication
        assert 'load_publication_input(' in publication
        assert "client.get_run(run).info.status=='FINISHED'" in publication
        assert re.search(r'LOG_TO_MLFLOW\s*=\s*True', publication)
    elif path.name.startswith("e5_"):
        assert "publication = publish_results(" in publication
        helper = (ROOT / "src/pettrace/experiments/e5.py").read_text()
        assert helper.index('"experiment-design.json"') < helper.index("public_run_id = publish_evaluation(")
    elif path.name == "dino_sequential_pipeline.ipynb":
        assert publication.index("(REPORT/'selection.json').write_text") < publication.index('ev=evaluate(')
        assert 'dimension=1024' in publication
        assert "head=parent_head()" in publication
        assert "compare(e3_golden,pipeline.embed(golden_images()))[0]" in publication
        assert 'parent_checkpoint_sha256=parent_hash' in publication
        assert 'training_run=publish_training(' in publication
        assert 'save_publication_input(ev,target)' in publication
        assert 'load_publication_input(target)' in publication
        assert 'assert len(publications)==16' in publication
        assert re.search(r'LOG_TO_MLFLOW\s*=\s*True', publication)
    elif path.name == "dino_pipeline.ipynb":
        assert publication.index('selection=select_family(') < publication.index('ev=evaluate(')
        assert publication.index("'runtime.json'") < publication.index('run=publish_evaluation(')
        assert 'save_publication_input(ev,target)' in publication
        assert 'load_publication_input(target)' in publication
        assert 'training_run=publish_training(' in publication
        assert re.search(r'LOG_TO_MLFLOW\s*=\s*True', publication)
    elif path.name == "dino_projection_test.ipynb":
        assert publication.index('lock.write_text(') < publication.index('ev=evaluate(')
        assert publication.index("'runtime.json'") < publication.index('run=publish_evaluation(')
        assert 'save_publication_input(ev,folder)' in publication
        assert 'load_publication_input(folder)' in publication
        assert "client.get_run(run).info.status=='FINISHED'" in publication
        assert "assert len(publications)==6" in publication
        assert re.search(r'LOG_TO_MLFLOW\s*=\s*True', publication)
    elif path.name == "c1_dinov2_reference.ipynb":
        assert publication.index('"runtime.json"') < publication.index("run_id = publish_evaluation(")
        assert "save_publication_input(ev,folder)" in publication
        assert "load_publication_input(folder)" in publication
        assert 'client.get_run(run_id).info.status == "FINISHED"' in publication
    else:
        assert "publication = publish_results(" in publication
        helper = (ROOT / "src/pettrace/experiments/zero_shot.py").read_text()
        assert helper.index('"experiment-design.json"') < helper.index("public_run_id = publish_evaluation(")

def test_e3_selects_by_validation_before_test_and_publishes_every_seed():
    contents = cells(ROOT / "notebooks/e3_mlp_head_triplet.ipynb")
    train_index = next(i for i, c in enumerate(contents) if c.get("id") == "e3-train")
    evaluation_index = next(i for i, c in enumerate(contents) if c.get("id") == "e3-evaluate")
    train = "".join(contents[train_index]["source"])
    evaluation = "".join(contents[evaluation_index]["source"])
    registration = next("".join(c["source"]) for c in contents if c.get("id") == "e3-register")

    assert train_index < evaluation_index
    assert "selected = select_validation_candidate(candidates)" in train
    assert "for candidate in candidates:" in evaluation
    assert "publication = publish_results(" in evaluation
    assert 'selected["publication"]' in registration
