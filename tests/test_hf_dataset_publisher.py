import csv
import importlib.util
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from pettrace.data.preparation import prepare
from pettrace.data.versions import record_version
from test_e3_prepare import setup_pipeline

SCRIPT = Path(__file__).parents[1] / "tools/publish_hf_dataset.py"
SPEC = importlib.util.spec_from_file_location("publish_hf_dataset", SCRIPT)
publish_hf_dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish_hf_dataset)


def test_lcw_v4_100_builds_public_v1_shards_without_dogs(tmp_path, monkeypatch):
    dataset = prepare(setup_pipeline(tmp_path, monkeypatch))
    reference = record_version(dataset.root, dataset.root.parent, version="v4.100")
    staging = tmp_path / "lcw-refined"
    monkeypatch.setattr(publish_hf_dataset, "_source_receipt",
                        lambda *args: {"ref": "dseidli/lcwlabeled-cats-in-the-wild",
                                      "license": "Apache 2.0 (publisher)"})

    result = publish_hf_dataset.build_release(
        reference, staging, shard_size=1024**2, license_text=b"Apache License fixture")
    assert result == staging.resolve()
    evidence = json.loads((staging / "versions/v1/dataset.json").read_text())
    assert evidence["source"] == "lcw"
    assert evidence["prepared_version"] == "v4.100"
    assert evidence["release_version"] == "v1"
    assert evidence["license"] == "apache-2.0"
    assert evidence["prepared_fingerprint"] == dataset.fingerprint
    assert "file_sha256" not in evidence["source_receipt"]
    rows = list(csv.DictReader((staging / "versions/v1/metadata.csv").open()))
    assert rows and {row["source"] for row in rows} == {"lcw"}
    assert len(rows) == evidence["photos"]
    with tarfile.open(staging / evidence["shards"][0]["path"]) as archive:
        names = archive.getnames()
    assert any(name.endswith(".json") for name in names)
    assert all("dogs-world" not in name for name in names)
    card = (staging / "README.md").read_text()
    assert "license: apache-2.0" in card and "pettrace-training@v4.100" in card
    assert publish_hf_dataset.build_release(
        reference, staging, shard_size=1024**2, license_text=b"unused") == staging.resolve()


def test_source_receipt_requires_publisher_license(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    receipt = {"sources": [{"ref": "dseidli/lcwlabeled-cats-in-the-wild",
                             "license": "Apache 2.0 (publisher)", "file_sha256": {"cat.jpg": "0" * 64}}]}
    (root / "sources.json").write_text(json.dumps(receipt))
    result = publish_hf_dataset._source_receipt(SimpleNamespace(root=root), publish_hf_dataset.SOURCES["lcw"])
    assert result["license"] == "Apache 2.0 (publisher)" and "file_sha256" not in result


def test_transition_notebook_pins_tool_and_separates_review_from_upload():
    root = Path(__file__).parents[1]
    notebook = json.loads((root / "datasets/lcw_refined_transition.ipynb").read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                           if cell["cell_type"] == "markdown")
    assert markdown.count("## Fase ") == 3
    assert "MLFLOW" not in "\n".join(cells.values())
    digest = publish_hf_dataset._sha256(root / "tools/publish_hf_dataset.py")
    assert f'TOOL_SHA256 = "{digest}"' in cells["lcw-transition-build"]
    assert "--upload" not in cells["lcw-transition-build"]
    assert "--upload" in cells["lcw-transition-publish"]


class PublicHub:
    def __init__(self, root):
        self.root = root
        self.tags = {}

    def create_repo(self, *args, **kwargs):
        assert kwargs == {"repo_type": "dataset", "private": False, "exist_ok": True}

    def dataset_info(self, repo_id):
        return SimpleNamespace(private=False)

    def list_repo_refs(self, *args, **kwargs):
        return SimpleNamespace(tags=[SimpleNamespace(name=name, target_commit=commit)
                                     for name, commit in self.tags.items()])

    def upload_folder(self, **kwargs):
        shutil.copytree(kwargs["folder_path"], self.root, dirs_exist_ok=True)
        return SimpleNamespace(oid="c" * 40)

    def list_repo_files(self, *args, **kwargs):
        return [path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()]

    def create_tag(self, repo_id, **kwargs):
        self.tags[kwargs["tag"]] = kwargs["revision"]


def test_publication_tags_only_after_complete_upload(tmp_path, monkeypatch):
    dataset = prepare(setup_pipeline(tmp_path, monkeypatch))
    reference = record_version(dataset.root, dataset.root.parent, version="v4.100")
    monkeypatch.setattr(publish_hf_dataset, "_source_receipt",
                        lambda *args: {"ref": "dseidli/lcwlabeled-cats-in-the-wild",
                                      "license": "Apache 2.0 (publisher)"})
    staging = publish_hf_dataset.build_release(
        reference, tmp_path / "staging", shard_size=1024**2, license_text=b"Apache License fixture")
    hub = PublicHub(tmp_path / "hub")
    monkeypatch.setattr(publish_hf_dataset, "HfApi", lambda token=None: hub)
    secret = staging / ".env"
    secret.write_text("must not upload")
    with pytest.raises(ValueError, match="unexpected"):
        publish_hf_dataset.publish_release(staging)
    secret.unlink()
    receipt = publish_hf_dataset.publish_release(staging)
    assert receipt == {"repo": "alessandro54/lcw-refined", "version": "v1", "commit": "c" * 40,
                       "prepared_fingerprint": dataset.fingerprint}
    assert hub.tags == {"v1": "c" * 40}
