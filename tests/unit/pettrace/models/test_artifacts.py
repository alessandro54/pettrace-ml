from pathlib import Path
from types import SimpleNamespace

import pytest

from pettrace.models import artifacts


def _release(root: Path):
    root.mkdir(parents=True)
    (root / "pipeline.yaml").write_text("steps: []\n")
    (root / "golden.npy").write_bytes(b"golden")
    (root / "encoder.onnx").write_bytes(b"onnx")


def test_download_model_resolves_revision_to_commit(tmp_path, monkeypatch):
    commit = "a" * 40
    release = tmp_path / "snapshot" / "e4"
    _release(release)
    calls = []

    class Api:
        def __init__(self, token=None):
            assert token == "secret"

        def model_info(self, repo_id, revision):
            assert (repo_id, revision) == ("owner/models", "v1")
            return SimpleNamespace(sha=commit)

    def download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return release.parent

    monkeypatch.setattr(artifacts, "HfApi", Api)
    monkeypatch.setattr(artifacts, "snapshot_download", download)

    result = artifacts.download_model(
        "owner/models", revision="v1", subfolder="e4", cache_dir=tmp_path / "cache", token="secret")

    assert result == release.resolve()
    assert calls == [("owner/models", {
        "repo_type": "model", "revision": commit, "token": "secret",
        "allow_patterns": ["e4/**"], "cache_dir": str(tmp_path / "cache"),
    })]


@pytest.mark.parametrize("subfolder", ["../e4", "/tmp/e4"])
def test_download_model_rejects_unsafe_subfolder(subfolder):
    with pytest.raises(ValueError, match="Unsafe"):
        artifacts.download_model("owner/models", revision="v1", subfolder=subfolder)


def test_download_model_requires_complete_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "HfApi", lambda token=None: SimpleNamespace(
        model_info=lambda *args, **kwargs: SimpleNamespace(sha="b" * 40)))
    monkeypatch.setattr(artifacts, "snapshot_download", lambda *args, **kwargs: tmp_path)

    with pytest.raises(FileNotFoundError, match="missing"):
        artifacts.download_model("owner/models", revision="v1")


def test_missing_local_model_is_not_treated_as_mlflow(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        artifacts.model_dir(tmp_path / "missing")
