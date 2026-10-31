"""Dataset downloads must use the same immutable revision as their provenance."""

from types import SimpleNamespace
from unittest.mock import Mock

from pettrace.data import benchmark as data


def test_load_resolves_tag_before_downloading_exact_commit(monkeypatch, tmp_path):
    import huggingface_hub

    commit = "a" * 40
    info = Mock(return_value=SimpleNamespace(sha=commit))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(dataset_info=info))

    def download(repo, **kwargs):
        info.assert_called_once_with("test/pets", revision="v4")
        assert repo == "test/pets"
        assert kwargs == {"repo_type": "dataset", "revision": commit, "cache_dir": str(tmp_path)}
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    loaded = object()
    from_dir = Mock(return_value=loaded)
    monkeypatch.setattr(data, "from_dir", from_dir)
    assert data.load("v4", repo="test/pets", cache_dir=str(tmp_path)) is loaded
    from_dir.assert_called_once_with(str(tmp_path / "snapshot"), repo="test/pets", version="v4", commit=commit)
