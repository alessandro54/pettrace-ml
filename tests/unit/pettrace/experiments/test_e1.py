"""Headless E1 configuration and pinned private benchmark checks."""

from types import SimpleNamespace

import pytest

from pettrace.experiments import e1


def test_default_private_benchmark_is_pinned_v5():
    assert e1.PRIVATE_VERSION == "v5"
    assert e1.PRIVATE_COMMIT == "1777734dbffda0df0bf43305233809ee73157d94"


def test_environment_requires_noninteractive_credentials_and_production(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://other.example")
    with pytest.raises(ValueError, match="must publish"):
        e1.require_environment()

    monkeypatch.setenv("MLFLOW_TRACKING_URI", e1.PRODUCTION_URI)
    for name in ("HF_TOKEN", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN.*MLFLOW_TRACKING_USERNAME.*MLFLOW_TRACKING_PASSWORD"):
        e1.require_environment()


def test_private_dataset_requires_tag_to_match_commit(tmp_path, monkeypatch):
    class Api:
        def __init__(self, token):
            assert token == "token"

        def dataset_info(self, repo, revision):
            return SimpleNamespace(sha="a" * 40)

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    with pytest.raises(ValueError, match="does not resolve"):
        e1._private_dataset("v5", "b" * 40, tmp_path)
