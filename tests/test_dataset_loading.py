import json
from pathlib import Path

import pytest

from pettrace import data
from pettrace.data import versions
from pettrace.data.checkpoints import DriveBackup
from pettrace.data.preparation import prepare
from test_e3_prepare import setup_pipeline
from test_training_data import install_hub, publish_fixture


@pytest.fixture
def refined(tmp_path, monkeypatch):
    dataset = prepare(setup_pipeline(tmp_path, monkeypatch))
    monkeypatch.setattr("pettrace.data.preparation.prepare", lambda *a, **k: pytest.fail("Loading cannot prepare data"))
    monkeypatch.setattr("pettrace.data.preparation.Detector", lambda *a, **k: pytest.fail("Loading cannot run YOLO"))
    monkeypatch.setattr("pettrace.data.sources.acquire", lambda *a, **k: pytest.fail("Loading cannot acquire raw data"))
    return dataset


def no_hub(monkeypatch):
    monkeypatch.setattr(versions, "HfApi", lambda: pytest.fail("No Hub request expected"))
    monkeypatch.setattr(versions, "snapshot_download", lambda *a, **k: pytest.fail("No download expected"))


def backup_refined(tmp_path, refined):
    versions.record_version(refined.root, refined.root.parent, version="v2.50")
    raw, reference = tmp_path / "raw", tmp_path / "reference"
    raw.mkdir()
    reference.mkdir()
    (raw / "unrequested.zip").write_bytes(b"raw must stay on Drive")
    (reference / "unrequested.onnx").write_bytes(b"models must stay on Drive")
    (refined.root.parent / "another-version.txt").write_text("do not restore")
    drive = tmp_path / "drive"
    DriveBackup({"prepared": refined.root.parent, "data": raw, "reference": reference},
                drive / "checkpoints/preparation")(force=True)
    return drive


def test_local_load_is_offline_and_rejects_corruption(refined, monkeypatch, tmp_path):
    versions.record_version(refined.root, refined.root.parent, version="v2.50")
    no_hub(monkeypatch)
    monkeypatch.setattr(DriveBackup, "restore", lambda *a, **k: pytest.fail("No Drive scan expected"))
    kwargs = dict(cache_dir=refined.root.parent, drive_root=tmp_path / "not-mounted")
    for _ in range(2):
        loaded, provenance = data.load("v2.50", **kwargs)
        assert loaded.fingerprint == refined.fingerprint
        assert provenance["version"] == "v2.50"
    Path(loaded.rows[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        data.load("v2.50", **kwargs)


def test_drive_restores_only_requested_snapshot_and_resumes_missing_file(refined, tmp_path, monkeypatch):
    drive = backup_refined(tmp_path, refined)
    no_hub(monkeypatch)
    cache = tmp_path / "new-runtime" / "refined"
    loaded, provenance = data.load("v2.50", cache_dir=cache, drive_root=drive)
    assert loaded.fingerprint == refined.fingerprint
    assert provenance["repo"] == versions.REPO
    assert set(p.name for p in cache.iterdir()) == {refined.root.name, "v2.50.json"}
    assert not (cache.parent / "raw").exists() and not (cache.parent / "hub-training").exists()
    photo = Path(loaded.rows[0]["path"])
    before = photo.read_bytes()
    photo.unlink()
    restored, _ = data.load("v2.50", cache_dir=cache, drive_root=drive)
    assert restored.fingerprint == refined.fingerprint and photo.read_bytes() == before


def test_broken_drive_backup_is_not_treated_as_missing(refined, tmp_path, monkeypatch):
    drive = backup_refined(tmp_path, refined)
    no_hub(monkeypatch)
    next((drive / "checkpoints/preparation").glob("*.tar")).unlink()
    with pytest.raises(FileNotFoundError):
        data.load("v2.50", cache_dir=tmp_path / "cache", drive_root=drive)


@pytest.mark.parametrize("drive_available", [False, True])
def test_hub_fallback_is_pinned_and_next_load_is_offline(refined, tmp_path, monkeypatch, drive_available):
    hub = install_hub(tmp_path, monkeypatch)
    publish_fixture(refined, hub)
    cache, drive = tmp_path / "cache", tmp_path / "drive"
    drive.mkdir()
    loaded, provenance = data.load("v1", cache_dir=cache, drive_root=drive if drive_available else None)
    assert loaded.fingerprint == refined.fingerprint and provenance["commit"] == "c" * 40
    assert len(hub.downloads) == 1
    pointer = (cache / "v1.json").read_bytes()
    hub.tags["v1"] = "d" * 40
    no_hub(monkeypatch)
    again, source = data.load("v1", cache_dir=cache)
    assert source == provenance and again.root == loaded.root
    assert (cache / "v1.json").read_bytes() == pointer


def test_missing_hub_photo_is_downloaded_from_original_commit(refined, tmp_path, monkeypatch):
    hub = install_hub(tmp_path, monkeypatch)
    publish_fixture(refined, hub)
    cache = tmp_path / "cache"
    loaded, provenance = data.load("v1", cache_dir=cache)
    photo = Path(loaded.rows[0]["path"])
    original = photo.read_bytes()
    photo.unlink()
    hub.tags["v1"] = "d" * 40
    restored, source = data.load("v1", cache_dir=cache)
    assert source == provenance and restored.root == loaded.root
    assert photo.read_bytes() == original
    assert len(hub.downloads) == 2 and hub.downloads[-1]["revision"] == "c" * 40


@pytest.mark.parametrize("change", [{"repo": "someone/else"}, {"version": "v2.20"}, {"path": "../outside"}])
def test_wrong_local_reference_cannot_trigger_fallback(refined, tmp_path, monkeypatch, change):
    alias = versions.record_version(refined.root, refined.root.parent, version="v2.50")
    alias.write_text(json.dumps({**json.loads(alias.read_text()), **change}))
    no_hub(monkeypatch)
    with pytest.raises(ValueError):
        data.load("v2.50", cache_dir=refined.root.parent, drive_root=tmp_path / "drive")


def test_invalid_version_and_unmounted_drive_fail_before_network(tmp_path, monkeypatch):
    no_hub(monkeypatch)
    with pytest.raises(ValueError, match="version"):
        data.load("../v1", cache_dir=tmp_path / "cache")
    with pytest.raises(FileNotFoundError, match="Mount Drive"):
        data.load("v2.50", cache_dir=tmp_path / "cache", drive_root=tmp_path / "missing")
    assert not (tmp_path / "cache").exists()


def test_hub_error_does_not_record_a_completed_version(tmp_path, monkeypatch):
    def denied():
        raise PermissionError("Hub access denied")

    monkeypatch.setattr(versions, "HfApi", denied)
    cache = tmp_path / "cache"
    with pytest.raises(PermissionError, match="access denied"):
        data.load("v2.50", cache_dir=cache)
    assert not (cache / "v2.50.json").exists()


def test_hub_must_resolve_the_requested_immutable_commit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(versions, "HfApi", lambda: SimpleNamespace(
        dataset_info=lambda *a, **k: SimpleNamespace(sha="d" * 40)))
    monkeypatch.setattr(versions, "snapshot_download", lambda *a, **k: pytest.fail("Wrong commit cannot download"))
    with pytest.raises(ValueError, match="different commit"):
        versions.load_training(version="v1", revision="c" * 40, cache_dir=tmp_path)
