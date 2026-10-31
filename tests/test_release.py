"""Release artifacts bind an installed package to the reviewed commit and lockfile."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools import build_release


def test_release_refuses_uncommitted_code(tmp_path, monkeypatch):
    monkeypatch.setattr(build_release, "ROOT", tmp_path)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: " M changed.py\n")
    with pytest.raises(SystemExit, match="Commit"):
        build_release.main()
    assert not (tmp_path / "dist").exists()


def test_release_exports_hashed_environments_and_preserves_build_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(build_release, "ROOT", tmp_path)
    package = tmp_path / "src/pettrace"
    package.mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "colab").mkdir()
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "tools/run_e1.py").write_text("# launcher")
    (tmp_path / "colab/e1.ipynb").write_text("{}")
    original = {"cells": [{"cell_type": "code", "outputs": ["historical"], "execution_count": 5}],
                "metadata": {"widgets": {}, "pettrace": {"report": "old"}}}
    (tmp_path / "notebooks/e1_clip_zero_shot.ipynb").write_text(json.dumps(original))
    for name, contents in {"pyproject.toml": '[project]\nversion = "0.2.0"\n', "uv.lock": "locked",
                           ".python-version": "3.11.15\n", "docs/releases.md": "Install guide"}.items():
        (tmp_path / name).write_text(contents)
    monkeypatch.setattr(subprocess, "check_output", lambda args, **kw:
                        "a" * 40 if args[:2] == ("git", "rev-parse") else "")
    captured = []

    def execute(args, **kwargs):
        if args[1] == "build":
            captured.append(json.loads((package / "_build.json").read_text()))
            output = Path(args[-1])
            (output / ".gitignore").write_text("*\n")
            (output / "pettrace_ml-0.2.0-py3-none-any.whl").write_bytes(b"wheel fixture")
            (output / "pettrace_ml-0.2.0.tar.gz").write_bytes(b"source fixture")
        elif args[1] == "export":
            assert "--locked" in args and "--no-emit-project" in args and "--no-hashes" not in args
            Path(args[args.index("--output-file") + 1]).write_text("numpy==2.4.6 --hash=sha256:fixture\n")
        else:
            raise AssertionError(args)

    monkeypatch.setattr(subprocess, "run", execute)
    build_release.main()
    output = tmp_path / "dist/v0.2.0"
    assert not (package / "_build.json").exists()
    assert captured[0]["git_sha"] == "a" * 40
    assert captured[0]["uv_lock_sha256"] == hashlib.sha256(b"locked").hexdigest()
    assert (output / "python-version.txt").read_text() == "3.11.15\n"
    assert ".gitignore" not in (output / "SHA256SUMS").read_text()
    clean = json.loads((output / "e1_clip_zero_shot.ipynb").read_text())
    assert clean["cells"][0]["outputs"] == []
    assert "pettrace" not in clean["metadata"]
    assert json.loads((tmp_path / "notebooks/e1_clip_zero_shot.ipynb").read_text()) == original
    for profile in ("evaluation", "e3"):
        requirements = (output / f"requirements-{profile}.txt").read_text()
        assert "releases/download/v0.2.0/pettrace_ml-0.2.0-py3-none-any.whl" in requirements
        assert hashlib.sha256(b"wheel fixture").hexdigest() in requirements
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        build_release.main()
