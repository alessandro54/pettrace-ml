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


def test_release_contains_only_distributions_and_checksums_with_build_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(build_release, "ROOT", tmp_path)
    package = tmp_path / "src/pettrace"
    package.mkdir(parents=True)
    for name, contents in {"pyproject.toml": '[project]\nversion = "0.2.0"\n', "uv.lock": "locked",
                           ".python-version": "3.11.15\n"}.items():
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
        else:
            raise AssertionError(args)

    monkeypatch.setattr(subprocess, "run", execute)
    build_release.main()
    output = tmp_path / "dist/v0.2.0"
    assert not (package / "_build.json").exists()
    assert captured[0]["git_sha"] == "a" * 40
    assert captured[0]["uv_lock_sha256"] == hashlib.sha256(b"locked").hexdigest()
    assert captured[0]["version"] == "0.2.0"
    assert captured[0]["reference_python"] == "3.11.15"
    assert captured[0]["git_dirty"] is False
    assert {p.name for p in output.iterdir() if not p.name.startswith(".")} == {
        "pettrace_ml-0.2.0-py3-none-any.whl", "pettrace_ml-0.2.0.tar.gz", "SHA256SUMS",
    }
    assert len((output / "SHA256SUMS").read_text().splitlines()) == 2
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        build_release.main()
