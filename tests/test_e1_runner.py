"""Isolation and failure recovery for the release's E1 entry point."""

import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import nbformat
import pytest
from nbclient.exceptions import CellExecutionError

from pettrace.experiments import e1
from pettrace.evaluation.research import PUBLIC_REPO, PUBLIC_COMMIT
from tools import run_e1


def notebook_cell(cell_id):
    notebook = nbformat.read(Path(__file__).parents[1] / "notebooks/e1_clip_zero_shot.ipynb", as_version=4)
    return next(c.source for c in notebook.cells if c.id == cell_id)


@pytest.mark.parametrize("failure", [None, "checksum", "install"])
def test_interactive_setup_installs_verified_lock_and_requests_restart(tmp_path, monkeypatch, capsys, failure):
    import hashlib
    from io import BytesIO
    from urllib import request

    monkeypatch.chdir(tmp_path)
    locked = b"fixture==1.0 --hash=sha256:abc\n"
    code = notebook_cell("e1-setup")
    digest = next(n.comparators[0].value for n in ast.walk(ast.parse(code))
                  if isinstance(n, ast.Compare) and isinstance(n.left, ast.Call)
                  and isinstance(n.left.func, ast.Attribute) and n.left.func.attr == "hexdigest")
    code = code.replace(digest, hashlib.sha256(locked).hexdigest())
    monkeypatch.setattr(request, "urlopen", lambda *a, **k: BytesIO(b"tampered" if failure == "checksum" else locked))
    commands = []

    def install(command, **kwargs):
        commands.append(command)
        assert Path(command[-1]).read_bytes() == locked
        assert "--require-hashes" in command
        return SimpleNamespace(returncode=1 if failure == "install" else 0)

    monkeypatch.setattr(subprocess, "run", install)
    if failure:
        with pytest.raises((ValueError, RuntimeError), match={
            "checksum": "checksum", "install": "Installation failed"}[failure]):
            exec(code, {})
    else:
        exec(code, {})
        assert "no restart required" in capsys.readouterr().out
    assert len(commands) == (0 if failure == "checksum" else 1)


def test_setup_requests_restart_only_for_changed_loaded_packages(tmp_path, monkeypatch, capsys):
    from importlib import metadata
    from io import BytesIO
    from urllib import request

    monkeypatch.chdir(tmp_path)
    lock = (Path(__file__).parents[1] / "requirements/e1-runtime.txt").read_bytes()
    monkeypatch.setattr(request, "urlopen", lambda *a, **k: BytesIO(lock))
    monkeypatch.setattr(metadata, "packages_distributions", lambda: {"sys": ["fixture-loaded"]})
    versions = iter(["1.0", "2.0", "2.0", "2.0"])
    monkeypatch.setattr(metadata, "version", lambda name: next(versions))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    namespace = {}
    exec(notebook_cell("e1-setup"), namespace)
    assert namespace["_E1_RESTART_REQUIRED"] == ["fixture-loaded"]
    assert "Restart this session once" in capsys.readouterr().out
    # Re-running installation must not forget that old objects remain in memory.
    exec(notebook_cell("e1-setup"), namespace)
    assert namespace["_E1_RESTART_REQUIRED"] == ["fixture-loaded"]


def test_e1_runtime_lock_excludes_notebook_infrastructure():
    from packaging.requirements import Requirement

    lock = (Path(__file__).parents[1] / "requirements/e1-runtime.txt").read_text()
    requirements = [Requirement(line.rstrip(chr(92) + " ").split(" --hash=")[0])
                    for line in lock.splitlines() if line and not line[0].isspace() and not line.startswith("#")]
    names = {r.name for r in requirements}
    assert {"numpy", "pillow", "matplotlib", "onnxruntime", "pettrace-ml"} <= names
    assert not any(name.startswith(("jupyter", "ipython", "ipykernel", "ipywidgets", "nbconvert", "nbclient"))
                   or name == "pip" for name in names)


def test_interactive_verification_refuses_installing_kernel():
    with pytest.raises(RuntimeError, match="Restart"):
        exec(notebook_cell("e1-dependencies"), {"_E1_RESTART_REQUIRED": ["numpy"]})


def test_interactive_verification_rejects_tampered_lock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "requirements-e1-runtime.txt").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        exec(notebook_cell("e1-dependencies"), {})


def test_interactive_verification_rejects_dependency_drift(tmp_path, monkeypatch):
    import hashlib
    import importlib.metadata

    monkeypatch.chdir(tmp_path)
    locked = b"numpy==99.0\n"
    (tmp_path / "requirements-e1-runtime.txt").write_bytes(locked)
    code = notebook_cell("e1-dependencies")
    digest = next(n.comparators[0].value for n in ast.walk(ast.parse(code))
                  if isinstance(n, ast.Compare) and isinstance(n.left, ast.Call)
                  and isinstance(n.left.func, ast.Attribute) and n.left.func.attr == "hexdigest")
    code = code.replace(digest, hashlib.sha256(locked).hexdigest())
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.0")
    with pytest.raises(RuntimeError, match="numpy differs"):
        exec(code, {})


def test_e1_science_runs_in_visible_cells_without_worker_guards():
    notebook = nbformat.read(Path(__file__).parents[1] / "notebooks/e1_clip_zero_shot.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert "E1_IN_WORKER" not in cell.source
            assert "run_e1(" not in cell.source
    assert "ev = evaluate(" in notebook_cell("c3cd5a9e")
    assert "display(" in notebook_cell("ead8756f")


def test_bootstrap_drops_host_python_and_jupyter_configuration(tmp_path, monkeypatch):
    for key in ("PYTHONPATH", "PYTHONHOME", "JUPYTER_PATH", "IPYTHONDIR", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        monkeypatch.setenv(key, "/host/packages")
    monkeypatch.setenv("HF_TOKEN", "secret-test-value")
    env = run_e1.clean_environment(tmp_path)
    assert "/host/packages" not in env.values()
    assert env["HF_TOKEN"] == "secret-test-value"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["IPYTHONDIR"] == str(tmp_path / ".ipython")
    assert env["JUPYTER_NO_CONFIG"] == "1"


def test_download_rejects_wrong_release_checksum(tmp_path, monkeypatch):
    from io import BytesIO
    monkeypatch.setattr(run_e1, "urlopen", lambda *a, **k: BytesIO(b"changed"))
    target = tmp_path / "requirements.txt"
    with pytest.raises(ValueError, match="checksum"):
        run_e1.download("requirements.txt", target, {"requirements.txt": "0" * 64})
    assert not target.exists()


def test_e1_rejects_host_kernel_before_scientific_imports(monkeypatch):
    monkeypatch.setattr(e1.platform, "python_version", lambda: "3.13.16")
    with pytest.raises(RuntimeError, match="isolated Python 3.11.15"):
        e1.check_environment()


def test_hub_retry_respects_service_wait_and_does_not_retry_bad_credentials(monkeypatch):
    from urllib3.response import HTTPResponse

    delays = []
    monkeypatch.setattr("urllib3.util.retry.time.sleep", delays.append)
    with e1.hub_session() as session:
        retry = session.get_adapter("https://huggingface.co").max_retries
        retry.sleep(HTTPResponse(status=429, headers={"Retry-After": "134"}))
        assert delays == [134]
        assert retry.is_retry("GET", 429, has_retry_after=True)
        assert not retry.is_retry("GET", 401)
        assert not retry.is_retry("POST", 429, has_retry_after=True)


def test_notebook_attachment_round_trips_exact_bytes_and_detects_corruption(tmp_path):
    original = b"notebook-evidence" * 100000
    (tmp_path / "e1.executed.ipynb").write_bytes(original)
    folder = e1.notebook_artifacts(tmp_path)
    assert all(p.stat().st_size <= 512 * 1024 for p in folder.iterdir())
    subprocess.run([e1.sys.executable, str(folder / "restore_notebook.py")], check=True, capture_output=True)
    assert (folder / "e1.executed.ipynb").read_bytes() == original
    next(folder.glob("*.part*")).write_bytes(b"corrupted")
    result = subprocess.run([e1.sys.executable, str(folder / "restore_notebook.py")], capture_output=True, text=True)
    assert result.returncode != 0 and "checksum mismatch" in result.stderr


def test_kernel_uses_selected_python_and_retains_failure_evidence(tmp_path):
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("import sys; print(sys.executable)"),
        nbformat.v4.new_code_cell("raise RuntimeError('intentional failure')"),
        nbformat.v4.new_code_cell("print('must not execute')", outputs=[nbformat.v4.new_output("stream", text="old")]),
    ])
    nbformat.write(notebook, tmp_path / "e1_clip_zero_shot.ipynb")
    with pytest.raises(CellExecutionError, match="intentional failure"):
        e1.execute_notebook(tmp_path)
    saved = nbformat.read(tmp_path / "e1.executed.ipynb", as_version=4)
    assert e1.sys.executable in saved.cells[0].outputs[0].text
    assert saved.cells[1].outputs[-1].ename == "RuntimeError"
    assert saved.cells[2].outputs == []
    assert nbformat.read(tmp_path / "e1_clip_zero_shot.ipynb", as_version=4).cells[2].outputs[0].text == "old"


def saved_domains(tmp_path):
    for folder, repo, version, commit in ((tmp_path / "public_test", PUBLIC_REPO, "v1", PUBLIC_COMMIT),
                                         (tmp_path, e1.PRIVATE_REPO, "v4", e1.PRIVATE_COMMIT)):
        folder.mkdir(exist_ok=True)
        ds = SimpleNamespace(repo=repo, version=version, commit=commit, fingerprint="fingerprint")
        e1.save_publication_input(SimpleNamespace(dataset=ds, protocol="zero-shot", result={"saved": True}), folder)
        (folder / "runtime.json").write_text(json.dumps({
            "source_sha256": "same-code", "model_sha256": {"encoder.onnx": e1.ENCODER_SHA256}}))


def test_publication_retry_uses_saved_results_and_preserves_original_runtime(tmp_path, monkeypatch):
    saved_domains(tmp_path)
    before = (tmp_path / "runtime.json").read_bytes()
    monkeypatch.setattr(e1, "source_identity", lambda: {"source_sha256": "same-code"})
    monkeypatch.setattr("pettrace.evaluation.core.load_model", lambda uri: object())
    monkeypatch.setattr("pettrace.evaluation.core.evaluate", lambda *a, **k: pytest.fail("must not recompute"))
    calls = []

    def publish(ev, model, report, source, **kwargs):
        assert ev.result == {"saved": True}
        assert kwargs["expected_model_sha256"] == {"encoder.onnx": e1.ENCODER_SHA256}
        calls.append(ev.dataset.repo)
        return f"run-{len(calls)}"

    monkeypatch.setattr("pettrace.tracking.publish_evaluation", publish)
    receipt = e1.resume_publication(tmp_path)
    assert receipt == {"published": True, "public_run_id": "run-1", "run_id": "run-2"}
    assert calls == [PUBLIC_REPO, e1.PRIVATE_REPO]
    assert (tmp_path / "runtime.json").read_bytes() == before


def test_retry_rejects_changed_snapshot_before_tracking_writes(tmp_path, monkeypatch):
    saved_domains(tmp_path)
    path = tmp_path / "evaluation-input.json"
    payload = json.loads(path.read_text())
    payload["dataset"]["commit"] = "changed"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(e1, "source_identity", lambda: {"source_sha256": "same-code"})
    monkeypatch.setattr("pettrace.evaluation.core.load_model", lambda *a: pytest.fail("must validate first"))
    with pytest.raises(ValueError, match="pinned E1"):
        e1.resume_publication(tmp_path)


@pytest.mark.parametrize("layout", ["bin", "local/bin"])
def test_bootstrap_finds_uv_in_the_actual_prefix_layout(tmp_path, monkeypatch, layout):
    directory = tmp_path / ".bootstrap"
    executable = directory / layout / "uv"

    def install(command, **kwargs):
        assert command[command.index("--prefix") + 1] == str(directory)
        assert command[-1] == f"uv=={run_e1.UV}"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"uv fixture")
        # A Python package named uv must not be mistaken for the executable.
        (directory / "lib/uv").mkdir(parents=True)

    monkeypatch.setattr(run_e1.subprocess, "run", install)
    assert run_e1.bootstrap_uv(directory, {}) == str(executable)


@pytest.mark.parametrize("layouts", [[], ["bin", "local/bin"]])
def test_bootstrap_rejects_missing_or_ambiguous_executable(tmp_path, monkeypatch, layouts):
    for layout in layouts:
        folder = tmp_path / layout
        folder.mkdir(parents=True)
        (folder / "uv").write_bytes(b"uv fixture")
    monkeypatch.setattr(run_e1.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="Expected one installed uv executable"):
        run_e1.bootstrap_uv(tmp_path, {})


def test_bootstrap_kernel_does_not_load_the_host_colab_profile(tmp_path, monkeypatch):
    import os
    import sys

    host_profile = tmp_path / "host-ipython/profile_default"
    host_profile.mkdir(parents=True)
    (host_profile / "ipython_kernel_config.py").write_text(
        'c = get_config()\nc.IPKernelApp.kernel_class = "google.colab._kernel.Kernel"\n')
    monkeypatch.setenv("IPYTHONDIR", str(host_profile.parent))
    work = tmp_path / "execution"
    work.mkdir()
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        'from IPython import get_ipython\n'
        'assert type(get_ipython().kernel).__module__ == "ipykernel.ipkernel"\n'
        'print("isolated kernel executed")')])
    nbformat.write(notebook, work / "e1_clip_zero_shot.ipynb")
    # The same installed interpreter rejects the host's Colab-only kernel class.
    host = subprocess.run([sys.executable, "-I", "-m", "ipykernel_launcher", "--show-config-json"],
                          env=dict(os.environ), capture_output=True, text=True, timeout=30)
    assert host.returncode != 0 and "google.colab._kernel.Kernel" in host.stderr
    result = subprocess.run([sys.executable, "-I", "-c",
                             "import sys; from pettrace.experiments.e1 import execute_notebook; "
                             "execute_notebook(sys.argv[1])", str(work)],
                            env=run_e1.clean_environment(work), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    saved = nbformat.read(work / "e1.executed.ipynb", as_version=4)
    assert "isolated kernel executed" in saved.cells[0].outputs[0].text
    assert "google.colab._kernel.Kernel" not in result.stderr


@pytest.mark.parametrize("name", ["ipython_config.py", "ipython_kernel_config.py"])
def test_private_kernel_class_overrides_system_colab_config(tmp_path, name):
    import sys

    global_config = tmp_path / "system-ipython"
    global_config.mkdir()
    (global_config / name).write_text(
        'c = get_config()\nc.IPKernelApp.kernel_class = "google.colab._kernel.Kernel"\n')
    work = tmp_path / "execution"
    work.mkdir()
    # Point the actual IPython loader at a fixture instead of changing /etc on the test host.
    code = ("import sys; from IPython.core import application; "
            "application.SYSTEM_CONFIG_DIRS = [sys.argv.pop(1)]; application.ENV_CONFIG_DIRS = []; "
            "from ipykernel.kernelapp import IPKernelApp; "
            "app = IPKernelApp.instance(); app.initialize(); "
            "assert app.kernel_class.__module__ == 'ipykernel.ipkernel'; "
            "print('standard kernel initialized')")
    command = [sys.executable, "-I", "-c", code, str(global_config), "--IPKernelApp.quiet=False"]
    env = run_e1.clean_environment(work)
    before = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert before.returncode != 0 and "google.colab._kernel.Kernel" in before.stderr
    run_e1.prepare_kernel_profile(work)
    after = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert after.returncode == 0, after.stderr
    assert "standard kernel initialized" in after.stdout
    assert "google.colab._kernel.Kernel" in (global_config / name).read_text()


@pytest.mark.parametrize("name", ["ipython_config.py", "ipython_kernel_config.py"])
@pytest.mark.parametrize("section", ["InteractiveShellApp", "IPKernelApp"])
@pytest.mark.parametrize("field", ["extensions", "extra_extensions"])
def test_private_profile_does_not_load_colab_extensions(tmp_path, name, section, field):
    import sys

    global_config = tmp_path / "system-ipython"
    global_config.mkdir()
    contents = ('c = get_config()\n'
                'c.IPKernelApp.reraise_ipython_extension_failures = True\n'
                f'c.{section}.{field} = ["google.colab"]\n')
    (global_config / name).write_text(contents)
    work = tmp_path / "execution"
    work.mkdir()
    code = ("import sys; from IPython.core import application; "
            "application.SYSTEM_CONFIG_DIRS = [sys.argv.pop(1)]; application.ENV_CONFIG_DIRS = []; "
            "from ipykernel.kernelapp import IPKernelApp; "
            "app = IPKernelApp.instance(); app.initialize(); "
            "assert app.kernel_class.__module__ == 'ipykernel.ipkernel'; "
            "assert app.extensions == [] and app.extra_extensions == []; "
            "assert 'google.colab' not in sys.modules; print('kernel without host extensions')")
    command = [sys.executable, "-I", "-c", code, str(global_config), "--IPKernelApp.quiet=False"]
    env = run_e1.clean_environment(work)
    before = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert before.returncode != 0 and "No module named 'google.colab'" in before.stderr
    run_e1.prepare_kernel_profile(work)
    after = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert after.returncode == 0, after.stderr
    assert "kernel without host extensions" in after.stdout
    assert "google.colab" not in after.stderr
    assert (global_config / name).read_text() == contents
