"""Archive the saved interactive notebook without touching scientific results."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pettrace.experiments.e1 import PRIVATE_COMMIT
from pettrace.evaluation.research import PUBLIC_COMMIT
from tools.attach_notebook import attach_e1


class ArtifactClient:
    def __init__(self):
        self.artifacts = {}
        self.corrupt = False
        self.status = "FINISHED"
        self.writes = []

    def get_run(self, run_id):
        return SimpleNamespace(info=SimpleNamespace(status=self.status), data=SimpleNamespace(
            tags={"experiment": "E1"}, params={"dataset.commit": PUBLIC_COMMIT if run_id == "public" else PRIVATE_COMMIT}))

    def list_artifacts(self, run_id, folder):
        return [SimpleNamespace(path=path) for rid, path in self.artifacts if rid == run_id and path.startswith(folder)]

    def log_artifact(self, run_id, filename, artifact_path):
        path = artifact_path + "/" + Path(filename).name
        self.artifacts[run_id, path] = Path(filename).read_bytes()
        self.writes.append((run_id, path))

    def download_artifacts(self, run_id, path, destination):
        data = self.artifacts[run_id, path]
        if self.corrupt and ".part" in path:
            data = b"damaged"
        target = Path(destination) / Path(path).name
        target.write_bytes(data)
        return str(target)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    client = ArtifactClient()
    monkeypatch.setattr("pettrace.tracking.production_context", lambda: (client, None))
    notebook = tmp_path / "e1.ipynb"
    notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "execution_count": 1,
        "source": "# " + "x" * 1100000,
        "outputs": [{"output_type": "stream", "text": "published public and private"}]}]}))
    return client, notebook


def test_archive_both_runs_receipts_and_exact_notebook(archive):
    client, notebook = archive
    original = notebook.read_bytes()
    attach_e1(notebook, "private", "public")
    for rid in ("public", "private"):
        manifest = json.loads(client.artifacts[rid, "notebook/manifest.json"])
        parts = [client.artifacts[rid, "notebook/" + p["file"]] for p in manifest["parts"]]
        assert b"".join(parts) == original
        assert max(map(len, parts)) <= 512 * 1024
        receipt = json.loads(client.artifacts[rid, "report/publication.json"])
        assert receipt["published"] and receipt["run_id"] == rid
    assert json.loads(client.artifacts["private", "report/publication.json"])["public_run_id"] == "public"
    saved = dict(client.artifacts)
    attach_e1(notebook, "private", "public")
    assert client.artifacts == saved
    assert notebook.read_bytes() == original


def test_archive_rejects_incomplete_run_before_upload(archive):
    client, notebook = archive
    client.status = "FAILED"
    with pytest.raises(ValueError, match="completed E1"):
        attach_e1(notebook, "private", "public")
    assert not client.writes


def test_archive_detects_corrupt_download_and_can_retry(archive):
    client, notebook = archive
    client.corrupt = True
    with pytest.raises(RuntimeError, match="verification failed"):
        attach_e1(notebook, "private", "public")
    assert not any(path == "report/publication.json" for _, path in client.writes)
    client.corrupt = False
    attach_e1(notebook, "private", "public")
    assert ("private", "report/publication.json") in client.artifacts


def test_archive_preserves_existing_different_notebook(archive):
    client, notebook = archive
    client.artifacts["private", "notebook/manifest.json"] = json.dumps({"sha256": "different"}).encode()
    with pytest.raises(ValueError, match="different notebook"):
        attach_e1(notebook, "private", "public")
    assert not client.writes


def test_notebook_uploads_final_receipts_for_both_domains(tmp_path):
    import ast

    notebook = json.loads((Path(__file__).parents[1] / "notebooks/e1_clip_zero_shot.ipynb").read_text())
    source = "".join(next(c for c in notebook["cells"] if c["id"] == "38c87fe3")["source"])
    loop = next(n for n in ast.parse(source).body if isinstance(n, ast.For) and "published_id" in ast.unparse(n.target))
    client = ArtifactClient()
    public = tmp_path / "public_test"
    public.mkdir()
    (tmp_path / "publication.json").write_text(json.dumps({"published": True, "run_id": "private", "public_run_id": "public"}))
    (public / "publication.json").write_text(json.dumps({"published": True, "run_id": "public"}))
    exec(compile(ast.Module(body=[loop], type_ignores=[]), "receipt-upload", "exec"),
         {"production_client": client, "public_run_id": "public", "run_id": "private", "REPORT": tmp_path, "PUBLIC_REPORT": public})
    assert len(client.artifacts) == 2
    assert all(json.loads(data)["published"] for data in client.artifacts.values())
