"""Attach an executed experiment notebook (with its outputs) to the MLflow run it logged.

    python tools/attach_notebook.py notebooks/e1_clip_zero_shot.ipynb      # run by `make notebook`

Interactive E1: supply --run-id PRIVATE --public-run-id PUBLIC, optionally --env-file .env.
Other notebooks declare metadata.pettrace.report and write run.json there.
Save the notebook after execution so the attachment includes its final outputs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import mlflow


def attach_e1(notebook, private_run, public_run):
    """Archive a saved interactive E1 in both completed runs without reevaluating."""
    from pettrace.experiments.e1 import PRIVATE_COMMIT, notebook_artifacts
    from pettrace.evaluation.research import PUBLIC_COMMIT
    from pettrace.tracking import production_context

    contents = Path(notebook).read_bytes()
    cells = json.loads(contents)["cells"]
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    outputs = json.dumps([cell.get("outputs", []) for cell in code])
    if (not code or any(cell.get("execution_count") is None for cell in code)
            or any(out.get("output_type") == "error" for cell in code for out in cell.get("outputs", []))
            or not all(run in outputs for run in (private_run, public_run))):
        raise ValueError("Save the executed E1 notebook with both publication links and no cell errors")
    for name in ("HF_TOKEN", "MLFLOW_TRACKING_PASSWORD"):
        secret = os.environ.get(name, "")
        if secret and secret.encode() in contents:
            raise ValueError("A credential appears in the notebook; refusing upload")
    client, _ = production_context()
    for run_id, commit in ((public_run, PUBLIC_COMMIT), (private_run, PRIVATE_COMMIT)):
        run = client.get_run(run_id)
        if (run.info.status != "FINISHED" or run.data.tags.get("experiment") != "E1"
                or run.data.params.get("dataset.commit") != commit):
            raise ValueError(f"Not a completed E1 run for the expected dataset: {run_id}")
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "e1.executed.ipynb").write_bytes(contents)
        attachment = notebook_artifacts(work)
        manifest = json.loads((attachment / "manifest.json").read_text())
        # Preflight both destinations before overwriting any evidence.
        for run_id in (public_run, private_run):
            if any(item.path == "notebook/manifest.json" for item in client.list_artifacts(run_id, "notebook")):
                dest = work / run_id
                dest.mkdir()
                old = client.download_artifacts(run_id, "notebook/manifest.json", str(dest))
                if json.loads(Path(old).read_text())["sha256"] != manifest["sha256"]:
                    raise ValueError(f"A different notebook is already archived in {run_id}")
        files = [part["file"] for part in manifest["parts"]] + ["restore_notebook.py", "manifest.json"]
        for run_id in (public_run, private_run):
            dest = work / run_id
            dest.mkdir(exist_ok=True)
            for name in files:
                client.log_artifact(run_id, str(attachment / name), artifact_path="notebook")
            digest = hashlib.sha256()
            for name in files:
                downloaded = Path(client.download_artifacts(run_id, "notebook/" + name, str(dest)))
                data = downloaded.read_bytes()
                if data != (attachment / name).read_bytes():
                    raise RuntimeError(f"Uploaded notebook verification failed: {run_id}/{name}")
                if name.startswith("e1.executed.ipynb.part"):
                    digest.update(data)
            if digest.hexdigest() != manifest["sha256"]:
                raise RuntimeError("Reconstructed notebook checksum mismatch")
            receipt = {"published": True, "run_id": run_id}
            if run_id == private_run:
                receipt["public_run_id"] = public_run
            path = dest / "publication.json"
            path.write_text(json.dumps(receipt, indent=2) + "\n")
            client.log_artifact(run_id, str(path), artifact_path="report")
            remote = client.download_artifacts(run_id, "report/publication.json", str(dest))
            if json.loads(Path(remote).read_text()) != receipt:
                raise RuntimeError(f"Publication receipt verification failed: {run_id}")
            print(f"Archived and verified E1 notebook and receipt: {run_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook")
    parser.add_argument("--run-id", help="Private E1 run; requires --public-run-id")
    parser.add_argument("--public-run-id", help="Public E1 run; requires --run-id")
    parser.add_argument("--env-file")
    args = parser.parse_args()
    if bool(args.run_id) != bool(args.public_run_id):
        parser.error("Provide both E1 run IDs")
    if args.env_file:
        from pettrace.experiments.e1 import credentials
        credentials(args.env_file)
    if args.run_id:
        from pettrace.tracking import PRODUCTION_URI
        mlflow.set_tracking_uri(PRODUCTION_URI)
        attach_e1(args.notebook, args.run_id, args.public_run_id)
        return 0
    nb_path = args.notebook
    meta = json.load(open(nb_path))["metadata"].get("pettrace", {})
    receipt_path = meta.get("publication_receipt")
    receipt = None
    if receipt_path:
        if not os.path.exists(receipt_path):
            print(f"{nb_path}: missing publication receipt; refusing to attach to an old run", file=sys.stderr)
            return 1
        with open(receipt_path) as f:
            receipt = json.load(f)
        if not receipt.get("published"):
            print(f"{nb_path}: publication disabled; notebook attachment skipped")
            return 0
    run_file = os.path.join(meta.get("report", ""), "run.json")
    if not os.path.exists(run_file):
        print(f"{nb_path}: no {run_file} (the notebook did not log a run)", file=sys.stderr)
        return 1
    run_id = json.load(open(run_file))["run_id"]
    if receipt is not None and receipt.get("run_id") != run_id:
        print(f"{nb_path}: publication receipt does not match run.json", file=sys.stderr)
        return 1
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(nb_path, artifact_path="notebook")
    print(f"attached {nb_path} to run {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
