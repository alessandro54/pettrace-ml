"""Fixed E3 control evidence for the progressive E4 experiment."""

import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from pettrace.tracking import production_context
from pettrace.training.lora import load_frozen_e3_head

CONTROL_RUNS = {
    "public": "959ba0156b484c15ad32692df44e67f8",
    "private": "79bdb46d24b24878a0e2f904743ea464",
}
CONTROL_MODEL_VERSION = "20"
CONTROL_HEAD_SHA256 = "4ff76c5f79bd1acaf6a8732905ee69bcaf9c93983d1e3332142965f18c2cb4ed"
CONTROL_SEED = "46"


def _query_signature(rows):
    return sorted(("/".join(PurePosixPath(row["photo"]).parts[-2:]),
                   row["animal"], row["species"], row["gallery_animals"]) for row in rows)


def _run(client, run_id):
    run = client.get_run(run_id)
    if (run.info.status != "FINISHED" or run.data.tags.get("experiment") != "E3"
            or run.data.params.get("training.seed") != CONTROL_SEED
            or not run.data.params.get("protocol", "").startswith("zero-shot:")):
        raise ValueError("The fixed E3-100 seed-46 control is unavailable or changed")
    return run


def load_control_head(output):
    """Download and verify the prospectively fixed E3 head from its private MLflow run."""
    client, _ = production_context()
    run = _run(client, CONTROL_RUNS["private"])
    version = client.get_model_version("pettrace-embedder", CONTROL_MODEL_VERSION)
    if version.status != "READY" or version.run_id != run.info.run_id:
        raise ValueError("Registered E3 control version does not match its fixed run")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(client.download_artifacts(run.info.run_id, "report/head.pt", str(output)))
    head = load_frozen_e3_head(checkpoint, CONTROL_HEAD_SHA256)
    receipt = {"run_id": run.info.run_id, "model_version": CONTROL_MODEL_VERSION,
               "head_sha256": CONTROL_HEAD_SHA256, "artifact": "report/head.pt"}
    (output / "e3-control.json").write_text(json.dumps(receipt, indent=2))
    return head, receipt


def frozen_control(evaluation, domain, output):
    """Load the matching published E3 evaluation and reject different queries or model hashes."""
    client, _ = production_context()
    run = _run(client, CONTROL_RUNS[domain])
    if (run.data.params.get("dataset.commit") != evaluation.dataset.commit
            or run.data.params.get("dataset.fingerprint") != evaluation.dataset.fingerprint):
        raise ValueError("E3 control uses a different benchmark")
    private = _run(client, CONTROL_RUNS["private"])
    hashes = {key: value for key, value in run.data.params.items() if key.startswith("model.sha256.")}
    private_hashes = {key: value for key, value in private.data.params.items() if key.startswith("model.sha256.")}
    if not hashes or hashes != private_hashes:
        raise ValueError("E3 control runs do not use the same model")
    folder = Path(output) / "baseline_e3"
    folder.mkdir(parents=True, exist_ok=True)
    evaluation_path = Path(client.download_artifacts(run.info.run_id, "evaluation.json", str(folder)))
    summary_path = Path(client.download_artifacts(run.info.run_id, "report/summary.json", str(folder)))
    result = json.loads(evaluation_path.read_text())

    if _query_signature(result["test"]["per_query"]) != _query_signature(evaluation.result["test"]["per_query"]):
        raise ValueError("E3 control queries or galleries differ from E4")
    receipt = {"run_id": run.info.run_id,
               "evaluation_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
               "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest()}
    (folder / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return SimpleNamespace(result=result, dataset=evaluation.dataset), json.loads(summary_path.read_text())
