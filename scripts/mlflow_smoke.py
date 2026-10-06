"""MLflow end-to-end smoke test: tracking, artifact store and model registry.

Needs only the MLflow client (`pip install mlflow-skinny`) and the standard env vars:
    MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD

Run locally:  python scripts/mlflow_smoke.py
In Colab:     run this file as a cell after setting the MLFLOW_* variables
It logs one run to experiment `pettrace-smoke`, uploads an artifact, registers it as a version of
`pettrace-smoke-model`, points the `candidate` alias at it, then reads everything back.
"""

import json
import os
import platform
import tempfile
import time

import mlflow
from mlflow import MlflowClient

EXPERIMENT = "pettrace-smoke"
MODEL = "pettrace-smoke-model"


def main() -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.chumpitaz.dev")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    with mlflow.start_run(run_name=f"smoke-{platform.node()}") as run:
        mlflow.log_params({"model": "clip-vit-b32", "pipeline": "E1", "host": platform.node()})
        for step, value in enumerate([0.60, 0.72, 0.81]):
            mlflow.log_metric("rank1", value, step=step)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pipeline.json")
            with open(path, "w") as f:
                json.dump({"detector": None, "embedding_dims": 512, "model_version": "smoke@0"}, f)
            mlflow.log_artifact(path, artifact_path="model")
        run_id = run.info.run_id

    # Artifact store round trip (server proxies R2/RustFS; the client has no storage credentials).
    local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model/pipeline.json")
    assert json.load(open(local))["embedding_dims"] == 512

    # Registry: new version + alias, then resolve the alias back.
    try:
        client.create_registered_model(MODEL)
    except mlflow.exceptions.MlflowException:
        pass  # already exists
    mv = client.create_model_version(MODEL, source=f"runs:/{run_id}/model", run_id=run_id)
    for _ in range(30):
        if client.get_model_version(MODEL, mv.version).status == "READY":
            break
        time.sleep(1)
    client.set_registered_model_alias(MODEL, "candidate", mv.version)
    resolved = client.get_model_version_by_alias(MODEL, "candidate")

    metric = client.get_run(run_id).data.metrics["rank1"]
    print(f"tracking  ✓ run {run_id} in '{EXPERIMENT}', rank1={metric}")
    print("artifacts ✓ model/pipeline.json uploaded and downloaded through the server")
    print(f"registry  ✓ {MODEL} v{mv.version}, alias candidate → v{resolved.version}")
    print(f"UI: {uri}/#/experiments  ·  {uri}/#/models/{MODEL}")


if __name__ == "__main__":
    main()
