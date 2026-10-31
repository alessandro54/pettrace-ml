"""Refresh the small experiment index from MLflow; never modify runs or download artifacts.

Run with tracking credentials in the environment, or use `make manifest` to load the local .env.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import mlflow
import yaml

from pettrace.evaluation.core import EXPERIMENT
from pettrace.tracking import PRODUCTION_URI


METRICS = {
    "rank1": "test.global.rank1", "rank5": "test.global.rank5", "mAP": "test.global.mAP", "eer": "val.eer",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "experiments.yaml")
    args = parser.parse_args()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", PRODUCTION_URI).rstrip("/")
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {EXPERIMENT!r} was not found.")
    entries, token = [], None
    while True:
        page = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time ASC"],
                                  max_results=100, page_token=token)
        for run in page:
            label = run.data.tags.get("experiment")
            if label not in {"E1", "E2", "E3", "E4", "E5"}:
                continue
            params, metrics = run.data.params, run.data.metrics
            entries.append({
                "experiment": label,
                "run_id": run.info.run_id,
                "url": f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}",
                "status": run.info.status,
                "started_at": datetime.fromtimestamp(run.info.start_time / 1000, timezone.utc).isoformat(),
                "dataset": run.data.tags.get("dataset"),
                "dataset_commit": params.get("dataset.commit"),
                "model": params.get("model.model_version"),
                "model_uri": params.get("model.uri"),
                "protocol": params.get("protocol"),
                "code_commit": run.data.tags.get("git_sha"),
                "metrics": {name: float(metrics[key]) for name, key in METRICS.items() if key in metrics},
            })
        token = page.token
        if not token:
            break
    manifest = {"schema_version": 1, "tracking_uri": tracking_uri,
                "mlflow_experiment": {"name": experiment.name, "id": experiment.experiment_id}, "runs": entries}
    text = ("# Small reference index; MLflow is the source of truth for complete results.\n"
            "# Metrics are snapshots with values on a 0-1 scale. Refresh with make manifest.\n"
            + yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix(args.output.suffix + ".tmp")
    pending.write_text(text)
    pending.replace(args.output)
    print(f"Indexed {len(entries)} evaluation runs in {args.output.name}.")


if __name__ == "__main__":
    main()
