"""Attach an executed experiment notebook (with its outputs) to the MLflow run it logged.

    python scripts/attach_notebook.py notebooks/e1_clip_zero_shot.ipynb      # run by `make notebook`

The notebook declares its report directory in metadata.pettrace.report and writes run.json there
when it logs its run; the notebook file itself can only be attached once execution has finished.
"""

import json
import os
import sys

import mlflow


def main() -> int:
    nb_path = sys.argv[1]
    meta = json.load(open(nb_path))["metadata"].get("pettrace", {})
    run_file = os.path.join(meta.get("report", ""), "run.json")
    if not os.path.exists(run_file):
        print(f"{nb_path}: no {run_file} (the notebook did not log a run)", file=sys.stderr)
        return 1
    run_id = json.load(open(run_file))["run_id"]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(nb_path, artifact_path="notebook")
    print(f"attached {nb_path} to run {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
