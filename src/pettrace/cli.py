"""PetTrace command-line interface.

    make evaluate DATASET=v3 MODEL=models:/pettrace-embedder/1 PROTOCOL=zero-shot [REPORT=reports/e1_v3]
    pettrace evaluate --model /path/to/model/dir --dataset /path/to/dataset/dir
    pettrace run e1
    pettrace run e2

`evaluate` is a generic evaluator. `run` executes a reproducible experiment and publishes it to MLflow.
"""

import argparse
import os
import sys

from pettrace.data import benchmark as data
from pettrace.evaluation.core import evaluate, load_model, log_run, report
from pettrace.tracking import tracking_configured, tracking_context


def evaluate_main(argv=None) -> int:
    """Run the generic evaluator; experiment runners live under `pettrace run`."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="local model directory or configured MLflow model URI")
    ap.add_argument("--dataset", required=True, help="version tag (v3) or a local dataset directory")
    ap.add_argument("--repo", default=data.REPO)
    ap.add_argument("--protocol", choices=("split", "zero-shot"), default="split",
                    help="zero-shot: every photo is a query (pretrained models only); EER on the same pairs")
    ap.add_argument("--report", help="directory for summary.json, scores.json and figures")
    ap.add_argument("--experiment", default="", help="E1…E5, tagged on the run")
    args = ap.parse_args(argv)
    if args.experiment and not tracking_configured():
        ap.error("--experiment requires MLFLOW_TRACKING_URI; untracked experiment runs are not allowed")

    ds = data.from_dir(args.dataset) if os.path.isdir(args.dataset) else data.load(args.dataset, args.repo)
    model = load_model(args.model)
    if not model.golden_ok:
        print(f"warning: {args.model} does not reproduce its golden fingerprint", file=sys.stderr)
    ev = evaluate(ds, model.pipeline, zero_shot=args.protocol == "zero-shot")
    print(report(ev.result))
    if args.report:
        from pettrace.evaluation.figures import save

        save(ev, model, args.report)
        print(f"report → {args.report}")
    if tracking_configured():
        _, tracking_experiment = tracking_context()
        run_id = log_run(ev, model, experiment=args.experiment, artifacts=args.report,
                         tracking_experiment_id=tracking_experiment.experiment_id)
        print(f"run {run_id}")
    else:
        print("MLflow not configured; evaluation was not published")
    return 0


def main(argv=None) -> int:
    """Dispatch the public CLI without duplicating experiment parsers."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["evaluate"]:
        return evaluate_main(argv[1:])
    if len(argv) >= 2 and argv[0] == "run" and argv[1] in {"e1", "e2"}:
        if argv[1] == "e1":
            from pettrace.experiments.e1 import main as experiment_main
        else:
            from pettrace.experiments.e2 import main as experiment_main
        return experiment_main(argv[2:])

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", nargs="?", choices=("evaluate", "run"))
    parser.add_argument("experiment", nargs="?", choices=("e1", "e2"))
    parser.parse_args(argv)
    parser.error("choose `evaluate`, `run e1` or `run e2`")


if __name__ == "__main__":
    sys.exit(main())
