"""Evaluate a `pettrace-embedder` version on a `pettrace-reid` version (thesis protocol).

    make evaluate DATASET=v3 MODEL=models:/pettrace-embedder/1 PROTOCOL=zero-shot [REPORT=reports/e1_v3]
    python -m pettrace.cli --model /path/to/model/dir --dataset /path/to/dataset/dir

The protocol, model identity and MLflow logging live in pettrace.evaluation (shared with the experiment
notebooks, which are the canonical record of each experiment). --report also writes the figures
and tables (pettrace.figures) and attaches them to the run.
"""

import argparse
import os
import sys

from pettrace import data
from pettrace.evaluation import evaluate, load_model, log_run, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="models:/pettrace-embedder@production")
    ap.add_argument("--dataset", required=True, help="version tag (v3) or a local dataset directory")
    ap.add_argument("--repo", default=data.REPO)
    ap.add_argument("--protocol", choices=("split", "zero-shot"), default="split",
                    help="zero-shot: every photo is a query (pretrained models only); EER on the same pairs")
    ap.add_argument("--report", help="directory for summary.json, scores.json and figures")
    ap.add_argument("--experiment", default="", help="E1…E5, tagged on the run")
    args = ap.parse_args()

    ds = data.from_dir(args.dataset) if os.path.isdir(args.dataset) else data.load(args.dataset, args.repo)
    model = load_model(args.model)
    if not model.golden_ok:
        print(f"warning: {args.model} does not reproduce its golden fingerprint", file=sys.stderr)
    ev = evaluate(ds, model.pipeline, zero_shot=args.protocol == "zero-shot")
    print(report(ev.result))
    if args.report:
        from pettrace.figures import save

        save(ev, model, args.report)
        print(f"report → {args.report}")
    run_id = log_run(ev, model, experiment=args.experiment, artifacts=args.report)
    print(f"run {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
