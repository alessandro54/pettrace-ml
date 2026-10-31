# E1: isolated, versioned execution

## Colab or VS Code connected to Colab

Open [the v0.2.2 launcher](https://colab.research.google.com/github/alessandro54/pettrace-ml/blob/v0.2.2/colab/e1.ipynb)
and run its three cells. CPU is sufficient. Supply these values through Colab Secrets or hidden prompts:

- `HF_TOKEN`: read access to `alessandro54/pettrace-reid` and `alessandro54/pettrace-training`.
- `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`: access to the existing
  `pettrace-reid-eval` experiment and model registry at `https://ml.pettrace.app`, including permission
  to create runs, log metrics and upload artifacts. A read-only MLflow account cannot complete E1.

The launcher downloads checksummed assets from v0.2.2, installs uv 0.11.22 in a separate bootstrap
directory, creates managed Python **3.11.15** and a virtual environment without system packages,
and installs the release wheel and exact dependencies with `--require-hashes`.
The experiment uses a separate Jupyter kernel whose executable is that environment's Python.
The host Colab interpreter and its NumPy/Pillow installations are not modified or imported by E1.
No kernel restart is needed. Keep the Colab session connected while it runs.

Before downloading images, the runner checks imports, HF authentication and both immutable dataset
commits, plus access to MLflow and model version 1. Write access is confirmed by actual publication.
The notebook then checks dataset fingerprints, model hashes and golden embeddings.

## Terminal

From the v0.2.2 checkout, with a local `.env` containing the credentials:

```bash
python tools/run_e1.py --work-dir reports/my-e1 --env-file .env
```

The host Python needs pip and network access. The launcher downloads the same published notebook
used in Colab; local source edits do not silently change the experiment. Use a new work directory
for each new execution. Secrets are passed through the environment, not written into evidence.

## Evidence and recovery

Each work directory contains `e1.executed.ipynb`, `execution.json`, the hashed requirements and
release identity, plus `reports/e1_v4/<run>/` with separate `public_test/` results.
The runner checkpoints notebook outputs after each cell and on failure. The original notebook
and its historical outputs are preserved. Each domain publishes to MLflow, and the runner attaches
the executed notebook and environment evidence to both runs. The top-level `publication.json`
exists only after both runs are finished and attachments have succeeded.

If publication fails after evaluation, correct credentials/connectivity and run:

```bash
python tools/run_e1.py --work-dir reports/my-e1 --env-file .env --resume
```

This reuses saved numerical results and their original provenance. It verifies the model again
but does not repeat dataset downloads or embedding evaluation. Existing publication keys resume
incomplete uploads or reuse completed runs. Do not run concurrent retries for the same evidence.
If the failure preceded evaluation completion, inspect the saved notebook before starting a new run.
Colab storage is temporary: download unpublished evidence before discarding the session.

## Reproducibility limits

The reference is a fixed Python, wheel, dependencies, datasets, model and protocol. Hardware and OS
are recorded, but not made identical by a virtual environment. E1 uses frozen CPU inference and
has no randomized training. Repeat executions on the same environment should be compared using
the saved full results; comparisons across architectures must report observed numeric differences.
Do not interpret successful environment setup or synthetic tests as reproduced scientific results.
Public and private galleries remain separate. Descriptive EER is fitted on the evaluated pairs;
the evaluation does not establish a production threshold or generalization across capture sessions.
