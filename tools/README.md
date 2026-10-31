# Operational tools

Experiment code lives in `src/pettrace/`; these scripts handle packaging and administration.
Interactive Colab experiments do not require running them. Install `pettrace-ml` first.

| Tool | When to use it |
|---|---|
| `attach_notebook.py` | After saving an executed notebook: archive its outputs in MLflow. E1/E2 use both run IDs; see [E1](../docs/e1_colab.md) or [E2](../docs/e2_colab.md). |
| `build_release.py` | Before publishing a package release: build wheel, sdist and checksums. |
| `sync_experiments.py` | After publishing experiments: refresh `experiments.yaml` with `make manifest` (read-only on MLflow). |
| `verify_model.py` | Check a model's golden embeddings with `make verify-model MODEL=...`; supports offline artifacts. |
| `register_v1.py` | Model administration: export/register CLIP with `make register-v1`. Can update a registry alias. |
| `register_e2.py` | Model administration: export/register the detector-crop pipeline with `make register-e2`. Can update a registry alias. |
| `run_e1.py` | Optional historical reproduction: run the released v0.2.2 notebook in isolated Python 3.11.15, or retry with `--resume`. |

The registration scripts are not needed to evaluate existing pinned model versions.
The historical dataset-squashing and synthetic MLflow smoke scripts were retired; Git retains them.

Reusable dataset acquisition is in `pettrace.data.acquisition`; optional E3 orchestration is in
`pettrace.experiments.legacy`. The evaluation CLI is `pettrace-evaluate` or `python -m pettrace.cli`.
Tests live only in `tests/` and are excluded from the runtime wheel.
