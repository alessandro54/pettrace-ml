# Operational tools

Experiment code lives in `src/pettrace/`; these scripts handle packaging and administration.
Interactive Colab experiments do not require running them. Install `pettrace-ml` first.

| Tool | When to use it |
|---|---|
| `attach_notebook.py` | After saving an executed notebook: archive its outputs in MLflow. E1/E2/E3 use both run IDs (`--experiment E3` for E3); see [E1](../docs/e1_colab.md) or [E2](../docs/e2_colab.md). |
| `build_release.py` | Before publishing a package release: build wheel, sdist and checksums. |
| `sync_experiments.py` | After publishing experiments: refresh `experiments.yaml` with `make manifest` (read-only on MLflow). |
| `repeat_e3.py` | Paired v2.50/v3.50 checks with seeds 42/43/44 in an already verified E3 kernel. `run(globals(), output)` reuses checksum-verified frozen embeddings, selects only on public validation, and publishes both benchmarks plus candidate models. |
| `run_e3_volume.py` | Queue the planned v4.20/50/100 series in the existing Colab runtime. Waits for successful preparation receipts and its process to exit, then uses `repeat_e3.py` for nine fits and mandatory MLflow publication. Requires the adjacent `volume-plan.json` and existing credentials. |
| `verify_model.py` | Check a model's golden embeddings with `make verify-model MODEL=...`; supports offline artifacts. |
| `register_v1.py` | Model administration: export/register CLIP with `make register-v1`. Can update a registry alias. |
| `register_e2.py` | Model administration: export/register the detector-crop pipeline with `make register-e2`. Can update a registry alias. |
| `run_e1.py` | Optional historical reproduction: run the released v0.2.2 notebook in isolated Python 3.11.15, or retry with `--resume`. |

The registration scripts are not needed to evaluate existing pinned model versions.
E3 repeats require the original manifests and complete embedding cache. Reusing the same output
reuses finished fits and retries publication; an interrupted fit is preserved and requires an
explicit new output. These repeats compare curation at 50%, not training volume.
The volume launcher checks the shared automatic policy and immutable benchmarks before extracting
missing embeddings. Run `python run_e3_volume.py /content/reports/e3-volume-RUN`; monitor `status.json`
there or `MyDrive/PetTrace/data-lookup/inventory100/experiment-status.json`. Only one volume launcher
can run per VM. It survives a terminal disconnect, not deletion of the Colab runtime.
The historical dataset-squashing and synthetic MLflow smoke scripts were retired; Git retains them.

Reusable dataset acquisition is in `pettrace.data.acquisition`; optional E3 orchestration is in
`pettrace.experiments.legacy`. The evaluation CLI is `pettrace-evaluate` or `python -m pettrace.cli`.
Tests live only in `tests/` and are excluded from the runtime wheel.
