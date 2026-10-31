# Operational tools

These commands support model administration and local automation. Experiment notebooks do not
import or execute these files. Install `pettrace-ml` before running them, or use the Make targets.

| Tool | Purpose |
|---|---|
| `register_v1.py` | Export/verify the original CLIP model |
| `register_e2.py` | Export/register the detector-crop model |
| `verify_model.py` | Check a downloaded model's golden fingerprint |
| `attach_notebook.py` | Attach a saved executed notebook to its published run |
| `mlflow_smoke.py` | Check tracking connectivity |
| `build_release.py` | Build the versioned wheel and source distribution with embedded provenance and checksums |
| `run_e1.py` | Bootstrap the released E1 notebook in isolated Python 3.11.15; retry publication with `--resume` |
| `sync_experiments.py` | Refresh `experiments.yaml` from MLflow without modifying runs |
| `archive/squash_dataset.py` | Historical one-off dataset maintenance; not part of experiment execution |

Reusable dataset acquisition is in `pettrace.data.acquisition`; optional E3 orchestration is in
`pettrace.experiments.legacy`. The evaluation CLI is `pettrace-evaluate` or `python -m pettrace.cli`.
Tests live only in `tests/` and are excluded from the runtime wheel.
