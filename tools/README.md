# Operational tools

Experiment and dataset logic lives in `src/pettrace/`. These scripts cover package release,
model administration and MLflow bookkeeping; Colab experiments do not call them.

| Tool | Purpose |
|---|---|
| `attach_notebook.py` | Attach a saved executed notebook to its already-published MLflow runs. |
| `build_release.py` | Build the wheel, source distribution and checksums for a package release. |
| `sync_experiments.py` | Refresh `experiments.yaml` from MLflow without modifying runs. |
| `verify_model.py` | Verify a model against its golden embeddings. |
| `register_v1.py` | Reproduce or administer the frozen E1 CLIP registry entry. |
| `register_e2.py` | Reproduce or administer the frozen E2 detector-crop registry entry. |

Registration and experiment publication remain separate. The E3 volume launchers were removed
after their published historical runs; the canonical multiseed study now lives in
`notebooks/e3_mlp_head_triplet.ipynb`.
