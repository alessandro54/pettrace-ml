# Notebooks: one per experiment

Each experiment E1–E5 has exactly one notebook, `eK_<name>.ipynb`, which is its canonical record.
Every notebook follows the same sections:

1. **Datos**: the pinned `pettrace-reid` version (commit, fingerprint, counts; never photos: the
   dataset is private and animals are pseudonymous).
2. **Modelo**: the registered version, its sha256 and the golden-fingerprint check.
3. **Entrenamiento**: the training itself when the experiment has one (E3–E5, Colab GPU; seeds,
   P×K sampler, registration as `@candidate`); "no aplica" for the zero-shot E1/E2.
4. **Evaluación**: `reid.evaluation` with the thesis protocol.
5. **Resultados**: tables and figures (`reid.figures`), compared with the previous experiments.
6. **Registro en MLflow**: one run in `pettrace-reid-eval` with code commit, dataset commit and model
   hashes; figures under `report/` and the executed notebook under `notebook/`.
7. **Conclusiones**.

| | Notebook | Status |
|---|---|---|
| E1 | [`e1_clip_zero_shot.ipynb`](e1_clip_zero_shot.ipynb) | done |
| E2 | [`e2_yolo_crop_zero_shot.ipynb`](e2_yolo_crop_zero_shot.ipynb) | done |
| E3 | `e3_mlp_head_triplet.ipynb` | needs a dataset with `train`/`val` |
| E4 | `e4_lora.ipynb` | after E3 |
| E5 | `e5_text_rerank.ipynb` | after E4 |

Run: `make notebook NB=notebooks/e1_clip_zero_shot.ipynb` (pinned image; commit first so the run
records a clean `git_sha`), then commit the notebook **with its outputs** and `reports/`.
Exploratory notebooks are not kept.
