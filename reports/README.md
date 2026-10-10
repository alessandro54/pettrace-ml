# Reports

Scores, figures and summaries per model × dataset version, produced by `make report` and attached to
the matching MLflow run. Current: `e1_v3`, `e2_v3` (dataset `pettrace-reid@v3`, pseudonymous codes).

Produced with onnxruntime 1.20.1 and Pillow 11.0; re-checked on 2026-10-10 after the move to
onnxruntime 1.31.0, Pillow 12.3.0 and mlflow-skinny 3.17.0 (the vision service's versions): same
Rank-1/5/10, mAP, EER and threshold for both, largest difference 5.4e-07 in `summary.json`, pair
scores within their 1e-05 rounding.
