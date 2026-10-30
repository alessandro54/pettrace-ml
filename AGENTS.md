# Experiment policy

All current and future experiment runs must publish their results to MLflow at https://ml.pettrace.app. This is a user requirement, effective 2026-10-30.

- Experiment notebooks must publish unconditionally; `LOG_TO_MLFLOW`, if displayed, must be `True` and cannot disable publication.
- Do not introduce environment variables or CLI flags that bypass experiment tracking.
- An experiment is complete only after successful publication. Preserve local evidence on errors and support retrying publication without retraining.
- Keep model version registration and serving alias promotion separate from publishing experiment results. Preserve the user's chosen registration setting.
- Dataset preparation and Hugging Face dataset uploads are separate workflows; this policy concerns experiments.
- Preserve executed notebook outputs when editing source. Never place credentials in notebook source or outputs.

## Thesis notebook standard

Use exactly three phases documented in notebooks/README.md: data preparation/preprocessing, representation/training, and evaluation/analysis. Data cleaning lives in datasets/; experiment notebooks load and verify prepared versions. Cite external methods next to their use and distinguish them from this research's own design choices. Evaluate the immutable public v1 test split and PetTrace separately; never mix their galleries or use public test identities to select E3 checkpoints. Each experiment must state its objective, question, hypothesis, intervention, controls, sample, training/selection protocol, evaluation metrics, results and limitations. Distinguish recorded evidence from prospective method changes. Describe closed-set same-species retrieval, photo-level leave-one-out, descriptive EER and uncertainty limits accurately. Keep dataset preparation separate, pin immutable benchmark commits, preserve per-run reports, cite primary method/data sources, and never invent results or publication receipts.
