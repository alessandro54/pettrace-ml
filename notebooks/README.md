# Notebooks

Only notebooks that **train or produce a model** live here (E3+ on Colab GPU). Each one:

1. reads a pinned dataset version, `snapshot_download("alessandro54/pettrace-reid", repo_type="dataset", revision="vN")`;
2. logs its run to MLflow (`https://mlflow.chumpitaz.dev`, `MLFLOW_*` from Colab secrets) with the
   dataset version, seeds and the commit of this repo;
3. registers the result as a `pettrace-embedder` version (`@candidate`), evaluated with
   `scripts/evaluate.py` like every other version.

Exploratory notebooks are not kept (the first E1/E2 explorations are in this repo's history).
Commit notebooks with outputs stripped and never with photos: the dataset is private.
