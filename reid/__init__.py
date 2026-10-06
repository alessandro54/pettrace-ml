"""Re-ID evaluation harness for PetTrace (thesis protocol), shared by notebooks, CI and the gate.

    data.load("v1")                    → a pinned `pettrace-reid` version from Hugging Face
    metrics.evaluate(E, animals, ...)  → Rank-1/5/10, mAP, per species + global
    metrics.eer(same, diff)            → EER and its threshold (calibrated on `val`)

`ml/scripts/evaluate.py` runs a registered model on a dataset version and logs it to MLflow.
"""
