# PetTrace ML

Python tools for research on visual re-identification of individual dogs and cats.
The distribution is **`pettrace-ml`**; the Python import is **`pettrace`**.
Dataset preparation, model inference, metric learning and evaluation live in the package.
Notebooks describe and run the experiments.

## Installation

Python 3.11–3.13 is supported. Install a published version in your Python environment:

```bash
python -m pip install 'pettrace-ml[notebooks]==0.2.5'
```

Use `pettrace-ml` for the base package, `[notebooks]` for plots and interactive review,
or `[notebooks,training]` to also install PyTorch, ONNX export and pandas.
For local development, install the checkout with `python -m pip install -e '.[notebooks]'`.

Model weights and datasets are separate artifacts and are not included in the package.
Data lookup uses CPU. GPU screening requires the separately validated E3 environment;
installing the training extra alone does not configure CUDA.

## Public methods

These are the supported import paths for the research workflows. Helpers whose names
start with `_` are internal. Existing experiment imports continue to work.

| Module | Methods | Purpose |
|---|---|---|
| `pettrace.data.lookup` | `lookup_duplicates`, `restore_lookup_review` | Audit exact duplicates and retrieve only the backed-up inventory from Drive. |
| `pettrace.data.sources` | `discover_body_ids`, `write_review` | Build a SHA-256 inventory from locally extracted source photos and publisher labels. |
| `pettrace.data.review` | `exclusion_summary`, `quarantine_groups`, `quarantine_gallery`, `save_decisions` | Inspect filtering, review conflicting labels and record explicit decisions. |
| `pettrace.data.workflow` | `TrainingPreparation.restore`, `.prepare`, `.publish` | Reuse or prepare versioned training data; Hugging Face publication is a separate explicit step. |
| `pettrace.data.checkpoints` | `DriveBackup.restore`, `stage_archive` | Verify and restore checkpoints; stage a source ZIP on local scratch storage. |
| `pettrace.data.versions` | `load_local_training`, `load_training`, `record_version` | Load verified local or Hub snapshots and assign immutable version names. |
| `pettrace.models.pipeline` | `Pipeline.preprocess`, `.embed` | Apply the artifact-defined preprocessing and ONNX encoder. |
| `pettrace.training.features` | `feature_cache` | Reuse frozen embeddings by image and model hashes. |
| `pettrace.training.triplet` | `Config`, `Head`, `fit`, `export_encoder` | Train the MLP projection using validation identities and export it with the frozen encoder. |
| `pettrace.evaluation.core` | `load_model`, `evaluate`, `report` | Verify registered models and calculate retrieval results. |
| `pettrace.evaluation.metrics` | `retrieval`, `pair_scores`, `eer`, `at_threshold` | Calculate same-species retrieval and pair-score metrics. |
| `pettrace.tracking` | `publish_evaluation`, `source_identity`, `save_provenance` | Publish experiment evidence and record its code/runtime identity. |
| `pettrace.experiments.e1` | `resume_publication` | Retry publication from saved evidence without repeating inference. |

## Duplicate audit without model inference

Given a raw `review.csv` containing source paths, labels and recorded SHA-256 hashes:

```python
import json
from pettrace.data.lookup import lookup_duplicates

report = lookup_duplicates("review.csv", "reports/data-lookup")
summary = json.loads((report / "summary.json").read_text())
print(summary["files"], summary["unique_images"], summary["duplicate_copies"])
```

This exports one representative per hash, every original label, a mapping of duplicate
copies and identity conflicts for review. It does not delete photos or merge identities.
Counts cover the supplied inventory, not unobserved photos. Recorded hashes are used
without rereading images; exact-byte duplicates do not identify every visually similar photo.

In Colab, after mounting Drive, recover the confirmed raw inventory of the selected volume:

```python
from pathlib import Path
from pettrace.data.lookup import restore_lookup_review

drive = Path("/content/drive/MyDrive/PetTrace")
review = restore_lookup_review(drive, "/content", percent=100)
report = lookup_duplicates(review, drive / "data-lookup")
```

This uses PetTrace preparation backups in `checkpoints/preparation`. It copies and verifies
the TAR block containing the CSV and restores only that file. Missing backups fail explicitly;
it does not silently change the percentage or run preparation. Reports persist in Drive.
The optional `quarantine_gallery` needs the original photos at their recorded paths.

## Experiments and reproducibility

Use the versioned
[experiment notebooks](https://github.com/alessandro54/pettrace-ml/tree/main/notebooks)
for the complete public-test and private-PetTrace protocols. Keep their galleries separate.
Retrieval is closed-set and same-species, with photo-level leave-one-out evaluation.
EER fitted and reported on the same pairs is descriptive, not an independently calibrated threshold.

All PetTrace experiment runs must publish results to
[MLflow](https://ml.pettrace.app). Low-level metric functions alone do not complete an experiment.
Model registration and serving promotion are separate from publication. Keep saved evidence
when publication fails so it can be retried. Data preparation is a separate workflow.

`pip install` resolves the declared dependency ranges; it does **not** apply `uv.lock`.
For a locked environment use the matching Git tag and `uv sync --locked`, or the experiment's
published runtime profile. Reproduction also requires the recorded data/model hashes,
configuration, seed and compatible hardware. A version pin does not guarantee identical
floating-point results across hardware.

The package code is MIT-licensed. Original datasets and model weights retain their own licenses.
See [release instructions](https://github.com/alessandro54/pettrace-ml/blob/main/docs/releases.md)
and [dataset preparation](https://github.com/alessandro54/pettrace-ml/blob/main/datasets/README.md).
