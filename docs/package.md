# PetTrace ML

Python tools for research on visual re-identification of individual dogs and cats.
The distribution is **`pettrace-ml`**; the Python import is **`pettrace`**.
Dataset preparation, model inference, metric learning and evaluation live in the package.
Notebooks describe and run the experiments.

## Installation

Python 3.11–3.13 is supported. Install a published version in your Python environment:

```bash
python -m pip install pettrace-ml==0.2.16
```

The default install includes plots. Colab already supplies IPython and widgets; the
`[notebooks]` extra adds them for standalone Jupyter environments. Use `[training]`
to also install PyTorch, ONNX export and pandas. For development, use `pip install -e .`.

Colab is the primary notebook target. The first cell of `datasets/data_lookup.ipynb` is:

```python
%pip install -q "pettrace-ml==0.2.12"
```

Use a fresh Colab CPU runtime. CI tests this ordinary installation against the reported
Linux x86_64 / Python 3.13.18 environment and rejects changes to its existing dependencies.
The hash-checked `requirements/data-lookup-colab.txt` profile reconstructs that CI baseline;
it is not installed over Colab. It records the supplied runtime constraints, not Google's
entire VM image. The hosted frontend and authenticated Drive mount need validation in Colab.
A plain pip install resolves compatible dependencies; it does not apply a full environment lock.

Model weights and datasets are separate artifacts and are not included in the package.
Data lookup uses CPU. GPU screening requires the separately validated E3 environment;
installing the training extra alone does not configure CUDA.

## Public methods

`lookup_duplicates(..., near_cache="/content/data-lookup/signatures", drive_root=...)`
adds conservative resized/recompressed-copy proposals before the distinct-photo minimum.
It compares unique image hashes across labels, quarantines involved labels and records each
dHash/RGB link for `quarantine_groups` / `quarantine_gallery`. Labels are never merged automatically.
Omit `near_cache` for the existing CSV-only audit. Drive restores signatures first and only
recovers originals without signatures. New signatures are saved in immutable incremental
chunks, shared across source volumes. This CPU pass still needs source bytes the first time;
it does not run YOLO, infer animal identity or guarantee detection of every edited copy.

Apply recorded human same-animal confirmations to a **completed original report**:

```python
reviewed = lookup_duplicates(report / "review.csv", report.parent,
                            identity_review="confirmed-groups.json")
```

The review must identify the original lookup fingerprint, reviewer, complete group labels,
image SHA256 values and counts. The new index retains the original CSV and records canonical
aliases in `identity-review.json`; it reuses the saved visual audit without reopening images.
Photo decisions, near-copy quarantine and minimum counts still apply. Use this new index in
preparation with a new dataset version; held-out or excluded aliases protect their whole group.
Further confirmations are cumulative against the original lookup, not chained corrected indices.

Load a refined dataset for an experiment (PetTrace 0.2.8+):

```python
from pettrace import data

dataset, provenance = data.load(
    "v2.50",
    drive_root="/content/drive/MyDrive/PetTrace",
)
photos = dataset.rows
```

The loader verifies a local version first, then restores the requested snapshot from the
mounted Drive preparation backup, or downloads its published `alessandro54/pettrace-training`
release from Hugging Face. Mount/authorize Drive beforehand; `HF_TOKEN` is needed only for
private Hub access. The default cache is `datasets/build/pettrace-training`; use `cache_dir`
to choose another local location. The returned dataset has `root`, `rows`, `summary` and
`fingerprint`; keep `provenance` with the experiment report.

The first Hub commit is saved locally and reused even if a remote tag moves. Missing files
can be recovered; corrupt hashes or unsafe paths fail validation. Drive is read-only here;
Hub downloads are cached locally. A lost Colab VM loses that local cache, while existing Drive
backups remain available. The loader never downloads raw ZIPs, runs YOLO, cleans data or trains.
An unavailable version must be prepared/published separately. Frozen experiment notebooks
retain their recorded loaders and pins; the immutable benchmark API remains in `data.versions`.

These are the supported import paths for the research workflows. Helpers whose names
start with `_` are internal. Existing experiment imports continue to work.

| Module | Methods | Purpose |
|---|---|---|
| `pettrace.data` | `load` | Load a verified refined version: local cache, mounted Drive backup, then its published Hub release. |
| `pettrace.data.lookup` | `lookup_duplicates`, `load_candidates`, `restore_lookup_review` | Audit duplicates, index pre-YOLO candidates and recover the raw inventory from Drive. |
| `pettrace.data.sources` | `discover_body_ids`, `write_review` | Build a SHA-256 inventory from locally extracted source photos and publisher labels. |
| `pettrace.data.review` | `exclusion_summary`, `quarantine_groups`, `quarantine_gallery`, `save_decisions` | Inspect filtering, review conflicting labels and record explicit decisions. |
| `pettrace.data.workflow` | `TrainingPreparation.restore`, `.stage`, `.prepare`, `.publish` | Stage indexed inputs on CPU, screen on GPU, or reuse refined data; HF publication is explicit. |
| `pettrace.data.checkpoints` | `DriveBackup.restore`, `stage_archive` | Verify and restore checkpoints; stage a source ZIP on local scratch storage. |
| `pettrace.data.versions` | `load_local_training`, `load_training`, `record_version` | Load verified local or Hub snapshots and assign immutable version names. |
| `pettrace.models.pipeline` | `Pipeline.preprocess`, `.embed` | Apply the artifact-defined preprocessing and ONNX encoder. |
| `pettrace.training.features` | `feature_cache` | Reuse frozen embeddings by image and model hashes. |
| `pettrace.training.triplet` | `Config`, `Head`, `fit`, `export_encoder` | Train the MLP projection using validation identities and export it with the frozen encoder. |
| `pettrace.evaluation.core` | `load_model`, `evaluate`, `report` | Verify registered models and calculate retrieval results. |
| `pettrace.evaluation.metrics` | `retrieval`, `pair_scores`, `eer`, `at_threshold` | Calculate same-species retrieval and pair-score metrics. |
| `pettrace.tracking` | `publish_evaluation`, `source_identity`, `save_provenance` | Publish experiment evidence and record its code/runtime identity. |
| `pettrace.experiments.e1` | `resume_publication` | Retry publication from saved evidence without repeating inference. |
| `pettrace.experiments.e3` | `load_benchmarks`, `load_published`, `publish_results`, `register_candidate` | Load separated pinned benchmarks, verify prior E3 evidence, publish both domains and register a candidate without promoting an alias. |

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

`candidates.csv` separately selects one image per hash after manual drops, whole-identity
quarantine and the distinct-photo minimum (default four). `excluded.csv` records every
noncandidate and its reason. Pending source labels remain unverified; this is not yet a
training dataset. `load_candidates(report, review_csv)` checks integrity and the matching
source inventory before returning candidates and exclusions.

Pass `candidate_index=report` to `PrepareConfig` or `TrainingPreparation` to apply this index
before YOLO. The latter requires an explicit new version, for example `version="v3.100"`.
The current review must match exactly; later decisions require a new index. Preparation
still protects the frozen benchmark, checks image bytes, runs screening and near-duplicate
curation and reapplies the photo minimum. Existing v2 versions are not changed.

For `TrainingPreparation`, run `preparation.stage()` **on CPU first**, with the pinned
screening environment. It reuses local candidate photos, recovers missing ones and matching
detections from confirmed backups, then writes compact packs under
`checkpoints/indexed/<index-sha256>/`. The completion receipt is written last. The first
staging may still copy large historical packs when needed photos are absent; it does not run
YOLO. Keep one writer and the same `work_dir` in both sessions (normally `/content`).

On GPU, `preparation.prepare()` requires that completed input and restores only its photos,
source receipts, index, frozen reference/model and compatible detections. It does not restore
the historical raw tree, stage ZIPs or rehash excluded images. All raw rows and exclusion
reasons remain in the audit; every candidate's bytes and the source membership are verified.
The CPU/GPU equivalence check and final filtering remain enforced. Compatible new detections
are checkpointed in the compact namespace for retries; refined versions are saved separately.
`restore()` reads only the requested refined version, including recovery of a partial restore.

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
