# PetTrace ML

Reusable Python tools for research on visual re-identification of individual dogs and cats.
The distribution is **`pettrace-ml`** and the Python import is **`pettrace`**.

PetTrace provides four independent layers:

1. immutable refined datasets from Hugging Face;
2. verified ONNX model artifacts and inference;
3. metric-learning training and retrieval evaluation;
4. an installable MLflow adapter; official PetTrace experiment publication remains mandatory.

The package contains methods, validation and provenance. Notebooks contain the research
question, experiment configuration, figures and interpretation.

## Installation

Python 3.11–3.13 is supported.

```bash
python -m pip install pettrace-ml
```

Install only the capabilities a workflow needs:

```bash
python -m pip install "pettrace-ml[training]"  # PyTorch, Transformers and ONNX export
python -m pip install "pettrace-ml[tracking]"  # MLflow publication and registry access
python -m pip install "pettrace-ml[notebooks]" # standalone Jupyter support
```

For a reproducible run, pin the exact version used by its notebook. `pip install` does not
apply this repository's `uv.lock`; use the matching Git tag and `uv sync --locked` when the
complete development environment must be reproduced.

## Datasets

The public v2 training corpus is the verified union of two immutable Hugging Face datasets.
It contains the audited v1 images, identities and splits plus the pinned YOLO localization
overlay. Loading never cleans data, runs YOLO or mutates the Hub repositories.

```python
from pettrace import data

training = data.training(version="v2", cache_dir="/content/pettrace-data")
print(training.fingerprint, training.summary)
```

Individual sources are available when an experiment needs them separately:

```python
dogs = data.dogs_of_the_world(version="v2")
cats = data.labeled_cats_in_the_wild(version="v2")
```

Every loader resolves the release tag to its audited 40-character commit, verifies the
manifests and hashes, and returns provenance with the materialized files. A token is optional
for public datasets and can be passed explicitly for a private mirror.

E1 and E2 derive their public test benchmark directly from these refined v2 releases. Raw
acquisition, cleaning, YOLO preparation and dataset publication are intentionally not package
APIs: those one-time workflows are preserved by immutable Hub releases and Git history.

The official zero-shot experiments are available as reproducible, mandatory-tracking commands:

```bash
pettrace run e1 --output reports/e1
pettrace run e2 --output reports/e2
```

## Models

Model releases are ordinary directories containing at least:

```text
pipeline.yaml
golden.npy
*.onnx
```

Load and verify a local artifact:

```python
from pettrace.models import load_model

model = load_model("artifacts/e4")
if not model.golden_ok:
    raise RuntimeError("Model failed its golden self-test")
embeddings = model.pipeline.embed(images)
```

Download a public or private Hugging Face model release by tag or commit, then run the same
verification. PetTrace resolves tags to immutable commits before downloading.

```python
from pettrace.models import download_model, load_model

artifact = download_model(
    "your-account/pettrace-reid-models",
    revision="v1",
    subfolder="e4",
)
model = load_model(artifact)
```

MLflow model URIs remain supported when the `tracking` extra is installed. They are a storage
adapter, not a requirement of inference or evaluation.

## Training and evaluation

The stable research primitives live in these modules:

| Module | Purpose |
|---|---|
| `pettrace.models` | Download, load and golden-verify model artifacts; run ONNX pipelines. |
| `pettrace.data` | Load pinned refined sources and the combined public v2 corpus. |
| `pettrace.training.features` | Cache frozen embeddings by image and model hashes. |
| `pettrace.training.triplet` | Train/export the E3 MLP projection head. |
| `pettrace.training.lora` | Inject, train, merge and export E4 LoRA adapters. |
| `pettrace.evaluation.core` | Evaluate a verified model under the shared retrieval protocol. |
| `pettrace.evaluation.metrics` | Rank-k, mAP, pair scores, descriptive EER and thresholds. |
| `pettrace.experiments` | Reusable orchestration and publication for the thesis experiments. |
| `pettrace.tracking` | Connect to a caller-selected MLflow server and publish saved evidence. |

Retrieval is closed-set, same-species and photo-level leave-one-out. Rank-k aggregates by the
maximum gallery-photo score for each identity; AP/mAP ranks gallery photos. EER fitted and
reported on the same pairs is descriptive, not an independently calibrated operating point.

The canonical notebooks in [`notebooks/`](../notebooks) demonstrate the complete three-phase
protocol: preparation, representation/training, and evaluation/analysis. Public v1 and PetTrace
are evaluated and published as separate domains; their galleries are never mixed.

Long-running loads and training loops render one normalized progress bar in terminals and
notebooks. Epoch bars show the current loss, validation mAP and selected best epoch; complete
records remain in `history.json`. Evaluation plots are deliberately explicit: each public plot
method returns one figure, while report orchestration only saves those named figures and tables.

| Method | Report file | Meaning |
|---|---|---|
| `confusion` | `confusion.png` | Top-1 confusion counts by identity and species. |
| `rank1_per_animal` | `rank1-per-animal.png` | Rank-1 accuracy for every identity. |
| `score_distribution` | `score-distribution.png` | Same-identity and different-identity cosine distributions. |
| `cmc` | `cmc.png` | Cumulative matching characteristic by rank. |
| `similarity_matrix` | `similarity-matrix.png` | Mean identity-to-identity cosine similarity. |
| `ranking_metrics` | `ranking-metrics.png` | Rank-1, Rank-2 and mAP by scope. |

## Tracking

The library does not hardcode a tracking server. Standard MLflow environment variables select
the destination:

```bash
export MLFLOW_TRACKING_URI=https://your-mlflow.example
export MLFLOW_TRACKING_USERNAME=...
export MLFLOW_TRACKING_PASSWORD=...
```

`tracking_context()` fails explicitly if no URI is configured or the tracking extra is absent.
Generic library calls and local CLI evaluation can run without MLflow. The official PetTrace
experiment notebooks are stricter: publication to `https://ml.pettrace.app` is mandatory, and
an experiment is complete only after both benchmark runs have been published. Publication can
be retried from saved evidence without retraining. Model registration and serving alias
promotion remain separate operations.

## Reproducibility contract

A reproducible result records all of the following:

- package version and source hash;
- dataset repository, immutable commit and fingerprint;
- model artifact hashes and golden fingerprint;
- seed, configuration, protocol and runtime versions;
- separate public and PetTrace results;
- MLflow run IDs for official experiments.

Version pins do not guarantee bit-identical floating-point values across different hardware.
Original datasets and upstream weights retain their own licenses; the package code is MIT.

Operational release and registry scripts live under `tools/` and are not part of the public
runtime API. Historical preparation implementations remain in Git history; current experiments
consume verified Hugging Face artifacts only.
