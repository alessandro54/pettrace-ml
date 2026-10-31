# PetTrace ML

Visual re-identification research for **individual dogs and cats**, developed as part of a thesis at Universidad Peruana de Ciencias Aplicadas (UPC). Given a pet photograph, the research pipeline ranks candidate identities of the same species by visual similarity. The task is to retrieve **the same animal**, rather than classify its breed. Human review remains necessary to confirm a candidate match.

This repository documents the research questions, experimental methods, reusable implementation, and evidence behind PetTrace. The application and its serving infrastructure live in a separate repository. The [research notebooks](notebooks/README.md) are written in Spanish; this README provides an English overview.

## Research progression

The experiments investigate incremental changes to a shared visual retrieval pipeline. External methods are cited below; their application, sampling rules, evaluation protocol, and measured PetTrace results are decisions and findings of this research.

| Experiment | Research question | Pipeline and parameter updates | Evidence and implementation status |
|---|---|---|---|
| [E1 — Frozen representation](notebooks/e1_clip_zero_shot.ipynb) | How well does a pretrained visual encoder retrieve individual pets without domain adaptation? | Image without animal localization → CLIP ViT-B/32 → 512-dimensional, L2-normalized embedding. No parameter updates. | Reference evaluation published on PetTrace v4. |
| [E2 — Animal localization](notebooks/e2_yolo_crop_zero_shot.ipynb) | How does localizing the animal affect retrieval with the same encoder? | YOLOv8n → highest-confidence cat/dog box with a 10% margin → the same frozen CLIP encoder. The image is retained when no valid box is available. | Reference evaluation published on PetTrace v4, including a recalculated E1 baseline. |
| [E3 — Metric adaptation](notebooks/e3_mlp_head_triplet.ipynb) | Does a projection trained on public identities transfer to private PetTrace identities? | Frozen E2 → MLP 512→256→512 → L2 normalization. Only the MLP is trained with batch-hard triplet loss. | First training/evaluation completed; recorded results are below. Its saved publication output still indicates publication was disabled during that run. |
| E4 — Encoder adaptation | What additional benefit might adapting the visual encoder provide? | Proposed LoRA-based adaptation. | Future research; no completed experiment is claimed. |
| E5 — Text-assisted retrieval | Can a sighting description improve visual candidate ranking? | Proposed integration of textual information or re-ranking. | Future research; no completed experiment is claimed. |

CLIP is based on [Radford et al. (2021)](https://arxiv.org/abs/2103.00020), localization uses [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), and E3's batch-hard objective follows [Hermans, Beyer, and Leibe (2017)](https://arxiv.org/abs/1703.07737). These references motivate components; their published results are not evidence of performance on PetTrace.

## Three-phase research workflow

### Phase 1 — Data preparation and preprocessing

The shared [dataset preparation notebook](datasets/prepare_training.ipynb) lives under `datasets/`, independently of the experiment notebooks. It acquires a bounded public-source pilot, checks files, filters ambiguous scenes, removes duplicates, and creates versioned snapshots with identity-disjoint partitions.

The pipeline records:

- Source versions, archive-index hashes, acquisition budgets, selected identities, and image hashes.
- A frozen E2 detector pass: exactly one detected animal of the expected species is required. Multiple animals, no detections, unexpected species, and unreadable images are excluded.
- Known ambiguous identities, exact-image conflicts across identities, exact duplicates, and near duplicates.
- At least four distinct retained photographs per identity, followed by train/validation/test assignment by identity.
- Review decisions, exclusion reasons, audit reports, and a reproducible visual sample.

The current policy is exploratory: publisher identity labels are retained without requiring individual approval of every photograph. Technical filters and a visual sample do not verify every identity or establish independent capture sessions. The selection is bounded rather than a uniform sample of the complete sources; dog acquisition scans a limited archive-order metadata prefix.

The prepared release is private on Hugging Face and includes its images, manifests, audit records, splits, and hashes. Named versions are immutable: another snapshot requires another version. The private PetTrace benchmark is curated/exported through a separate workflow. See [dataset documentation](datasets/README.md) and the [pipeline protocol](docs/e3_data_pipeline.md).

### Phase 2 — Representation and training

Each experiment loads a prepared version and verifies the model artifacts. Experiment notebooks do not download the complete source archives or repeat dataset cleaning.

E1 and E2 perform inference with frozen pretrained components. E3 caches frozen E2 embeddings and trains only its MLP:

| E3 setting | Configuration |
|---|---|
| Optimizer | AdamW, learning rate `1e-3`, weight decay `1e-4` |
| Objective | Batch-hard triplet loss, margin `0.2` |
| Sampling | 8 identities × 4 images; same-species batches alternating cats and dogs |
| Schedule | Up to 30 epochs, 100 steps per epoch, early-stopping patience 7 |
| Seed | 42 |
| Checkpoint selection | Mean of the public validation mAP values for cats and dogs |
| Private benchmark participation | No private images or identities used for fitting or checkpoint selection |
| Execution | GPU training of the MLP; frozen ONNX inference uses CPU |

The first E3 run selected **epoch 3**, with public validation macro-species mAP **0.6603**, and stopped at epoch 10. Training loss continued to decrease without a sustained validation improvement. This is consistent with overfitting, but one run does not establish its cause.

The MLP architecture, species-alternating sampling, hyperparameters, and checkpoint rule are study-specific choices. The [batch-hard reference](https://arxiv.org/abs/1703.07737) concerns person re-identification and does not establish their effectiveness for pets.

### Phase 3 — Evaluation, analysis, and publication

The updated E1/E2/E3 notebooks evaluate two datasets separately:

| Evaluation domain | Scope | Photographs | Identities |
|---|---|---:|---:|
| Public-source benchmark | Published `pettrace-training@v1` test split, originating from Dogs of the World and LCW on Kaggle | 277 | 60: 34 cats, 26 dogs |
| PetTrace benchmark | All eligible photographs in `pettrace-reid@v4` | 305 | 31: 15 cats, 16 dogs |

Both domains use the same evaluator, with **separate galleries and separate MLflow runs**. Public train/validation identities are excluded from the public test gallery. E3 evaluates test data only after fixing its checkpoint. Once observed, the public test is no longer described as an untouched benchmark and must not be used for subsequent checkpoint selection.

The public split is **15.2% of the prepared photographs and 14.9% of the prepared identities**, not 20%. The existing v1 release is preserved rather than silently re-split. Different gallery sizes and capture conditions mean that public/private score differences are descriptive, not causal estimates of a dataset effect.

The reports contain confusion matrices, CMC curves, Rank-1/Rank-2/mAP charts, score distributions, similarity matrices, and per-identity analysis. E3 also includes training/validation curves. Long identity strings receive compact plot labels with a reversible `identity-labels.json` mapping. Figures are calculated by this project rather than copied from the literature.

**The new public-versus-private comparisons are implemented but have not been rerun yet.** Historical metrics below do not include public-test results or a newly measured Rank-2 value.

## Datasets and version identities

### Public training release

[PetTrace Training v1](https://huggingface.co/datasets/alessandro54/pettrace-training/tree/v1) combines a cleaned pilot from:

- [Dogs of the World, version 1](https://www.kaggle.com/datasets/lextoumbourou/dogs-world): public dog identities.
- [LCW — Labeled Cats in the Wild, version 1](https://www.kaggle.com/datasets/dseidli/lcwlabeled-cats-in-the-wild): public cat identities.

After filtering, v1 contains **1,826 photographs of 402 identities**: 229 cats and 173 dogs. The complete prepared snapshot is approximately **212 MB**.

| Partition | Photographs | Identities | Purpose |
|---|---:|---:|---|
| Train | 1,276 | 282 | E3 parameter updates |
| Validation | 273 | 60 | E3 checkpoint selection |
| Test | 277 | 60 | Shared public evaluation for the upcoming E1/E2/E3 runs |

Publication commit: `dd7a00cb8b3638759ffa0216251e62d334b5bf10`.

Prepared fingerprint: `38ef098d26afa792a1712f9e37487d143f60a9a3bc647a5fdde5505f790f8c7d`.

The initial candidate pilot sampled 300 identities per species and retained 402 identities after cleaning. The source archives themselves contain 313,671 dog image entries and 380,923 cat image entries in the counted source directories. Those counts were verified against the complete ZIP indexes; their versions, scopes, and hashes are recorded in [source_capacity_v1.json](datasets/source_capacity_v1.json).

### Dataset coverage metrics

Dataset coverage measures experiment scale against a fixed source denominator, independently of the train/validation/test split:

```text
dataset.train_pct = 100 × training photographs / original source image entries
```

For the first E3 run:

| Metric | Value |
|---|---:|
| `dataset.source_photos` | 694,594 |
| `dataset.train_photos` | 1,276 |
| `dataset.train_pct` | 0.1837% |
| `dataset.train_val_photos` | 1,549 |
| `dataset.train_val_pct` | 0.2230% |

Per-species coverage is also recorded. The denominator counts source images **before cleaning**; it is not a verified maximum of usable re-identification images or a percentage of available identities. Increasing this percentage does not imply proportional performance gains. E1/E2 do not optimize parameters, so their evaluation samples must not be described as training coverage.

Historical E1/E2 v3 reports were produced with onnxruntime 1.20.1 and Pillow 11.0 and re-checked on 2026-10-10 with onnxruntime 1.31.0, Pillow 12.3.0 and mlflow-skinny 3.17.0: Rank-1/5/10, mAP, EER and threshold were unchanged for both; the largest difference in `summary.json` was 5.4e-07, with pair scores within their 1e-05 rounding.

### Private PetTrace benchmark

[PetTrace ReID v4](https://huggingface.co/datasets/alessandro54/pettrace-reid/tree/v4) is a private evaluation snapshot pinned to commit `8b12db0dce41040a821f9792bea5b352342263ce`.

The notebooks verify the tag and download that exact commit. They check exact-image overlap between the prepared public source and PetTrace before evaluation. Identity labels are pseudonymous in the research reports; private photographs and credentials are not included in this code repository.

## Evaluation protocol and interpretation

The evaluation is **closed-set retrieval**: an eligible query has another image of its own identity in the gallery and at least one rival identity of the same annotated species.

- Each photograph is a query; its own image is excluded from its gallery.
- Candidate identities are restricted to the query's annotated species.
- Cosine similarity is computed through the dot product of L2-normalized embeddings.
- Rank-k uses the maximum photograph score for each candidate identity. Rank-1 and Rank-2 indicate whether the correct identity occurs in the first one or two positions.
- AP ranks individual photographs; mAP averages AP over eligible queries. Global query averages and macro averages over identities are distinguished.
- The reported EER is fitted and measured on the same evaluation pairs. It is descriptive and does not validate a production decision threshold.

CMC and AP/mAP have established use in re-identification, including [Zheng et al. (2015)](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html). Our same-species gallery and maximum-score identity aggregation are study-specific adaptations, not a reproduction of that paper's camera-based protocol. The interpretation of matching errors is contextualized by [Fiumara/NIST (2022)](https://www.nist.gov/blogs/taking-measure/tale-two-errors-measuring-biometric-algorithms); NIST does not validate this implementation's threshold.

The evaluator's `zero-shot` label selects leave-one-out evaluation over all supplied photographs. It does not mean text-prompt classification. E3 is trained on public identities; its private evaluation identities remain excluded from fitting and checkpoint selection.

These experiments do not establish open-set performance, cross-session generalization, statistical significance, or deployment accuracy. PetTrace v4 is small and was already observed in E1/E2. Photographs of the same animal are dependent observations; E3 currently has one training seed. The next research steps include more identities and capture sessions, repeated seeds, uncertainty estimates by identity, and an additional independent benchmark for future model selection.

## Recorded results

### PetTrace v4 reference evaluations

E1/E2/E3 were compared on the same 305 private photographs. E2 recalculates E1, and E3 recalculates both baselines on matching queries and galleries.

| Global metric | E1 | E2 | E3 |
|---|---:|---:|---:|
| Rank-1 | 65.90% | 67.87% | 72.79% |
| Rank-5 | 94.10% | 92.13% | 96.39% |
| mAP | 0.3751 | 0.3960 | 0.5682 |
| Descriptive EER | 36.23% | 33.83% | 16.76% |

E2 improves Rank-1, mAP, and descriptive EER over E1, while Rank-5 decreases. E3 improves the reported global indicators over both baselines. Compared with E2, its Rank-1 increases by **4.92 percentage points**; the saved full-precision mAP difference is approximately **+0.1721**. Deltas computed from rounded table entries can differ in the final decimal place.

The Rank-1 change is uneven across species: E2→E3 changes from 67.31% to **76.28%** for cats and from 68.46% to **69.13%** for dogs. Dog mAP nevertheless increases from 0.3971 to **0.5170**. These observations support further investigation of metric adaptation rather than a claim that every identity or species benefits equally.

Published references:

- [E1 v4 — MLflow](https://ml.pettrace.app/#/experiments/3/runs/8601747ecd384d948d429b3b490449c9).
- [E2 v4 — MLflow](https://ml.pettrace.app/#/experiments/3/runs/9117658f1945404ca9d1058257bf97f0).

The first E3 evidence comes from the saved execution `20261030T153523Z-e258882a`. Its last saved publication output says publication was disabled at that time. **Changing the notebook policy does not retroactively publish that run.** Its actual publication must be confirmed through the session's `publication.json` receipt; no published E3 run link is invented here.

### Historical PetTrace v3 evaluations

Earlier E1/E2 evaluations used 128 photographs of 14 identities, a different sample. They are retained as historical evidence and are not merged with v4 comparisons.

| Global metric | E1 v3 | E2 v3 |
|---|---:|---:|
| Rank-1 | 73.4% | 70.3% |
| Rank-5 | 98.4% | 99.2% |
| mAP | 0.509 | 0.527 |
| Descriptive EER | 39.2% | 35.0% |

Historical runs: [E1 v3](https://ml.pettrace.app/#/experiments/3/runs/af8a88b1e2a6434c9eeb31b62c9010e4), code `f80d1d6`; [E2 v3](https://ml.pettrace.app/#/experiments/3/runs/e0d3e49aec554a4588efa3bce99955e1), code `c9dd2b9`. Dataset commit: `354763693e9105eb953c559620037dd98995326b`.

## MLflow and evidence preservation

**Every current and future experiment must publish its results to [PetTrace MLflow](https://ml.pettrace.app).** Publication cannot be disabled through an environment variable or `--no-mlflow`. Reading existing runs requires read access; executing and publishing an experiment requires write access.

The canonical notebook evaluation experiment is `pettrace-reid-eval`. Each new notebook execution publishes one public-test run and one private-benchmark run, with dataset-specific inputs and linked receipts. Model version registration is separate from result publication; E3 retains its explicit `REGISTER_E3` setting. Publishing results does not promote a serving alias.

Each complete reproduction creates `reports/eN_v4/<timestamp-id>/` and preserves, as applicable:

- `experiment-design.json`, dataset commits/fingerprints, model hashes, and runtime/code provenance.
- `summary.json`, `scores.json`, `analysis.json`, generated `findings.md`, and figures.
- `public_test/` with the public selection, evaluation, runtime provenance, and publication receipt.
- `benchmark-comparison.json` and `benchmark-comparison.png`, keeping public/private galleries separate.
- E3 manifests/audits, training configuration, history/checkpoint, learning curves, dataset coverage, and exported ONNX artifacts.
- A root `publication.json` linking the private `run_id` and `public_run_id` after both evaluations are published.

If an upload fails, retain the local evidence and retry publication without retraining. A failed or incomplete upload is not a completed publication. Keep the original session variables and report path when retrying; do not rerun configuration cells that create a new run directory.

[`experiments.yaml`](experiments.yaml) is a small, versioned index of MLflow runs, not a second results store. `make manifest` refreshes it through a read-only MLflow query. Inspect run status alongside metrics: the index may include failed or incomplete runs. Reports, source data, wheels, and temporary output are not committed as generated research artifacts. Existing notebook outputs document historical executions; source edits do not create new experimental evidence.

## Reproduction

### Local development

```bash
git clone https://github.com/alessandro54/pettrace-ml.git
cd pettrace-ml
cp .env.example .env
uv sync --locked --group dev --group figures --group notebooks --extra notebooks --extra training
make test
make lint
```

Set credentials directly in the local `.env`, runtime environment, Colab Secrets, or hidden prompts. Do not put tokens or passwords in notebook source. A local `.env` is not automatically transferred to Colab. Dataset access needs `HF_TOKEN`; model acquisition and result publication need the configured MLflow credentials.

The distribution is `pettrace-ml`, while the import package is `pettrace`. The reference interpreter is Python 3.11.15; the locked training/notebook environment supports Python 3.11–3.12. [`uv.lock`](uv.lock) fixes dependency resolution. See [versioned releases](docs/releases.md) for installing the published wheel together with its hashed dependencies.

### Google Colab / VS Code

1. Open the selected E1, E2, or E3 notebook. Use a GPU runtime for E3; a T4 is sufficient for the current MLP configuration.
2. Run the first installation cell in a fresh runtime. It installs **v0.2.0** and all dependencies from the release's hashed requirements file, without resolving GitHub `main`.
3. Restart the kernel, then continue from the package-verification cell. It verifies the package version and displays the embedded source commit and lockfile hash.
4. Keep Python, CUDA/GPU and effective runtime versions with the results. Hosted Colab runtimes can differ from the reference environment; locks do not guarantee bit-identical GPU results.
5. Execute the three phases and publish both evaluation domains to MLflow. Do not reinstall packages during an active run.
6. Save the executed notebook. From VS Code, the remote kernel cannot read the notebook file open on the local machine; use the attachment tool after saving it locally when applicable.

Package tests validate implementation, not empirical E1/E2/E3 results. Preserved notebook outputs belong to their recorded runtime and code, not automatically to the new release.

### Docker and supporting tools

```bash
make image
make notebook NB=notebooks/e1_clip_zero_shot.ipynb
make notebook NB=notebooks/e2_yolo_crop_zero_shot.ipynb
make evaluate DATASET=v4 MODEL=models:/pettrace-embedder/2 PROTOCOL=zero-shot REPORT=reports/e2_cli_v4
make verify-model MODEL=models:/pettrace-embedder/2
make manifest
```

Use the GPU Colab workflow for E3; the commands above do not configure GPU access for Docker. The generic evaluation CLI evaluates the supplied dataset; the notebooks implement the two-domain research comparison.

`make prepare-e3` runs the independent data-preparation CLI through Docker. Preparation is not required again when consuming an existing verified release. Its optional Hugging Face upload is distinct from mandatory MLflow experiment publication.

The encoder/detector ONNX files are verified by SHA-256 and a synthetic golden set. E1's encoder hash is `4773e649274ca8774c4a63d45f0438fe598ee7ba67e65def97537aa0eca9a576`; E2's detector hash is `da8f9b9ebea5157cdcefa7bea9fd782249ebdf119f917c07d64e5b41942b9c9d`. Model version numbers identify registry entries on a specific server; artifact hashes identify the evaluated files. Registration/export tools are available in [`tools/`](tools/README.md).

## Repository layout

```text
datasets/          Shared preparation notebook, dataset documentation, source-capacity reference
notebooks/         Three-phase E1/E2/E3 experiment reports and research index
src/pettrace/data/        Acquisition, curation, screening, preparation and versioned snapshots
src/pettrace/models/      ONNX inference, golden checks and CLIP reference
src/pettrace/training/    Frozen feature cache and metric learning/export
src/pettrace/evaluation/  Protocols, metrics, figures and public/private research comparisons
src/pettrace/experiments/ Explicitly separate legacy orchestration
src/pettrace/tracking.py  Publication and source/runtime provenance
tests/             Implementation/protocol checks, not empirical research results
tools/             Model registration, verification, notebook attachment, run-index synchronization
docs/              Execution guides, pipeline protocol, references.bib
experiments.yaml   Small Git-tracked index of MLflow runs
reports/           Ignored working reports; complete published evidence belongs in MLflow
```

The package's inference/export reference modules mirror the corresponding vision-service modules in the application repository. Golden-set checks guard against incompatible model artifacts. Preparation logic remains outside the experimental notebooks, while common evaluation and visualization code is reused across conditions.

## Data access, attribution, and licenses

Private benchmark photographs and credentials are not stored in this repository. Research reports use identity codes; reversible plot-label mappings refer to those identity codes, not owner identities. Keep private dataset access and execution artifacts within the authorized research workflow.

The source receipts retain publisher-declared dataset licensing information. Consult the original dataset pages and upstream model licenses before redistributing their assets; this repository's code license does not grant rights to every dataset or model artifact. Project code is licensed under [MIT](LICENSE).

External methods and datasets are cited near their use in the notebooks. Reusable bibliography entries are maintained in [`docs/references.bib`](docs/references.bib). PetTrace's measured results, generated figures, filtering rules, and experimental choices are identified as this research's own contributions and limitations.
