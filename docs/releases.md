# Package releases

## 0.2.40 — deterministic E5 morphology reranking

- Freezes E4 seed 46 with `single` representation and maximum identity aggregation.
- Adds identity-blind, closed-vocabulary Qwen3-VL extraction from the frozen YOLO crops and stores
  the result as a hashed Parquet artifact.
- Calibrates the visual ambiguity, text weight and text-advantage gates on public validation only.
- Evaluates E4, ungated E5 and double-gated E5 once on separate Public v2 and PetTrace v6 tests,
  with mandatory MLflow publication and no serving-alias promotion.

## 0.2.39 — bounded cosine diagnostics

- Clips cosine matrices to their mathematical `[-1, 1]` range after floating-point operations.
- Reports the maximum different-identity cosine with six decimals so it is not mistaken for a percentage.
- Pins the E4 reproduction to the corrected PetTrace v6 benchmark while preserving its prior v5 outputs as historical evidence.

## 0.2.38 — E4 reproduction on current benchmarks

- Pins E4 to the validation-selected E3 seed 46, registered model 20 and verified head hash.
- Moves the E4 reproduction to Public v2 and PetTrace v5 while preserving historical notebook outputs.
- Compares frozen controls by stable photo identity instead of runtime-specific cache paths.

## 0.2.37 — Stable E3 frozen-control coverage

- Compares frozen E1/E2 coverage by immutable source/image name, identity, species and gallery size.
- Ignores runtime-specific cache roots and query indices while still rejecting changed benchmark photos.

## 0.2.36 — E3 on refined public v2 and PetTrace v5

- Updates the canonical E3 study to train on the complete refined v2 corpus and evaluate Public v2
  and PetTrace v5 separately.
- Pins the published E1/E2 controls for those exact benchmarks and reuses the already-loaded public
  corpus instead of downloading it again inside the benchmark helper.
- Installs the complete training, notebook and MLflow extras from PyPI in a fresh Colab T4 runtime.

## 0.2.35 — composite E2 audit paths

- Records stable image names when the refined public benchmark spans multiple cache roots.
- Prevents E2 detector auditing from treating the process working directory as the dataset root.

## 0.2.34 — E2 refined paired benchmark

- Replaces the duplicated E2 notebook implementation with a tested headless runner and CLI command.
- Evaluates E2 and a paired E1 control on the same refined public v2 and PetTrace v5 queries.
- Audits every detector decision, verifies fallback equivalence, and preserves qualitative inputs.
- Publishes both E2 domains mandatorily and supports publication retry without repeating inference.
- Preserves the executed public v1/PetTrace v4 notebook as clearly labeled historical evidence.

## 0.2.33 — E1 refined public benchmark

- Evaluates E1 on the immutable test identities from both public refined v2 sources.
- Records both Hugging Face commits in the composite benchmark receipt and keeps PetTrace v5 separate.
- Preserves the pilot public v1 results as historical evidence instead of relabeling them.

## 0.2.32 — E1 headless notebook

- Replaces the duplicated E1 notebook implementation with the tested headless experiment runner.
- Keeps the three thesis phases visible while loading immutable benchmarks, evaluating, rendering figures
  and publishing both domains through the package.
- Installs the complete notebook and MLflow extras from PyPI and reads credentials only from Colab Secrets.
- Records the current PetTrace v5 evidence and links the executed historical v4 notebook immutably.

## 0.2.31 — complete benchmark metadata

- Loads a benchmark without a `split` column as one complete test set.
- Preserves explicit train, validation and test assignments when they are present.
- Enables the headless E1 runner to evaluate the immutable PetTrace v5 release.

## 0.2.30 — current Hugging Face Hub compatibility

- Removes the retired `configure_http_backend` integration and relies on the current Hub client.
- Keeps E1/E3 benchmark loading, immutable revision checks and MLflow publication unchanged.
- Updates the E1/E2 notebook setup cells to install the compatible release from PyPI.

## 0.2.29 — OSS package boundary and archived E4 diagnostic

- Removes the one-off validation notebook and diagnostic-only package API after successful MLflow publication.
- Preserves the protocol, results, selection decision and immutable MLflow receipt in the thesis documentation.
- Freezes E4 seed 46, epoch 9 and registered candidate 19 by run IDs and ONNX hashes for E5.
- Leaves the frozen E4 model and standard evaluator unchanged.
- Moves MLflow into the optional `tracking` extra and configures it exclusively through standard
  MLflow environment variables, so local evaluation no longer depends on the author's server.
- Makes the CLI require an explicit model; unlabeled diagnostics can stay local, while a named
  experiment requires configured MLflow publication.
- Adds an OSS-facing Hugging Face model loader that resolves releases to immutable commits and
  requires a complete ONNX pipeline plus golden self-test artifact.
- Separates reusable public APIs from historical Drive preparation compatibility in the package
  documentation without deleting the code needed to audit old runs.
- Rejects an untracked CLI invocation carrying an experiment label while preserving local,
  unlabeled evaluation as a reusable diagnostic.

## 0.2.28 — E4 validation inference ablation

- Adds a validation-only diagnostic for deterministic horizontal-flip TTA and identity aggregation.
- Loads the frozen seed-46 E4 candidate from MLflow without retraining or evaluating either test benchmark.
- Publishes the six-condition diagnostic unconditionally to MLflow with an idempotent receipt.
- Keeps the E4 runtime free of Google Drive.
- Removes Google Drive from E4 training, publication and candidate registration.
- Keeps Hugging Face as the immutable dataset source and MLflow as the published evidence and model store.
- Retains local evidence for publication retries while the Colab runtime remains active.
- Records the completed seed-46 E4 run and candidate 19 without promoting a serving alias.

## 0.2.27 — E4 on the selected E3 control

- Pins E4 to the E3 seed-46 winner selected only by validation, registered as model version 18.
- Runs LoRA with seed 46 and publishes it as a distinct v2 experiment candidate without moving an alias.
- Keeps the previous seed-42 E4 run as historical evidence rather than silently changing its control.

## 0.2.26 — canonical E3 notebook

- Consolidates E3 into one thesis-facing 100% experiment notebook backed by the immutable Hugging Face v2 corpus.
- Removes Drive and the historical 20%/50% launchers from the experiment workflow; their published MLflow runs remain unchanged.
- Keeps Hub progress quiet and lets notebook callbacks render compact epoch updates without duplicate raw dictionaries.
- Runs prospective seeds 45–49, selects the winner only by public validation mAP before test evaluation, and publishes both test domains for every seed.
- Registers only the selected candidate and records an `e3-handoff.json` receipt for E4/E5 without promoting a serving alias.

## 0.2.25 — public v2 training corpus

- Adds `pettrace.data.training(version="v2")` for the verified union of the two public refined sources.
- Pins both Hugging Face v2 commits and proves their images, labels, splits and preparation fingerprint match v4.100.
- Makes E4 consume those public shards and YOLO boxes directly.
- Preserves mandatory separate MLflow publication for the public v1 and PetTrace benchmarks.

## 0.2.24 — matching Colab CUDA plugins

- Selects the official, hash-pinned ONNX Runtime CUDA 12 or CUDA 13 plugin that matches PyTorch.
- Requires CUDA 12.8+ or CUDA 13.x and keeps their inference caches separate.
- Prints the observed PyTorch and CUDA versions in the YOLO v2 notebook setup.

## 0.2.23 — Colab CUDA 12 compatibility

- Accepts Colab PyTorch builds using any CUDA 12.x minor release instead of requiring CUDA 12.8.
- Keeps the pinned CUDA provider, graph-assignment check and CPU/GPU equivalence gate unchanged.

## 0.2.22 — explicit Colab Secrets

- Experiment and preparation notebooks now load credentials directly with `google.colab.userdata`.
- The package no longer prompts for or reads credentials; it consumes the runtime environment only.
- YOLO v2 setup validates that every Colab secret is a non-empty text value before data access.

## 0.2.21 — immutable YOLO localization overlays

- Adds a Colab data-preparation notebook that derives `v2` from the two immutable public `v1` releases.
- Reuses the full-frame images, identities and splits; only a compact, hash-verified YOLO box overlay is added.
- Requires the pinned E2 detector and a CPU/GPU equivalence gate before bulk screening.
- Keeps resumable inference caches on Drive while Hugging Face remains the canonical public dataset source.
- Updates each dataset card in the same verified publication step; the package itself never mutates a repository.
- Adds `pettrace.data.dogs_of_the_world()` and `labeled_cats_in_the_wild()` as pinned, cached Hugging Face loaders.

## 0.2.20 — visible benchmark loading

- Shows the immutable benchmark load as five explicit stages with a final duration and photo counts.
- Restores Hugging Face's native download progress in notebooks.
- Retries temporary Hub 429 responses using the service-provided wait instead of failing the experiment.
- Standardizes package logs, generated reports, figures, and review controls in English; notebook prose remains Spanish.
- Removes dataset mutation from the package and preparation notebook; a tested one-shot tool stages
  `pettrace-training@v4.100` LCW as the public, Apache-2.0 `lcw-refined@v1` release.

## 0.2.19 — linear Colab installation

- Accepts Colab's compatible preloaded Pillow instead of replacing it inside the running kernel.
- Prevents mixed in-memory/on-disk Pillow modules when E4 installs directly from PyPI.
- Keeps the exact development and release resolution in `uv.lock`.

## 0.2.18 — E4 LoRA experiment

- Adds the fixed E4 LoRA training/export path on top of the published E3-100 head.
- Reuses verified YOLO boxes from `pettrace-training@v4.100`; E4 does not repeat data cleaning or screening.
- Adds the three-phase E4 Colab notebook with mandatory dual-domain MLflow publication.
- Adds `transformers==4.47.1` to the training extra required by the pinned CLIP source model.

## 0.2.17 — readable zero-shot notebooks

E1 and E2 keep their research design, metrics, figures and interpretation visible,
while the shared qualitative-retrieval and mandatory MLflow publication mechanics now
live in the package. Both notebooks install the pinned PyPI package directly; their
executed historical outputs remain unchanged.

## 0.2.16 — bounded-memory E3 preparation

- Removes the redundant per-photo copy from `prepared.json`; `review.csv` and its SHA-256 remain the canonical input inventory.
- Streams exclusion rows and counters instead of retaining the complete exclusion table in memory.
- Pins E3 notebooks to the published patch for large `v4.100` preparation.

## 0.2.15 — readable E3 experiment notebooks

E3 notebooks keep their research choices visible—dataset pins, `Config`, training,
metrics, figures and interpretation—while shared operational code now lives in
`pettrace.experiments.e3`. The package verifies previously published models/results,
loads pinned benchmarks with leakage guards, publishes public v1 and PetTrace
separately with identical training evidence, and registers an idempotent candidate
without promoting a serving alias. Publication has no disable flag and can be retried
from saved reports without retraining.

The notebooks install one pinned PyPI package command; no requirements file or wheel
build is embedded. Existing executed outputs remain as historical evidence and do not
claim a new experiment. Training, data selection and evaluation methods are unchanged.

## 0.2.14 — compact inputs before GPU screening

Supersedes the GitHub-only 0.2.13 build, whose PyPI publication was blocked by a notebook
unit fixture expecting its older pinned installation. The fixture now emulates that installation
explicitly; the actual wheel/install checks remain separate.

`TrainingPreparation.stage()` runs on CPU and packages only verified indexed candidates,
compatible detection caches and the frozen reference/model. GPU preparation requires its
completed receipt and never falls back to full historical restoration or ZIP acquisition.
Raw provenance and exclusions remain recorded, while excluded image bytes are unnecessary.
YOLO settings, identity protection and final curation are unchanged. Tests compare retained
images and splits with the full-input path after simulating runtime loss; no production speedup
is claimed without a measured run. Named refined restoration now selects only that version.

## 0.2.12 — apply human identity reviews without rescanning images

`lookup_duplicates(..., identity_review=...)` validates explicit same-animal decisions
against complete quarantine groups in the original report. It creates a new candidate index,
keeps original labels in the unchanged review CSV and records the alias map and evidence.
Manual drops, minimum-photo requirements and unresolved near-copy quarantine remain enforced.
The saved visual audit is reused without downloading or decoding images or running YOLO.

Preparation carries confirmed aliases into its manifest and archives their evidence.
Any held-out identity/image or excluded label protects the entire merged identity.
Existing prepared versions and experiment results remain unchanged. These confirmations
do not approve photo quality, distinct capture sessions or final training eligibility.

## 0.2.11 — restore original LCW image folders

Near-copy lookup now accepts the original LCW `CatFullDataset/<individual-ID>/<image>`
layout in Drive backups. Version 0.2.10 incorrectly required the flat `images/` layout
used by Dogs of the World and stopped before restoring LCW originals. Source-root and
identity-folder validation remain enforced; image comparison criteria are unchanged.

## 0.2.10 — review groups for resized and recompressed copies

Data lookup can compare all unique recorded images before the photo-count minimum, using
the existing dHash/aspect/RGB criteria across labels. It records proposed links and scores,
quarantines whole involved labels and groups cases for visual review. No identity is merged
or photo deleted. The default library call remains an exact-hash-only inventory audit.

New image signatures are verified against the source SHA256 and stored in immutable chunks.
Optional Drive checkpoints restore signatures first, then only one original per uncached
hash. Restarts reuse completed signatures across source volumes. The notebook enables this
CPU pass; its old outputs remain explicitly labeled as the earlier exact-only audit.
The first pass still transfers/decodes originals; full-volume speed and recall are unmeasured.
Recompressed/cropped/edited images may be missed and all proposed links need human review.

## 0.2.9 — cleaning index before YOLO

Data lookup now exports a candidate index alongside the full raw duplicate audit. It respects
manual drops, quarantines conflicting identity labels, retains one copy per hash and requires
four distinct photos per candidate identity by default. Every excluded row keeps a reason;
original files and labels are untouched. Candidates are not yet training-approved.

Preparation can consume this verified index before YOLO and archive its provenance. The
index must match the current review and minimum-photo rule; public v1 validation/test remain
reserved. Indexed workflows require a new dataset version. Existing experiment pins and
recorded outputs are preserved; no real preparation or experiment is run by this release.

## 0.2.8 — cached refined dataset loading

`pettrace.data.load("v2.50", drive_root=...)` returns a verified refined dataset and its
provenance. It reuses local data, restores only the requested Drive snapshot or downloads its
published Hub version. The first Hub commit is pinned in the local cache; missing files can
be recovered from that same source. Corrupt data, unsafe paths and unavailable backups fail
explicitly instead of starting preprocessing or switching versions.

Drive mounting and authentication stay with the caller. Loading reads existing Drive backups
and caches Hub downloads locally; it does not publish, run YOLO or train. Frozen notebooks and
recorded results are unchanged. CI exercises the loader in the reported Colab dependency profile.

## 0.2.7 — direct package installation

`pip install pettrace-ml==0.2.7` includes plotting dependencies. Colab's existing IPython and
widgets are reused; the `[notebooks]` extra remains available for standalone Jupyter.
Data lookup now has a one-line installation cell. No installer, Colab-specific extra or
kernel replacement is needed.

CI performs a normal package installation against the recorded Colab environment and rejects
changes to its dependencies. The reference profile also incorporates the additional host
constraints reported by pip (including dateutil, decorator, protobuf and fsspec). That profile
is for CI, not for replacing Colab's preinstalled libraries. Existing notebook outputs and
frozen experiment versions are preserved. No research result changes.

## 0.2.6 — Colab dependency compatibility

The notebook extra now accepts Colab's IPython 7.34 requirement and no longer depends on pip.
The data lookup installation includes the installed `google-colab` version in the same
resolution, so its dependency pins are respected, including when repairing the IPython
upgrade introduced by 0.2.5. Saved notebook outputs and the frozen experiment pins are preserved.
Release checks install the notebook extra with `ipython==7.34.0` on Python 3.11–3.13.

## 0.2.5 — first PyPI publication

Published on [PyPI](https://pypi.org/project/pettrace-ml/0.2.5/) on 2026-11-02 through
[Trusted Publishing](https://github.com/alessandro54/pettrace-ml/actions/runs/41510819962).
The release passed 247 tests and fresh installation checks on Linux Python 3.11, 3.12 and 3.13.
Both PyPI downloads were verified against the GitHub build hashes. The distribution name is
`pettrace-ml`; imports remain under `pettrace`. See [the public methods](package.md).

This version includes the resumable preparation/training work, raw duplicate audits,
identity-conflict galleries, and selective recovery of raw inventories from Drive backups.
These are package capabilities; no experiment is rerun or new result claimed by this release.
E1, E2 and the existing E3 notebooks keep their recorded package/runtime pins.

Starting with `v0.2.3`, a release distributes the **pettrace-ml package**:

- `pettrace_ml-<version>-py3-none-any.whl`: the installable package.
- `pettrace_ml-<version>.tar.gz`: the Python source distribution.
- `SHA256SUMS`: checksums for those two files.

GitHub also provides two automatic repository archives. Documentation, notebooks, launchers and
the environment lockfile remain in the repository at the matching version tag. Build provenance
is embedded in the package; the source distribution also includes `uv.lock` and `.python-version`.

The E1 launcher remains pinned to its verified `v0.2.2` environment. That release's assets remain
available for existing users. See [E1 execution](e1_colab.md). The packaging change does not change
the experiment protocol or establish new experimental results.

## Install a release

To install the package, use a fresh virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install pettrace-ml==0.2.10
```

The matching wheel and source distribution are also available in the
[GitHub release](https://github.com/alessandro54/pettrace-ml/releases/tag/v0.2.10).

This resolves dependencies within the package's declared ranges. Installing a wheel does **not**
apply `uv.lock`. For the locked development/training environment, use uv **0.11.22** and the
matching repository tag:

```bash
git clone --branch v0.2.10 --depth 1 https://github.com/alessandro54/pettrace-ml.git
cd pettrace-ml
uv python install 3.11.15
uv sync --locked --python 3.11.15 --group dev --group notebooks --extra notebooks --extra training
```

For E1, follow its interactive notebook setup below; installing the latest package alone does
not reproduce an experiment's recorded environment, data and model.

To verify downloaded release assets, download `SHA256SUMS` and the listed assets into one
directory, then run `sha256sum --check SHA256SUMS` (macOS: `shasum -a 256 -c SHA256SUMS`).

## Colab and Jupyter

Current notebooks install an exact published version directly from PyPI:

```python
%pip install -q "pettrace-ml==0.2.17"
```

CI validates the ordinary installation, package tests and notebook structure. Colab's hosted
image and authenticated Drive mount remain external and are recorded by each experiment's
`runtime.json`; an exact package version fixes PetTrace code, not Google's entire VM image.

E1 and E2 install `pettrace-ml[notebooks]==0.2.17`; E3 and preparation remain pinned to
`pettrace-ml[training,notebooks]==0.2.16` until their next recorded release. Research cells
display results directly (see [E1 execution](e1_colab.md),
[E2 execution](e2_colab.md) and [E3 execution](e3_colab.md)). Historical releases retain their
original requirements assets and the Python 3.11.15 E1 reference launcher.
Do not replace packages in the middle of a training run. A new experiment must create a new
report directory; retry publication in the original session without retraining.

Colab's hosted Python, GPU and NVIDIA driver are managed externally. Its environment may
contain additional packages. E1's launcher fixes Python 3.11.15; E2 and data preparation support Python 3.11–3.13;
The current E3 Colab profile requires Linux/Python 3.13; the historical repository export profile remains separate. Use the reference interpreter in a dedicated environment for
closer reproduction. Do not silently upgrade dependencies to accommodate a different runtime.

The locked PyTorch 2.5.1 Linux distribution uses CUDA 12.4 runtime dependencies; macOS uses
its platform-specific wheel. A compatible NVIDIA driver and GPU are still required for E3.
The lockfile fixes the package resolution per platform; it cannot make different hardware
produce bit-identical floating-point results. `runtime.json` records Python, platform,
package versions, CUDA/cuDNN and GPU names. Preserve the original runtime, seed, dataset
commits, model hashes and checkpoint alongside every published result.

## Package layout

```text
src/pettrace/
  data/          # benchmark loading, acquisition, curation, screening, preparation, versions
  models/        # ONNX inference, golden checks, CLIP export reference
  training/      # frozen feature cache and triplet learning/export
  evaluation/    # shared evaluator, metrics, figures and research comparisons
  experiments/   # isolated E1 execution and separate legacy orchestration protocol
  tracking.py    # publication and runtime/source provenance
  cli.py         # evaluation command
```

Imports changed in 0.2.1; the repository notebooks and tools use the new paths. Install and
restart the kernel as one migration. The legacy runner is not the notebook's two-domain
protocol; moving it does not make their results interchangeable.

## Build and publish

From a clean, committed checkout with uv **0.11.22**:

```bash
uv sync --locked --group dev --group notebooks --extra notebooks --extra training
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check .
uv run --no-sync python tools/build_release.py
```

The builder creates `dist/v0.2.10/` with the wheel, source distribution and `SHA256SUMS`.
It refuses to replace an existing release directory. The package embeds its version,
source commit, lockfile hash, reference Python and build-tool version in `pettrace/_build.json`;
`source_identity()` hashes every Python module, including nested packages.

Publish those files under the matching Git tag after checking the installed wheel. Never
replace assets or move a published version tag; make a new version for any correction.
The source distribution intentionally excludes datasets, credentials, reports and notebooks.
GitHub's automatically generated repository archives are separate from that distribution.

### One-time PyPI setup

Create a personal [PyPI account](https://pypi.org/account/register/), verify its email and enable
two-factor authentication. In [Publishing](https://pypi.org/manage/account/publishing/), add a
pending GitHub publisher with these exact values:

| Field | Value |
|---|---|
| PyPI project name | `pettrace-ml` |
| GitHub owner | `alessandro54` |
| Repository | `pettrace-ml` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

The pending publisher creates the PyPI project on its first successful upload; configuring it
does not reserve the name. This follows [PyPI's Trusted Publishing setup](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
No PyPI API token or password belongs in this repository or in Colab.

### Release workflow

`.github/workflows/publish.yml` runs when a GitHub Release is published. Its tag must match
`pyproject.toml`, for example `v0.2.10`; local `+lookup` build versions cannot be published.
The workflow runs the locked tests and lint, builds wheel/sdist with commit provenance,
checks the installed wheel, and checks fresh installations on Linux Python 3.11, 3.12 and 3.13.
Only those two distribution files are handed to the separate `pypi` environment job.
That job uses the official [PyPA publish action](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
with GitHub's short-lived OIDC identity, without stored upload credentials.

Before the first release, configure the publisher above and the GitHub `pypi` environment.
Once checks succeed, publish the matching GitHub Release. Keep wheel/sdist/checksums on GitHub
as before; `uv.lock`, runtime profiles and notebooks remain in the tagged repository.
Verify the published PyPI version and distribution hashes before updating notebook installation
instructions to rely on PyPI. A failed upload is not a completed release.

Package changes merged into `main` are released automatically when `src/pettrace/**` changes.
The branch must bump `project.version` in `pyproject.toml` and refresh `uv.lock`; reusing an
existing version fails CI instead of replacing a release. Documentation, notebooks and tests
alone do not publish a package. Manual GitHub Releases remain supported by the same pipeline.
