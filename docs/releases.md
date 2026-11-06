# Package releases

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
