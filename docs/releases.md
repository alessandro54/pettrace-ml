# Versioned package and experiment environments

Release `v0.2.0` reorganizes the package under `src/pettrace`. It does not claim a new
E3 result or change the research protocol. Existing notebook outputs remain historical evidence.

## Install a release

Use a fresh virtual environment with Python **3.11.15**, the reference interpreter in
`.python-version`. Download `requirements-e3.txt` from the matching GitHub Release:

```bash
uv python install 3.11.15
uv venv --python 3.11.15 .venv
curl --fail --location --output requirements-e3.txt \
  https://github.com/alessandro54/pettrace-ml/releases/download/v0.2.0/requirements-e3.txt
uv pip install --python .venv/bin/python --require-hashes -r requirements-e3.txt
uv pip check --python .venv/bin/python
```

The requirements include the exact release wheel URL and hash, as well as all transitive
dependencies exported from `uv.lock`. Use `requirements-evaluation.txt` for E1/E2 without
the training extra. Installing the wheel alone does **not** apply the lockfile.

To verify downloaded release assets, download `SHA256SUMS` and the listed assets into one
directory, then run `sha256sum --check SHA256SUMS` (macOS: `shasum -a 256 -c SHA256SUMS`).

## Colab and Jupyter

The notebooks install a specific release's hashed requirements into the selected kernel.
Restart the kernel after installation, then run from the package verification cell onward.
Do not replace packages in the middle of a training run. A new experiment must create a new
report directory; retry publication in the original session without retraining.

Colab's hosted Python, GPU and NVIDIA driver are managed externally. Its environment may
contain additional packages. The notebooks reject Python versions outside the release's
training support (3.11–3.12); use the reference interpreter in a dedicated environment for
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
  experiments/   # explicitly separate legacy orchestration protocol
  tracking.py    # publication and runtime/source provenance
  cli.py         # evaluation command
```

Imports changed in 0.2.0; the repository notebooks and tools use the new paths. Install and
restart the kernel as one migration. The legacy runner is not the notebook's two-domain
protocol; moving it does not make their results interchangeable.

## Build and publish

From a clean, committed checkout with uv **0.11.22**:

```bash
uv sync --locked --group dev --extra notebooks --extra training
uv run --locked python -m pytest -q
uv run --locked ruff check .
uv run --no-sync python tools/build_release.py
```

The builder creates `dist/v0.2.0/` with the wheel, source distribution, lockfile,
hashed requirements, Python version, build provenance and checksums. It refuses to replace
an existing release directory. The wheel embeds its source commit and lockfile hash;
`source_identity()` hashes every Python module, including nested packages.

Publish those files under the matching Git tag after checking the installed wheel. Never
replace assets or move a published version tag; make a new version for any correction.
The source distribution intentionally excludes datasets, credentials, reports and notebooks.
GitHub's automatically generated repository archives are separate from that distribution.
