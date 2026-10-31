# Package releases

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
.venv/bin/python -m pip install \
  https://github.com/alessandro54/pettrace-ml/releases/download/v0.2.3/pettrace_ml-0.2.3-py3-none-any.whl
```

This resolves dependencies within the package's declared ranges. Installing a wheel does **not**
apply `uv.lock`. For the locked development/training environment, use uv **0.11.22** and the
matching repository tag:

```bash
git clone --branch v0.2.3 --depth 1 https://github.com/alessandro54/pettrace-ml.git
cd pettrace-ml
uv python install 3.11.15
uv sync --locked --python 3.11.15 --group dev --group notebooks --extra notebooks --extra training
```

For reproducing E1, use its dedicated launcher below; installing the latest package alone does
not reproduce an experiment's recorded environment, data and model.

To verify downloaded release assets, download `SHA256SUMS` and the listed assets into one
directory, then run `sha256sum --check SHA256SUMS` (macOS: `shasum -a 256 -c SHA256SUMS`).

## Colab and Jupyter

E1 uses `colab/e1.ipynb` to create its own Python 3.11.15 environment and kernel. It never installs
experiment dependencies into the host kernel. The executed research notebook checks this environment.
The E2/E3 and preparation notebooks still install v0.2.1's hashed requirements into the selected kernel.
Restart the kernel after installation, then run from the package verification cell onward.
Do not replace packages in the middle of a training run. A new experiment must create a new
report directory; retry publication in the original session without retraining.

Colab's hosted Python, GPU and NVIDIA driver are managed externally. Its environment may
contain additional packages. E1's launcher fixes Python 3.11.15; E2 and data preparation support Python 3.11–3.13;
E3 training requires Python 3.11–3.12 with the locked PyTorch version. Use the reference interpreter in a dedicated environment for
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

The builder creates `dist/v0.2.3/` with the wheel, source distribution and `SHA256SUMS`.
It refuses to replace an existing release directory. The package embeds its version,
source commit, lockfile hash, reference Python and build-tool version in `pettrace/_build.json`;
`source_identity()` hashes every Python module, including nested packages.

Publish those files under the matching Git tag after checking the installed wheel. Never
replace assets or move a published version tag; make a new version for any correction.
The source distribution intentionally excludes datasets, credentials, reports and notebooks.
GitHub's automatically generated repository archives are separate from that distribution.
