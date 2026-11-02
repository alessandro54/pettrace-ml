# Package releases

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
.venv/bin/python -m pip install 'pettrace-ml[notebooks]==0.2.6'
```

The matching wheel and source distribution are also available in the
[GitHub release](https://github.com/alessandro54/pettrace-ml/releases/tag/v0.2.6).

This resolves dependencies within the package's declared ranges. Installing a wheel does **not**
apply `uv.lock`. For the locked development/training environment, use uv **0.11.22** and the
matching repository tag:

```bash
git clone --branch v0.2.6 --depth 1 https://github.com/alessandro54/pettrace-ml.git
cd pettrace-ml
uv python install 3.11.15
uv sync --locked --python 3.11.15 --group dev --group notebooks --extra notebooks --extra training
```

For E1, follow its interactive notebook setup below; installing the latest package alone does
not reproduce an experiment's recorded environment, data and model.

To verify downloaded release assets, download `SHA256SUMS` and the listed assets into one
directory, then run `sha256sum --check SHA256SUMS` (macOS: `shasum -a 256 -c SHA256SUMS`).

## Colab and Jupyter

Data lookup uses `requirements/data-lookup-colab.txt`, a committed hash-checked lock for
Linux x86_64 / Python 3.13.18. Its `.in` file records the user's reported `google-colab` requirements
and numerical-package versions. The lock includes PetTrace 0.2.6 and all resolved dependencies;
`google-colab` itself remains supplied by the hosted runtime.

The shared `.github/workflows/colab.yml` installs this profile in CI and before publication,
checks dependency consistency, exercises data lookup and starts a Jupyter kernel. It also
replaces only PetTrace with the candidate wheel and requires its notebook extra to resolve
without changing any dependency. This validates the reported Python requirements, not the
hosted Google frontend or an authenticated Drive mount. CI never regenerates the profile.
Update the `.in` inputs and regenerate/review the lock explicitly when the Colab target changes:

```bash
uv pip compile requirements/data-lookup-colab.in --python-version 3.13.18 --python-platform x86_64-unknown-linux-gnu --generate-hashes --output-file requirements/data-lookup-colab.txt --no-annotate
```

The notebook must reference the profile by immutable Git commit, not `main`. The package's
general dependency ranges remain available for other environments; they are not this lock.

E1 and E2 install the v0.2.2 wheel with the immutable `requirements/e1-runtime.txt` profile in Python 3.11–3.13.
This profile excludes notebook infrastructure supplied by Colab. Restart only if setup detects
changed loaded packages, then continue from verification; research cells display results directly
(see [E1 execution](e1_colab.md) and [E2 execution](e2_colab.md)). The historical `tools/run_e1.py` remains a separate
Python 3.11.15 reference workflow.
Existing E3 and source-volume preparation notebooks remain pinned to their v0.2.5.dev3 wheel
plus `requirements/e3-runtime.txt` (Linux/Python 3.13, PyTorch 2.9.1, ONNX 1.19.1).
Publishing 0.2.5 to PyPI does not migrate those frozen workflows. Use their documented wheel
and checksums as described in [E3 execution](e3_colab.md). Restart only if setup requests it.
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

The builder creates `dist/v0.2.6/` with the wheel, source distribution and `SHA256SUMS`.
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
`pyproject.toml`, for example `v0.2.6`; local `+lookup` build versions cannot be published.
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
