# E3 · Frozen CLIP + triplet-trained MLP

E3 adds a 512 → 256 → 512 MLP, ReLU and L2 normalization to the frozen E2 detector-crop encoder.
Training uses same-species P×K batches and batch-hard triplet loss. Frozen embeddings are cached;
only the head is optimized. There is no image augmentation in this first implementation.

## Run independently

Open `notebooks/e3_mlp_head_triplet.ipynb`, select a GPU kernel, and run from the top. Dependencies
and the `pettrace-ml` package are installed in visible cells. The package source must be published
before a remote source install can use it; a release wheel can be selected using
`PETTRACE_PACKAGE_SOURCE`. No checkout or previous E1/E2 reports are required beside the notebook.

Provide HF and production MLflow credentials through Secrets, environment variables, or hidden
prompts. Public downloads can require additional Kaggle access. MPDD is pinned by URL/SHA-256;
Cat Individuals resolves and records its dataset version and archive hash. Downloading that
public data needs substantial disk space. If automatic download is unavailable, pass extracted
folders with `prepare(DATA_ROOT, cat_root=..., mpdd_root=...)` in the data cell.

## Training and evaluation boundaries

`pettrace.acquisition` downloads and safely extracts the public sources, then calls the shared
identity discovery, duplicate audit and split logic in `pettrace.public_data`. Splits are by
identity. Training receives only `train` and `val` rows; the original public test rows are kept
out of checkpoint selection. Validation macro-species mAP selects the head checkpoint.

The notebook exports a fused ONNX artifact and verifies it against the serving pipeline and
golden set. It evaluates the private pinned PetTrace snapshot (currently v3) using the shared
zero-shot protocol, and recomputes E1/E2 comparisons on that same snapshot. Its displayed EER
is descriptive: fitting and reporting use the same private pairs. It does not establish an
independently validated production threshold or report public held-out test metrics.

The optional `pettrace.experiments` programmatic runner additionally calibrates thresholds on
public validation data, evaluates public/local held-out tests, checks cross-source duplicates,
and verifies real-image serving parity. That extended runner and its outputs are distinct from
the notebook's current private zero-shot report.

## Publication and saved evidence

`LOG_TO_MLFLOW` defaults to false. Enable it to publish the evaluation/report to the existing
production experiment. `REGISTER_E3` is a separate opt-in for registering the exported model;
it does not promote a serving alias. Baseline figures, training history and model artifacts
are saved inside this experiment's own report directory.

Save the executed notebook in VS Code and preserve unpublished reports before ending the remote
runtime. Synthetic tests establish implementation checks, not real E3 training results. The
first real training run remains pending.

## Sources

- MPDD: https://data.mendeley.com/datasets/v5j6m8dzhv/1
- Cat Individual Images: https://www.kaggle.com/datasets/timost1234/cat-individuals
- Reference reader: https://pypi.org/project/wildlife-datasets/1.0.9/

The original harness came from `pettrace-ml`, base commit
`81472d9ae9dfd9773582ff4301682c963d73036b`. The installed package version and source hash identify
subsequent changes. Check each downloaded dataset's license before use or redistribution.
