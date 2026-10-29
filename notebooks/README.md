# Independent experiment notebooks

Each experiment has one notebook that can run from top to bottom in a fresh kernel with an
installed `pettrace-ml` package and authorized access to its data and models. It does not require
another notebook's execution, report files, a repository checkout, or an import from `tools/`.

The notebook contains dependency installation, experiment configuration, the research question,
method explanations and formulas, dataset audit, model identity, training where applicable,
results, diagnostic figures, interpretation, limitations, and optional production MLflow logging.
The `pettrace` package contains the reusable implementation behind those cells.

| Experiment | Notebook | Status |
|---|---|---|
| E1 | [`e1_clip_zero_shot.ipynb`](e1_clip_zero_shot.ipynb) | full-image CLIP baseline on v4; real v4 run pending |
| E2 | [`e2_yolo_crop_zero_shot.ipynb`](e2_yolo_crop_zero_shot.ipynb) | detector crop; computes its own E1 comparison on v3 |
| E3 | [`e3_mlp_head_triplet.ipynb`](e3_mlp_head_triplet.ipynb) | public-data triplet training; computes E1/E2 comparisons on its v3 evaluation snapshot; real training pending |
| E4 | `e4_lora.ipynb` | planned |
| E5 | `e5_text_rerank.ipynb` | planned |

Do not compare E1's new v4 run to E2/E3's v3 run as a model improvement. Their snapshots must be
aligned for a new cross-experiment benchmark. Historical v3 reports remain under `reports/`.

## Package installation

Run the visible dependency cell and package-install cell. A local development kernel can use
`uv sync --group dev --extra training`. A remote kernel installs the public package source when
`pettrace` is absent. This requires committing/publishing the current source first. Set
`PETTRACE_PACKAGE_SOURCE` to a versioned release wheel URL when one is available; the first
wheel release has not been published by this cleanup. There are no encoded source bundles.

The package's code hash, installed version, and available Git/VCS commit identify the run even
when the kernel has no repository checkout. Notebooks write only their own outputs under
`reports/eN_<dataset>` by default; `PETTRACE_OUTPUT_DIR` can select a different output location.
Colab storage is temporary, so save the executed notebook and unpublished reports before closing.

## Production MLflow

Provide `HF_TOKEN`, `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and
`MLFLOW_TRACKING_PASSWORD` through the kernel environment, Colab Secrets or hidden prompts.
The tracking and registry endpoint is `https://ml.pettrace.app`; the evaluation experiment must
already exist. Publication defaults to false. Set `LOG_TO_MLFLOW = True` in the relevant notebook
(or `E1_LOG_TO_MLFLOW`, `E2_LOG_TO_MLFLOW`, `E3_LOG_TO_MLFLOW` in its environment).

Shared publication logic reuses matching completed runs and resumes failed uploads. It creates
no experiment and changes no model alias. E3's optional model registration is a separate,
explicit setting. Do not publish the same result concurrently from multiple kernels.

For local Docker execution, `make notebook NB=notebooks/e1_clip_zero_shot.ipynb` runs and saves the
notebook, then optionally attaches it using `tools/attach_notebook.py`. A publication receipt
prevents attachment to a stale run. The remote kernel cannot automatically read the notebook
file open on your Mac; save its outputs locally in VS Code.

See [E1 execution notes](../docs/e1_colab.md) and [E3 protocol](../docs/e3_colab.md).
