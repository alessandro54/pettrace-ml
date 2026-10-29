# Independent experiment notebooks

Convención de idioma: identificadores, comentarios, mensajes de ejecución y errores en inglés;
texto Markdown, tablas narrativas, pies de figura y conclusiones de la tesis en español.

Each experiment has one notebook that can run from top to bottom in a fresh kernel with an
installed `pettrace-ml` package and authorized access to its data and models. It does not require
another notebook's execution, report files, a repository checkout, or an import from `tools/`.

The notebook contains dependency installation, experiment configuration, the research question,
method explanations and formulas, dataset audit, model identity, training where applicable,
results, diagnostic figures, interpretation, limitations, and optional production MLflow logging.
The `pettrace` package contains the reusable implementation behind those cells.

| Experiment | Notebook | Status |
|---|---|---|
| E1 | [`e1_clip_zero_shot.ipynb`](e1_clip_zero_shot.ipynb) | [v4 run recorded in MLflow](https://ml.pettrace.app/#/experiments/3/runs/8601747ecd384d948d429b3b490449c9) |
| E2 | [`e2_yolo_crop_zero_shot.ipynb`](e2_yolo_crop_zero_shot.ipynb) | [v4 registrado en MLflow](https://ml.pettrace.app/#/experiments/3/runs/9117658f1945404ca9d1058257bf97f0); auditoría del detector y comparación pareada con E1 recalculado |
| E3 | [`e3_mlp_head_triplet.ipynb`](e3_mlp_head_triplet.ipynb) | public-data triplet training; computes E1/E2 comparisons on its v3 evaluation snapshot; real training pending |
| E4 | `e4_lora.ipynb` | planned |
| E5 | `e5_text_rerank.ipynb` | planned |

E2 fija el commit de v4 usado por E1 y verifica el mismo encoder CLIP, las mismas consultas y
galerías antes de comparar. Su última celda publica en MLflow por defecto; usa
`E2_LOG_TO_MLFLOW=false` o `LOG_TO_MLFLOW = False` para una revisión sin publicación.
La auditoría del detector, diferencias por consulta/identidad/luz, conclusiones calculadas y
reportes de la línea base se adjuntan al run E2. Ejecuta las celdas desde arriba en el kernel de Colab.

E3 todavía utiliza v3; no compares sus resultados con v4 como una mejora de modelo.
Los reportes históricos de v3 se conservan en sus ejecuciones de MLflow:
[E1 v3](https://ml.pettrace.app/#/experiments/3/runs/af8a88b1e2a6434c9eeb31b62c9010e4) and
[E2 v3](https://ml.pettrace.app/#/experiments/3/runs/e0d3e49aec554a4588efa3bce99955e1).

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
Generated `reports/` directories are Git-ignored; MLflow holds the published experiment record.

## Production MLflow

Provide `HF_TOKEN`, `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and
`MLFLOW_TRACKING_PASSWORD` through the kernel environment, Colab Secrets or hidden prompts.
The tracking and registry endpoint is `https://ml.pettrace.app`; the evaluation experiment must
already exist. Publication is controlled by the notebook configuration. Set `LOG_TO_MLFLOW = True` in the relevant notebook
(or `E1_LOG_TO_MLFLOW`, `E2_LOG_TO_MLFLOW`, `E3_LOG_TO_MLFLOW` in its environment).

Shared publication logic reuses matching completed runs and resumes failed uploads. It creates
no experiment and changes no model alias. E3's optional model registration is a separate,
explicit setting. Do not publish the same result concurrently from multiple kernels.

For local Docker execution, `make notebook NB=notebooks/e1_clip_zero_shot.ipynb` runs and saves the
notebook, then optionally attaches it using `tools/attach_notebook.py`. A publication receipt
prevents attachment to a stale run. The remote kernel cannot automatically read the notebook
file open on your Mac; save its outputs locally in VS Code.

See [E1 execution notes](../docs/e1_colab.md) and [E3 protocol](../docs/e3_colab.md).

## Índice de ejecuciones

[`experiments.yaml`](../experiments.yaml) conserva los enlaces de las ejecuciones registradas,
su dataset/modelo y una instantánea de las métricas principales. MLflow sigue siendo el registro
completo. Tras publicar desde Colab, ejecuta `make manifest` localmente para actualizar el índice;
el kernel remoto no modifica el repositorio Git. Versiona el manifiesto y conserva `reports/`
fuera de Git. Las ejecuciones fallidas o en curso también se incluyen si llevan una etiqueta E1–E5.
