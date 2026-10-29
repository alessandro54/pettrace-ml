# pettrace-ml

Investigación sobre **re-identificación visual de mascotas** para PetTrace, desarrollada en el
marco de una tesis de la UPC. Ante una fotografía de un avistamiento, el sistema ordena candidatos
de la misma especie por similitud visual; la confirmación de identidad corresponde a la persona.
La pregunta es cómo recuperar **al mismo individuo**, no cómo clasificar su raza.

Este repositorio presenta el recorrido experimental, las decisiones metodológicas y la evidencia
obtenida. E1 establece una línea base con CLIP; E2 estudia el recorte mediante detección; E3 plantea
una proyección métrica entrenable. Lo implementado y lo ejecutado se distinguen de las etapas
propuestas. Los [cuadernos de investigación](notebooks/README.md) integran explicación, cálculo,
resultados y límites; MLflow conserva la evidencia completa.

El paquete de evaluación, registro de modelos y herramientas permite reproducir los análisis.
La aplicación (iOS, Android, core en Go y servicio de visión) se mantiene en otro repositorio.

## Experiments

| ID | Pipeline | Training | Status |
|---|---|---|---|
| E1 | full image → CLIP ViT-B/32 → 512-d, L2 | none (zero-shot) | registered, in production |
| E2 | YOLOv8n detects the animal → crop (10 % margin) → same CLIP | none (zero-shot) | registered, candidate |
| E3 | E2 + frozen CLIP + MLP head, triplet loss | public identity train/val | implemented; first Colab run pending ([guide](docs/e3_colab.md)) |
| E4 | E3 + LoRA fine-tuning (r = 8, α = 16, q/v) | LoRA on `train` | design |
| E5 | E4 + re-ranking with the sighting's text description | weight on `val` | design |

### Evidencia actual: comparación E1/E2 en `pettrace-reid@v4`

Las ejecuciones registradas comparten el commit de dataset
`8b12db0dce41040a821f9792bea5b352342263ce`. E2 recalcula E1 sobre las mismas consultas y galerías
para estudiar el recorte con el encoder constante.

| Indicador | E1 | E2 |
|---|---:|---:|
| Rank-1 | 65.90 % | 67.87 % |
| Rank-5 | 94.10 % | 92.13 % |
| mAP | 0.3751 | 0.3960 |
| EER descriptiva | 36.23 % | 33.83 % |

En esta muestra, el recorte mejora Rank-1, mAP y EER, mientras Rank-5 disminuye. El contraste es
descriptivo: no demuestra significancia estadística ni qué señales causan la recuperación.
La separación leave-one-out no distingue sesiones de captura, y la EER no usa calibración
independiente. Los detalles y diagnósticos están en [E1 v4](https://ml.pettrace.app/#/experiments/3/runs/8601747ecd384d948d429b3b490449c9)
y [E2 v4](https://ml.pettrace.app/#/experiments/3/runs/9117658f1945404ca9d1058257bf97f0), localizables
también desde [`experiments.yaml`](experiments.yaml).

E3 prepara un piloto revisado de Dogs of the World y LCW y compara contra las líneas base
sobre el mismo commit de v4. Descargas completas, publicación y registro están desactivados
por defecto. El entrenamiento real y sus métricas siguen pendientes; el uso previo de v4
lo convierte en un benchmark exploratorio, no en una prueba final nunca observada.

### Antecedente: resultados en `pettrace-reid@v3` (zero-shot)


14 animals (7 cats, 7 dogs), 128 photos. Every photo is a query against the other photos of the
same species (leave-one-out); each animal scores with its best photo. Chance Rank-1 = 1/7 = 14.3 %.
The EER threshold is chosen on the same pairs (optimistic).

| Metric | E1 | E2 |
|---|---|---|
| Rank-1 (all / cats / dogs) | 73.4 % / 75.4 % / 71.6 % | 70.3 % / 65.6 % / 74.6 % |
| Rank-5 | 98.4 % | 99.2 % |
| mAP | 0.509 | 0.527 |
| EER (threshold) | 39.2 % (0.794) | 35.0 % (0.812) |

En v3, el recorte se asocia con menor EER y mayor mAP, ayuda a perros y reduce el Rank-1 de gatos.
La confusión entre `cat-01` y `cat-02`, de patrón atigrado similar, aparece en 6 de 10 consultas de
`cat-01`. Es una observación descriptiva: no aísla una causa atribuible al fondo o al pelaje.
El tamaño de muestra y la dependencia entre fotos limitan la interpretación del delta global.
Per-animal tables and figures:
[E1 v3 in MLflow](https://ml.pettrace.app/#/experiments/3/runs/af8a88b1e2a6434c9eeb31b62c9010e4),
[E2 v3 in MLflow](https://ml.pettrace.app/#/experiments/3/runs/e0d3e49aec554a4588efa3bce99955e1).

| | Notebook (canonical record) | MLflow run (`pettrace-reid-eval`) | Code commit |
|---|---|---|---|
| E1 | [`e1_clip_zero_shot.ipynb`](notebooks/e1_clip_zero_shot.ipynb) | [`af8a88b1`](https://ml.pettrace.app/#/experiments/3/runs/af8a88b1e2a6434c9eeb31b62c9010e4) | `f80d1d6` |
| E2 | [`e2_yolo_crop_zero_shot.ipynb`](notebooks/e2_yolo_crop_zero_shot.ipynb) | [`e0d3e49a`](https://ml.pettrace.app/#/experiments/3/runs/e0d3e49aec554a4588efa3bce99955e1) | `c9dd2b9` |

Each notebook is an independently executable experiment report. It installs the `pettrace-ml`
package, declares its own configuration, loads its own data/models, and presents the method,
tables, figures, interpretation and limitations. E2 and E3 compute their baseline comparisons
within the same run; neither requires reports from an earlier notebook. Publication to production
MLflow is explicitly controlled in each notebook.

For development, install from the checkout with `uv sync --group dev --extra training`.
For a remote kernel, notebooks use the installed package. Until the first wheel release exists,
`PETTRACE_PACKAGE_SOURCE` defaults to a pip source install from the public repository's `main`.
Publish the code first; local edits are not available to Colab. A release wheel URL can be used
instead without changing the experiment cells. Dependencies remain visible in each notebook.
Package metadata permits Python 3.11–3.13. Verification here uses Python 3.11; a real Colab
3.13 run remains to be checked. The Docker environment is fixed by `uv.lock`.

The distribution is named `pettrace-ml`; its Python import package is `pettrace`:

```python
from pettrace import data, metrics
from pettrace.evaluation import evaluate
```

## Experiment results

El índice versionado [`experiments.yaml`](experiments.yaml) reúne las ejecuciones publicadas de
E1–E5: dataset, modelo, estado, commit, métricas principales y enlace a MLflow. Las métricas del
índice son una instantánea; los resultados completos y sus artefactos siguen en MLflow.
Después de publicar nuevas ejecuciones, actualiza el índice con `make manifest` y versiona solo
el manifiesto. El comando consulta MLflow en modo lectura y no descarga reportes ni crea runs.


MLflow is the source of truth for metrics, figures, analysis, conclusions, provenance and executed
notebooks. The completed [E1 v4 run](https://ml.pettrace.app/#/experiments/3/runs/8601747ecd384d948d429b3b490449c9)
is recorded separately from the historical v3 runs above. Server read access is required.

Local `reports/` directories are working output only and are Git-ignored. Keep unpublished or
failed-upload outputs until publication succeeds; results can be downloaded from MLflow later.
Do not commit generated reports or create a second versioned results archive in this repository.

## Identidad de los artefactos históricos de v3

Registry version numbers and run IDs are local to one MLflow server; these are not:

| | Identity |
|---|---|
| Dataset | Hugging Face `alessandro54/pettrace-reid`, tag `v3`, commit `354763693e9105eb953c559620037dd98995326b` |
| E1 model | `clip-vit-b32@v1`: `encoder.onnx` sha256 `4773e649274ca8774c4a63d45f0438fe598ee7ba67e65def97537aa0eca9a576`, golden fingerprint `5e40c7774bd7ad51` |
| E2 model | `clip-vit-b32-crop@v1`: same encoder + `detector.onnx` sha256 `da8f9b9ebea5157cdcefa7bea9fd782249ebdf119f917c07d64e5b41942b9c9d` |
| Code | the commit of this repository recorded on each run (`git_sha` tag) |

Every evaluation run in MLflow (`https://ml.pettrace.app`, experiment `pettrace-reid-eval`)
records the dataset commit and fingerprint, the resolved model version, the sha256 of every ONNX
file it evaluated, the code commit, the ONNX Runtime version and the CPU.

## Reproduce

Needs Docker, read access to the private dataset (Hugging Face token) and an MLflow user
(reviewers get a read-only one; ask the author).

```bash
git clone https://github.com/alessandro54/pettrace-ml && cd pettrace-ml
git checkout <commit from the run's git_sha tag>   # f80d1d6 (E1), c9dd2b9 (E2)
cp .env.example .env            # MLflow user/password + HF_TOKEN
make image                      # Linux, Python 3.11, versions from uv.lock

make notebook NB=notebooks/e1_clip_zero_shot.ipynb        # E1: executes; enable E1_LOG_TO_MLFLOW=true to publish
make notebook NB=notebooks/e2_yolo_crop_zero_shot.ipynb    # E2, including its own E1 baseline
make evaluate DATASET=v3 MODEL=models:/pettrace-embedder/2 PROTOCOL=zero-shot REPORT=reports/e2_v3  # CLI
```

Evaluating writes a run, so a read-only user can reproduce the numbers without logging:
`docker run --rm --env-file .env -v $PWD:/repo pettrace-ml -m pettrace.cli --dataset v3 --model models:/pettrace-embedder/2 --protocol zero-shot --no-mlflow`.

The optional evaluation CLI is `pettrace-evaluate` (or `python -m pettrace.cli`).

The models are downloaded from MLflow and checked by hash. To rebuild them from scratch instead:
`make image-export && make register-v1` re-exports E1 from the pinned Hugging Face revision and
must give the same sha256; E2's detector is the COCO-pretrained `yolov8n.pt` exported with
Ultralytics 8.3.253 (command in [`tools/register_e2.py`](tools/register_e2.py)); a re-export
is not guaranteed to be byte-identical, so the validated file is the one inside version 2.

## Layout

```
pettrace/    reusable loading, inference, training, evaluation, figures and tracking
notebooks/   independently executable experiment reports (E1–E5)
tests/       package correctness checks; not experimental results or runtime code
tools/       optional registration, verification and notebook-attachment commands
docs/        execution notes and experiment protocols
experiments.yaml  small Git-tracked index of MLflow runs
reports/     ignored working output, generated when needed; published results live in MLflow
```

`pettrace/pipeline.py`, `pettrace/golden.py` and `pettrace/clip_reference.py` mirror the vision service's
modules in the app repository; the golden fingerprint on every registered version (checked by the
service at startup and by `make verify-model`) fails if they diverge.

## Data and licences

- **Photos are not in this repository.** `pettrace-reid` is a private dataset: owners gave consent
  for a private research dataset; EXIF/GPS is stripped. Reports contain scores and file names only.
- **Animals are pseudonymous.** The dataset and everything here use codes (`cat-01`, `dog-03`);
  pets' names never leave the curation tool, because names plus photos identify the owners.
  Published v3 and v4 snapshots have distinct commits; historical name-based versions were removed.
- Code: MIT ([LICENSE](LICENSE)).
- YOLOv8 weights (E2 detector) are AGPL-3.0 (Ultralytics); they are not distributed here.
- CLIP ViT-B/32: OpenAI, MIT.
