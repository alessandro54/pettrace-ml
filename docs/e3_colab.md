# E3 · Volumen de las fuentes públicas

E3 entrena únicamente la MLP 512→256→512 sobre YOLO y CLIP congelados. Las tres
variantes cambian la muestra adquirida de **cada fuente original**, no un porcentaje
sobre las 1.276 fotos del piloto de entrenamiento:

| Notebook | Dogs of the World | LCW | Total adquirido solicitado |
|---|---:|---:|---:|
| [E3 20%](../notebooks/e3_mlp_head_triplet_20.ipynb) | 62.734 | 76.184 | 138.918 |
| [E3 50%](../notebooks/e3_mlp_head_triplet_50.ipynb) | 156.835 | 190.461 | 347.296 |
| [E3 100%](../notebooks/e3_mlp_head_triplet_100.ipynb) | 313.671 | 380.923 | 694.594 |

Se redondea hacia abajo por fuente. El denominador son entradas de imagen anteriores
al filtrado, según [los índices verificados](../datasets/source_capacity_v1.json).
Una foto adquirida puede quedar excluida por etiqueta ambigua, duplicación, detección
o insuficientes fotos por identidad. El 100% adquirido no significa que todo sirva
para entrenar.

## Estado y preparación necesaria

**La muestra ampliada del 20% todavía no está preparada ni publicada.** El pipeline
actual de `datasets/prepare_training.ipynb` adquiere un piloto acotado por identidades,
con un prefijo de metadatos para perros; no implementa por sí solo estas cuotas.
No basta con renombrar v1 o cambiar la constante del notebook.

La preparación ampliada debe muestrear desde los índices completos de ambas fuentes,
con semilla y orden reproducibles, y mantener muestras anidadas 20 ⊆ 50 ⊆ 100. Debe
conservar en `sources.json` los recibos de adquisición (`ref`, `version`,
`archive_index_sha256`, `extraction_pending: false`, `file_sha256` por ruta original).
Se cuentan imágenes bajo `images/` en perros y `CatFullDataset/` en gatos, con las
extensiones del importador `.jpg`, `.jpeg`, `.png`; los metadatos no cuentan como fotos.
Esos recibos deben formar parte del inventario verificado de `prepared.json`.
El notebook comprueba cada cuota y que las fotos de train pertenezcan a la muestra.

Antes de construir train/val, la preparación debe reservar **todas las identidades**
de validación y test de `pettrace-training@v1`. La validación conserva exactamente
sus 273 fotos y 60 identidades; el test público conserva sus 277 fotos y 60 identidades.
Las fotos adicionales de esas identidades no pasan a train. Los notebooks rechazan
solapamiento de train/val con el test v1 por identidad o hash, y validación distinta.
La limpieza vive en `datasets/`, no en estos experimentos.

## Ejecución

Configura una versión preparada ampliada mediante `HF_TRAINING_DATASET_VERSION`,
`HF_TRAINING_DATASET_REVISION` (commit inmutable) y
`HF_TRAINING_DATASET_FINGERPRINT`. Para una copia local verificada, usa
`HF_TRAINING_DATASET_REPO=""` y `E3_PREPARED_DATASET`. No se inventan commits ni
fingerprints para versiones aún inexistentes.

Empieza por E3 20 en una GPU T4. Estos notebooks conservan el entorno de entrenamiento
v0.2.1: **Python 3.11–3.12, PyTorch 2.5.1**. No están adaptados al kernel Python 3.13
de la corrida interactiva E2. Instala desde la primera celda, reinicia y continúa
con la verificación. Las credenciales van en entorno, Colab Secrets o campos ocultos.

Las tres variantes mantienen semilla 42, arquitectura, P×K=8×4, AdamW, margen 0,2,
100 pasos por época, hasta 30 épocas y paciencia 7. Una época no equivale a recorrer
todas las fotos. El checkpoint se selecciona por mAP macro de validación pública;
los tests público y PetTrace se evalúan por separado después de fijarlo.

## Volumen reportado

| Métrica MLflow | Significado |
|---|---|
| `dataset.sample_requested_pct` | 20, 50 o 100 solicitado de cada fuente |
| `dataset.sampled_photos` / `dataset.sampled_pct` | Fotos adquiridas antes de limpieza / porcentaje de fuentes |
| `dataset.train_photos` / `dataset.train_identities` | Volumen disponible para ajustar pesos después de limpieza y split |
| `dataset.train_pct` | 100 × fotos de train / 694.594 fotos originales |
| `dataset.val_photos` / `dataset.val_identities` | Validación usada para seleccionar el checkpoint |
| `dataset.train_val_photos` / `dataset.train_val_pct` | Train más validación, sin test |

Los conteos y porcentajes también se registran por especie (`dataset.cat.*`,
`dataset.dog.*`). Se conserva el volumen en `dataset-coverage.json`,
`training-config.json`, `experiment-design.json`, el análisis y los hallazgos.
**Ambos runs MLflow**, público y privado, reciben las métricas de volumen y entrenamiento.
`source_sample_pct` identifica la variante y los directorios son
`reports/e3_{20,50,100}_v4/<fecha-id>`.

Publicar es obligatorio. Reintenta la última celda en la misma sesión si falla,
sin repetir entrenamiento; conserva `REPORT` antes de cerrar Colab. `REGISTER_E3`
permanece en `True`, separado de publicar resultados, y no promueve un alias.

## Evidencia histórica

Las salidas anteriores se conservaron en el notebook 20, identificadas explícitamente
como **piloto histórico, no resultado del 20%**. Usaron 1.276 fotos de train
(0,1837% de las fuentes), 273 de validación y seleccionaron la época 3. El último
estado guardado no acredita publicación completa. Los notebooks 50 y 100 nacen sin
salidas; ninguna de las nuevas cuotas se presenta como una corrida ya ejecutada.

Los resultados serán descriptivos: una sola semilla, fotos dependientes por identidad,
PetTrace v4 ya observado y sin garantía de igualdad bit a bit entre GPU. La metodología
y referencias siguen junto a las celdas; la adquisición se documenta en
[el pipeline de datos](e3_data_pipeline.md).
