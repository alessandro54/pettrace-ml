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

## Preparación progresiva y reutilizable

Para una nueva preparación usa [`datasets/prepare_training_v2.ipynb`](../datasets/prepare_training_v2.ipynb)
con el wheel local `0.2.5.dev3` (ver [instalación y cachés](../datasets/README.md)).
`prepare_training.ipynb` conserva la corrida anterior: no ejecutes ambos a la vez sobre Drive.
La preparación comparte el algoritmo de limpieza,
y cambia únicamente `SOURCE_SAMPLE_PCT` entre 20, 50 y 100. El código compartido está
en `pettrace.data.preparation` y `pettrace.data.volume`; no hay tres algoritmos de limpieza.

El orden SHA-256 de las rutas originales con semilla 42 define muestras anidadas
20 ⊆ 50 ⊆ 100, antes de mirar etiquetas o calidad. Se verifican los índices completos
de ambas fuentes. Los metadatos de perros se recorren completos, sin el prefijo del piloto.
Fotos sin identidad válida siguen contando como procesadas y reciben un descarte;
no se sustituyen por otra foto para mejorar artificialmente la muestra.

**Mantén el respaldo de `DATA_ROOT` en Drive** al cambiar de sesión Colab:

- `source-cache/` reutiliza fotos y metadatos verificados por CRC/SHA-256.
- `animal-screening-cache/` reutiliza detecciones por imagen, modelo, configuración y runtime.
- `feature-cache/` reutiliza embeddings E2 por imagen, artefactos y runtime, independientemente del volumen.

20→50 añade 30 puntos porcentuales de fotos y 50→100 añade 50. Se reaudita el conjunto
para detectar duplicados/conflictos nuevos y se reconstruyen los manifiestos; la
caché evita repetir las inferencias costosas, no las comprobaciones de integridad.
Los conjuntos adquiridos son anidados; los retenidos pueden cambiar al detectar
conflictos. Una caché de otro modelo/runtime se mantiene separada.

Se reutilizan exactamente las fotos/identidades de validación y test públicos v1.
Sus identidades y hashes se excluyen antes de formar el nuevo train. Se reutilizan
el mismo filtro YOLO, limpieza, deduplicación y mínimo de cuatro fotos por identidad.
La preparación solo se marca completa si cada foto muestreada tiene un resultado:
train o exclusión con motivo. Publicar datasets en HF sigue separado y opcional.
Las versiones propuestas son `v2.20`, `v2.50`, `v2.100`; no existen hasta prepararlas/publicarlas.

## Paquete y ejecución

El nuevo código requiere **pettrace-ml v0.2.4**. Construye el paquete desde un commit
limpio con `python tools/build_release.py`. Hasta publicar esa release, coloca en
el runtime de Colab el wheel, `SHA256SUMS` y una copia de
[`requirements/e3-runtime.txt`](../requirements/e3-runtime.txt) llamada
`requirements-e3-runtime.txt`. Los notebooks verifican sus hashes; no descargan una
release inexistente ni importan código de GitHub `main`.

El perfil E3 fija Linux/Python 3.13, PyTorch 2.9.1 y ONNX 1.19.1. Conserva los pins
numéricos de evaluación de E1/E2, sin reinstalar IPython o el kernel. Reinicia solo
si setup detecta que reemplazó una biblioteca ya cargada. El runtime histórico del
piloto era distinto; sus métricas no acreditan resultados con este nuevo perfil.

Después de preparar una versión, configura `HF_TRAINING_DATASET_VERSION`,
`HF_TRAINING_DATASET_REVISION` (commit) y `HF_TRAINING_DATASET_FINGERPRINT`, o usa
`HF_TRAINING_DATASET_REPO=""` y `E3_PREPARED_DATASET` para su copia local verificada.
Empieza por E3 20 en una GPU T4. Las credenciales van en entorno, Secrets o campos ocultos.

Los tres notebooks mantienen las fases de E1/E2: datos, representación/entrenamiento
y evaluación/análisis. Comparten semilla 42, arquitectura, P×K=8×4, AdamW, margen 0,2,
100 pasos por época, hasta 30 épocas y paciencia 7. Cada MLP **empieza desde cero**:
continuar pesos del 20 en el 50 confundiría volumen con entrenamiento adicional.
Una época no equivale a recorrer todas las fotos. MLflow registra pasos, ejemplos
procesados y fotos distintas vistas por el optimizador.

**E1/E2 están congelados.** E3 lee sus métricas/per-query publicados, verifica
modelo, snapshot, protocolo y consultas, y guarda los IDs y hashes de los artefactos
usados como controles. No vuelve a evaluar ni publicar E1/E2. El extractor E2 sí
produce las representaciones congeladas necesarias para entrenar la MLP.

## Volumen reportado

| Métrica MLflow | Significado |
|---|---|
| `dataset.sample_requested_pct` | 20, 50 o 100 solicitado de cada fuente |
| `dataset.sampled_photos` / `dataset.sampled_pct` | Fotos adquiridas antes de limpieza / porcentaje de fuentes |
| `dataset.processed_photos` / `dataset.processed_pct` | Fotos con procesamiento terminado, incluida exclusión temprana por etiquetas |
| `dataset.excluded_photos` | Fotos adquiridas que no entraron a train, con motivos en el reporte |
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

### Persistencia con Drive (v0.2.4)

Instalar el wheel v0.2.4 y sus checksums antes de una nueva corrida. Mantener el
mismo perfil `requirements-e3-runtime.txt`; no hay dependencias nuevas. Los ZIP
originales opcionales van en `MyDrive/PetTrace/raw/`. El notebook de preparación
restaura y respalda el trabajo local mediante bloques verificados. E3 restaura
las versiones preparadas `v2.20`, `v2.50` o `v2.100` cuando existen y no se ha
seleccionado una versión HF explícita; conserva los controles públicos/privados.

Los embeddings se respaldan entre lotes y los reportes durante/al terminar el
entrenamiento y al publicar. Una sola sesión puede escribir en cada carpeta de
caché compartida. Una interrupción brusca recupera el último respaldo confirmado,
no el estado del optimizador. Ver [detalle y límites](../datasets/README.md#drive-y-recuperación-entre-sesiones-paquete-v024).

### Reintento de publicación tras HTTP 413

En un kernel que ya entrenó con v0.2.4, subir `src/pettrace/artifact_upload.py`
a `/content/artifact_upload.py` y ejecutar únicamente la última celda actualizada.
No reinstalar ni reiniciar: el helper adapta solo el transporte de artefactos y
el publicador conserva la clave del run para reanudarlo. Los JSON locales permanecen
intactos. En MLflow, los JSON grandes se guardan como partes gzip de hasta 512 KiB
y un `<nombre>.transport.json` con tamaños, orden y hashes SHA-256. Para recuperar
el original, concatenar las partes en ese orden, descomprimir gzip y verificar el
hash original del manifiesto. Los modelos y demás archivos conservan sus rutas.
