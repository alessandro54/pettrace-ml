# Datasets de PetTrace

Esta carpeta contiene el pipeline de datos compartido. Los notebooks de `notebooks/` siguen dedicados a los experimentos.

| Dataset | Función | Versiones |
|---|---|---|
| `pettrace-training` | Identidades públicas de perros y gatos para entrenar y validar representaciones | `v1`, `v2`, …; cada versión fija imágenes, splits y fingerprint |
| `pettrace-reid` | Benchmark privado de mascotas de PetTrace | `v4` permanece como referencia de evaluación |

## Pipeline común

Ejecutar [prepare_training.ipynb](prepare_training.ipynb) en el entorno donde se preparan los datos. Usa Dogs of the World y LCW, con adquisición acotada o extracciones existentes. Aplica:

1. Verificación de archivos y hashes.
2. YOLO y NMS: exactamente un perro o gato de la especie esperada.
3. Exclusiones conocidas y cuarentena de IDs con imágenes exactas compartidas entre etiquetas.
4. Duplicados exactos y casi duplicados dentro de cada identidad.
5. Al menos cuatro fotos distintas por individuo.
6. Splits por identidad con semilla fija y conteos mínimos por especie.
7. Auditoría, muestra visual reproducible y registro local de la versión.

El modo exploratorio no exige aprobar cada imagen. Las etiquetas del publicador siguen sin verificación visual individual; el reporte conserva esa limitación y los posibles errores de YOLO.

## Organización

```text
datasets/
  prepare_training.ipynb       # Pipeline compartido, versionado en Git
  public/e3-body/              # Fuentes y cachés existentes, fuera de Git
  build/pettrace-training/
    <fingerprint>/            # Imágenes y manifiestos de una versión preparada
    v1.json                   # Nombre estable → fingerprint
    v2.json
```

La descarga usa cuatro conexiones por defecto (`DOWNLOAD_WORKERS`), con progreso, velocidad y ETA. Al reanudar verifica por CRC y omite las fotos ya válidas. El paralelismo no cambia la selección de identidades ni obliga a borrar la raíz existente.

La ruta histórica `public/e3-body` se conserva para reutilizar las descargas existentes. Los snapshots generados sirven a E3 y a futuros experimentos; no dependen de volver a adquirir las fuentes. Un nombre local como `v1` no se mueve a otro fingerprint.

## Hugging Face y consumo

`PUBLISH_TO_HF = False` deja la preparación local. Al activarlo, el cuaderno solicita `HF_TOKEN` mediante entorno, Secrets o campo oculto y publica únicamente el inventario verificado en el repositorio privado `alessandro54/pettrace-training`, bajo `versions/v1/`. Se crea una etiqueta `v1` ligada al commit; no se sobrescriben etiquetas existentes con otro dataset. Se necesita permiso de escritura HF para publicar y solo lectura para consumir.

La publicación de datos no activa el registro de modelos ni la publicación de resultados en MLflow. El benchmark `pettrace-reid` está protegido contra usos como repositorio de entrenamiento.

Un experimento puede cargar:

```python
from pettrace.data.versions import load_local_training, load_training

prepared, provenance = load_local_training("datasets/build/pettrace-training/v1.json")
# Después de publicar:
# prepared, provenance = load_training(
#     repo_id="alessandro54/pettrace-training", version="v1", revision="<commit publicado>")
```

E3 acepta `E3_PREPARED_DATASET` para una versión local. Para HF utiliza `HF_TRAINING_DATASET_REPO`, `HF_TRAINING_DATASET_VERSION`, `HF_TRAINING_DATASET_REVISION` y, opcionalmente, `HF_TRAINING_DATASET_FINGERPRINT`. El loader resuelve la referencia una vez, descarga por commit y valida todos los hashes antes de entrenar. La procedencia se conserva en el reporte del experimento.

El pipeline también permanece disponible como CLI (`pettrace-prepare-e3` o `make prepare-e3`). El [contrato de snapshots y filtros](../docs/e3_data_pipeline.md) describe sus artefactos y límites.

## Fase 1 y evaluación reservada

Esta carpeta implementa la fase 1 del estudio: preparación y preprocesamiento. Las fuentes externas se citan en el notebook y en [references.bib](../docs/references.bib); filtros, muestreo y particiones son decisiones de esta investigación.

La versión v1 ya publicada conserva 282 identidades de train, 60 de validación y 60 de test. E1/E2/E3 comparan el mismo **test reservado de 277 fotos** con PetTrace v4, manteniendo galerías separadas. No se modifica la versión para denominarlo 20 %: representa el 15,2 % de las fotos preparadas. Una vez observado, ese test no se presenta como una prueba nunca consultada ni se usa para seleccionar checkpoints.
