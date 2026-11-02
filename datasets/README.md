# Datasets de PetTrace

Esta carpeta contiene el pipeline de datos compartido. Los notebooks de `notebooks/` siguen dedicados a los experimentos.

| Dataset | Función | Versiones |
|---|---|---|
| `pettrace-training` | Identidades públicas de perros y gatos para entrenar y validar representaciones | `v1`, `v2`, …; cada versión fija imágenes, splits y fingerprint |
| `pettrace-reid` | Benchmark privado de mascotas de PetTrace | `v4` permanece como referencia de evaluación |

## Pipeline común

Usar [prepare_training.ipynb](prepare_training.ipynb) con el wheel local `0.2.5.dev3`.
Es el único notebook de preparación para los tres volúmenes. Usa Dogs of the World y LCW
y el mismo algoritmo de limpieza. Aplica:

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
  prepare_training.ipynb       # Pipeline incremental; valida CUDA antes del lote
  public/e3-body/              # Fuentes y cachés existentes, fuera de Git
  build/pettrace-training/
    <fingerprint>/            # Imágenes y manifiestos de una versión preparada
    v1.json                   # Nombre estable → fingerprint
    v2.json
```

La adquisición usa cuatro trabajadores por defecto (`workers=4`). Al reanudar verifica por CRC y omite las fotos ya válidas. El paralelismo no cambia la selección de identidades ni obliga a borrar la raíz existente.

El flujo primero restaura solo `prepared`: si la versión solicitada existe, verifica sus hashes y
la reutiliza sin ZIP, adquisición ni YOLO. Si falta, restaura la caché completa y copia únicamente
los ZIP cuya adquisición aún no está completa. La copia de un ZIP necesario sigue siendo completa.
La extracción local usa cuatro trabajadores por lotes de hasta 1.000 fotos; CRC y SHA256 de
cada foto reutilizada se calculan sobre la misma lectura. Los respaldos de adquisición reciben
los archivos escritos; en CUDA solo reciben los resultados del bloque recién terminado, sin recorrer
toda la caché. El modo CPU conserva el recorrido acotado a `animal-screening-cache`. Al terminar o
fallar con una excepción se revisan todas las raíces. Los TAR y recibos anteriores son compatibles.
Las verificaciones de integridad no se omiten y la limpieza conjunta se vuelve a auditar.
El reporte `animal-screening.json` incluye `timings_seconds` de planificación, validación, lectura de
caché CPU, screening CUDA y respaldos. Los tiempos de screening excluyen los respaldos y no incluyen
la curación posterior ni el respaldo final.

Medición local del respaldo (2026-11-01): 20.000 JSON sin cambios y 100 nuevos, tres repeticiones
alternando el orden; medianas de 0,346 s recorriendo la caché y 0,029 s enviando solo archivos terminados.
Ambas rutas guardaron el mismo inventario. Es una prueba de E/S local, no una medición de Drive, GPU
ni del tiempo total de preparación.

El flujo comparte un índice de verificaciones en memoria (`pettrace.data.verification`) durante cada
preparación. Guarda `ruta → (dispositivo, inode, tamaño, mtime, ctime, SHA256)`: una firma igual reutiliza
el hash ya calculado; un cambio obliga a leer el archivo otra vez. Las comparaciones contra el manifiesto
siguen ejecutándose. El índice se descarta al terminar, también si hay un error; no se confía en firmas
de otro kernel. Los hashes de archivos transferidos y bloques de Drive se siguen calculando íntegros.
Este índice supone disco local sin escritores concurrentes: no detecta corrupción silenciosa que no
altere metadatos durante la misma ejecución. Una nueva sesión realiza una verificación nueva.

Medición local del índice (2026-11-01): 400 archivos de 256 KiB y seis etapas de comprobación,
tres repeticiones alternadas. Lecturas: 2.400 → 400; bytes leídos: aproximadamente 600 → 100 MiB;
mediana: 0,325 → 0,064 s. Los hashes coincidieron en todas las etapas. Esta mejora mide únicamente
la verificación local, no inferencia ni el proceso completo en Colab.

Una referencia de HF ya completa, fijada por commit y fingerprint, se verifica localmente sin consultas
HTTP ni solicitud de credenciales. Las referencias por etiqueta sin commit fijo siguen resolviéndose
en HF; un snapshot local corrupto se rechaza.

La orquestación está en el paquete, no en el notebook:

```python
from pettrace.data.workflow import TrainingPreparation

job = TrainingPreparation(drive_root="/content/drive/MyDrive/PetTrace", work_dir="/content",
                          source_percent=50, screening_provider="cuda")
prepared = job.restore()   # Solo refined; devuelve None si todavía falta.
prepared = job.prepare()   # Reutiliza refined o amplía la caché y prepara la versión.
# job.publish()           # Publicación HF explícita e independiente.
```

Drive debe estar montado previamente. Las credenciales se leen del entorno, Colab Secrets o campos
ocultos. `prepare()` también funciona sin llamar antes a `restore()`. El modo CPU sigue disponible
con `screening_provider="cpu"`; el notebook exige CUDA para nuevas detecciones y no cae silenciosamente a CPU.

La GPU usa el [plugin oficial CUDA 12 v0.3.0](https://github.com/microsoft/onnxruntime/releases/tag/plugin-ep-cuda/v0.3.0),
verificado por SHA256, sin reemplazar ONNX Runtime 1.31.0. Se comprueba la asignación de las convoluciones
a CUDA y hasta 32 imágenes (16 por especie) contra CPU antes de procesar el lote pendiente. Deben coincidir
cantidad, clase y orden de detecciones, con tolerancias de 1e-4 en puntuación y 0,1 píxeles en cajas;
también debe mejorar el tiempo mediano de screening. Son criterios de esta investigación, no una garantía
de equivalencia sobre todo el dataset. Los tiempos incluyen preprocesamiento y NMS, sin lectura del disco.
El diagnóstico queda en `animal-screening-cache/cuda/<clave>/gpu-validation.json` incluso si falla.
Las detecciones CPU previas se reutilizan sin reescribir; CUDA tiene caché separada y el reporte distingue
su procedencia. Se requiere Linux x86_64, Python 3.13 y el perfil E3 con Torch/CUDA 12.8.

Subir `dist/v0.2.5.dev3/pettrace_ml-0.2.5.dev3-py3-none-any.whl`, `SHA256SUMS` y
`requirements/e3-runtime.txt` (sin renombrarlo) a `/content` en una sesión nueva.
La celda también acepta el nombre anterior `requirements-e3-runtime.txt`.
Es una compilación local de desarrollo, no una release publicada. Solo una sesión puede preparar
datos en la misma carpeta de Drive; no actualices una sesión que sigue procesando.
Para migrar en la misma VM: interrumpir una vez, esperar el respaldo, instalar el wheel y reiniciar
solo el kernel cuando se solicite. No eliminar el runtime ni sus archivos locales.
La procedencia registra los cambios de implementación; las versiones refinadas existentes no se
sobrescriben. No se ha medido la aceleración con las bases completas en Colab.

La ruta histórica `public/e3-body` se conserva para reutilizar las descargas existentes. Los snapshots generados sirven a E3 y a futuros experimentos; no dependen de volver a adquirir las fuentes. Un nombre local como `v1` no se mueve a otro fingerprint.

## Hugging Face y consumo

`PUBLISH_TO_HF = False` conserva la preparación local y su respaldo en Drive. Al activarlo, el cuaderno solicita `HF_TOKEN` mediante entorno, Secrets o campo oculto y publica únicamente el inventario verificado en el repositorio privado `alessandro54/pettrace-training`, bajo `versions/<versión>/`. La versión es `v2.20`, `v2.50` o `v2.100`, según el porcentaje solicitado. Su etiqueta queda ligada al commit; no se sobrescriben etiquetas existentes con otro dataset. Se necesita permiso de escritura HF para publicar y solo lectura para consumir.

La publicación de datos no activa el registro de modelos ni la publicación de resultados en MLflow. El benchmark `pettrace-reid` está protegido contra usos como repositorio de entrenamiento.

Un experimento puede cargar:

```python
from pettrace.data.versions import load_local_training, load_training

prepared, provenance = load_local_training("datasets/build/pettrace-training/v2.50.json")
# Después de publicar:
# prepared, provenance = load_training(
#     repo_id="alessandro54/pettrace-training", version="v2.50", revision="<commit publicado>")
```

E3 acepta `E3_PREPARED_DATASET` para una versión local. Para HF utiliza `HF_TRAINING_DATASET_REPO`, `HF_TRAINING_DATASET_VERSION`, `HF_TRAINING_DATASET_REVISION` y, opcionalmente, `HF_TRAINING_DATASET_FINGERPRINT`. El loader resuelve la referencia una vez, descarga por commit y valida todos los hashes antes de entrenar. La procedencia se conserva en el reporte del experimento.

El pipeline también permanece disponible como CLI (`pettrace-prepare-e3` o `make prepare-e3`). El [contrato de snapshots y filtros](../docs/e3_data_pipeline.md) describe sus artefactos y límites.

## Fase 1 y evaluación reservada

Esta carpeta implementa la fase 1 del estudio: preparación y preprocesamiento. Las fuentes externas se citan en el notebook y en [references.bib](../docs/references.bib); filtros, muestreo y particiones son decisiones de esta investigación.

La versión v1 ya publicada conserva 282 identidades de train, 60 de validación y 60 de test. E1/E2/E3 comparan el mismo **test reservado de 277 fotos** con PetTrace v4, manteniendo galerías separadas. No se modifica la versión para denominarlo 20 %: representa el 15,2 % de las fotos preparadas. Una vez observado, ese test no se presenta como una prueba nunca consultada ni se usa para seleccionar checkpoints.

## Drive y recuperación entre sesiones (paquete v0.2.4)

El notebook monta Drive con autorización del usuario. Los cálculos siguen en el
disco local de Colab; `MyDrive/PetTrace/checkpoints/preparation` recibe archivos
TAR incrementales y recibos SHA-256. Se respaldan fuentes extraídas, metadatos,
detecciones, la referencia pública y los datasets preparados. Los ZIP originales
son opcionales: `MyDrive/PetTrace/raw/dogs-world.zip` y `lcw.zip`, ambos versión 1.
Se copian a `/content/pettrace-archives` antes de extraer; el índice fijado y los
CRC se validan igual que en adquisición remota. No se confía solo en el nombre.

Al iniciar, el notebook restaura bloques confirmados y valida sus hashes. Cada
cinco minutos, en el siguiente punto de guardado de la adquisición o detección,
se respaldan archivos nuevos o modificados, en bloques de aproximadamente
512 MiB (un archivo individual puede superar ese tamaño). Un cierre normal o
una excepción de Python fuerza otro respaldo. Si desaparece la VM, se conserva
el último bloque confirmado; puede perderse el trabajo posterior. Si Drive falla,
el error se propaga y la copia local permanece. No borrar el runtime antes de
confirmar el respaldo.

Usar una sola sesión escritora por carpeta de respaldos. No borrar ni renombrar
bloques TAR o recibos JSON: una restauración requiere la cadena completa. Los
archivos temporales, directorios ocultos y credenciales `.env` quedan fuera. La
caché SQLite se copia con su API de backup, no mientras se escriben sus bytes.
Mantener las mismas rutas locales entre sesiones: las revisiones de preparación
incluyen rutas absolutas. La migración a otras rutas requiere revalidarlas.

Los tres notebooks E3 recuperan únicamente el dataset preparado (no todas las
fuentes), y reutilizan embeddings de `checkpoints/features`. Los reportes y el
mejor checkpoint de entrenamiento se guardan en `checkpoints/runs/<run-id>`.
Estos respaldos no sustituyen la publicación obligatoria en MLflow. El optimizador
no se reanuda automáticamente tras perder un kernel; los embeddings sí se
recuperan y cada nuevo entrenamiento conserva su inicialización independiente.

Esto conserva la definición actual de 20/50/100% de fotos originales antes del
filtrado. Procesar 100% y luego muestrear porcentajes del entrenamiento limpio es
un cambio metodológico distinto; no se aplica silenciosamente al añadir Drive.
La instalación v0.2.3 de una sesión activa no adquiere estas capacidades hasta
instalar v0.2.4 y reiniciar. No hay respaldo retroactivo de una VM ya eliminada.
