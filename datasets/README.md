# Datasets de PetTrace

Esta carpeta contiene el pipeline de datos compartido. Los notebooks de `notebooks/` siguen dedicados a los experimentos.

| Dataset | Función | Versiones |
|---|---|---|
| `pettrace-training` | Identidades públicas de perros y gatos para entrenar y validar representaciones | `v1`, `v2`, …; cada versión fija imágenes, splits y fingerprint |
| `pettrace-reid` | Benchmark privado de mascotas de PetTrace | `v4` permanece como referencia de evaluación |

## Pipeline común

Para **consumir una versión refinada** con PetTrace 0.2.8+, sin repetir la preparación:

```python
from pettrace import data

prepared, source = data.load("v2.50", drive_root="/content/drive/MyDrive/PetTrace")
```

Monta Drive previamente. La librería verifica la copia local, recupera solo el snapshot
solicitado de `checkpoints/preparation` o descarga su versión publicada de Hugging Face.
La caché local predeterminada es `datasets/build/pettrace-training`; se puede cambiar con
`cache_dir`. Se conservan la versión, el fingerprint y el commit de HF cuando corresponde.
Las siguientes llamadas reutilizan la copia verificada sin consultar HF ni leer Drive.
Si faltan archivos de una copia existente, se recuperan desde su respaldo/origen conocido.
Los errores de integridad se propagan; no se cambia silenciosamente de versión.

Esta carga solo lee Drive y guarda las descargas de HF localmente. No descarga ZIP raw,
no ejecuta YOLO ni genera versiones nuevas. El acceso privado a HF requiere `HF_TOKEN`.
Los notebooks congelados conservan su configuración y evidencias anteriores.

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

## Revisar exclusiones y cuarentena

El paquete permite contar las identidades descartadas por tener 1, 2 o 3 fotos distintas,
y exportar lotes HTML para revisión visual sin ejecutar YOLO ni modificar el dataset:

```python
from pettrace.data.review import exclusion_summary, quarantine_groups, quarantine_gallery

# Ruta del snapshot terminado, no la carpeta de la preparación que sigue ejecutándose.
snapshot = "/content/datasets/build/pettrace-training/<fingerprint>"
print(exclusion_summary(snapshot))
groups = quarantine_groups(snapshot)  # Solo audit.json; propuestas para revisar, no fusiones.
quarantine_gallery(snapshot, "/content/reports/cuarentena-001.html", start=0, count=8)
```

`exclusion_summary` solo necesita `audit.json` y `excluded.csv`. La recuperación con mínimo
dos es un conteo hipotético con los demás filtros conservados, no una aprobación de las etiquetas.
`quarantine_gallery` requiere además `review.csv` y las imágenes originales en sus rutas registradas;
las fotos excluidas no forman parte de `images/` del snapshot refinado. Verifica sus hashes antes
de mostrar miniaturas. Cada caso reúne las etiquetas conectadas directa o indirectamente por
imágenes exactas. Cada imagen aparece una sola vez, con todas sus etiquetas, archivos y motivos
previos de exclusión. El borde naranja señala las imágenes compartidas entre etiquetas.
Usa `start=8`, `start=16`, etc. para los lotes siguientes (orden del primer SHA256 compartido).
`quarantine_groups` devuelve los mismos grupos y el mapa hash → etiquetas, sin cargar fotos.
Una conexión indirecta puede venir de una etiqueta mezclada: agrupar casos no aprueba fusionarlos.

Descarga el HTML y ábrelo en el navegador: funciona sin widgets, conexión ni GPU. Todos los
casos siguen pendientes; el visor no guarda decisiones ni reincorpora imágenes. Anota el SHA256
compartido, identidad y archivo al decidir si es el mismo animal, una asignación incorrecta o un
caso incierto. Corregir etiquetas o reducir el mínimo requiere una versión nueva del dataset.
Estas funciones están en el código actual del paquete; no actualices el wheel de una preparación activa.

## Data lookup sin YOLO

[`data_lookup.ipynb`](data_lookup.ipynb) corre en **Colab CPU** y monta `MyDrive/PetTrace`.
Elige `SOURCE_SAMPLE_PCT = 20`, `50` o `100`. Recupera el `review.csv` raw de ese volumen
desde `checkpoints/preparation`, con los hashes calculados antes de YOLO. Este inventario
cubre candidatos con etiquetas identificables; no incluye necesariamente cada foto del ZIP.

Solo lee los recibos y copia el bloque TAR que contiene la última versión confirmada del CSV.
Verifica el hash del bloque y del CSV, y extrae únicamente ese archivo en una carpeta separada
bajo `/content/data-lookup/restored`. No restaura imágenes, ZIP, modelos ni la caché completa.
Si el CSV local ya coincide con el respaldo, no vuelve a transferir el bloque. Si falta el
respaldo del porcentaje solicitado, se detiene sin cambiar de volumen ni ejecutar preparación.

Los métodos viven en el paquete:

```python
from pettrace.data.lookup import lookup_duplicates, restore_lookup_review

review = restore_lookup_review("/content/drive/MyDrive/PetTrace", "/content", percent=100)
report = lookup_duplicates(review, "/content/drive/MyDrive/PetTrace/data-lookup")
```

La primera celda instala el paquete directamente desde PyPI, incluidos los gráficos:

```python
%pip install -q "pettrace-ml==0.2.9"
```

Usa una **sesión Colab CPU nueva**. Colab ya proporciona IPython y los widgets; no hace falta
un extra ni instalar un archivo de dependencias. La CI prueba esta instalación normal con
el entorno reportado de Python 3.13.18 y exige que no cambie sus dependencias existentes.
El perfil con hashes permanece en `requirements/data-lookup-colab.txt` para reproducir esa
prueba; no se instala encima del runtime de Colab. No representa toda la imagen de Google.

En tu Mac, `python -m pip install pettrace-ml==0.2.9` instala los métodos y los gráficos.
Para un Jupyter propio, el extra `[notebooks]` añade IPython y widgets.
Después autoriza el montaje de Drive y continúa con las tres fases del notebook.
No reinstales el paquete en un kernel que siga preparando o entrenando datos.

El notebook exporta un representante por SHA256 en `unique-images.csv`, todas sus etiquetas,
las copias sobrantes en `duplicate-copies.csv`, el inventario original y los conflictos para
revisión. Los gráficos distinguen archivos, hashes únicos y fotos distintas por etiqueta.
Los conteos raw abarcan toda la entrada. Aparte, `candidates.csv` indexa candidatos para YOLO:
respeta `drop`, pone identidades conflictivas completas en cuarentena, conserva una copia por
hash (priorizando `keep`) y exige cuatro fotos distintas por etiqueta. `excluded.csv` registra
cada descarte. No fusiona etiquetas ni borra imágenes. Los candidatos pendientes siguen
siendo exploratorios: aún **no son una versión entrenable**.

La preparación consume el índice de forma explícita, en una versión nueva:

```python
from pettrace.data.workflow import TrainingPreparation

preparation = TrainingPreparation(
    drive_root="/content/drive/MyDrive/PetTrace", work_dir="/content",
    source_percent=100, candidate_index=report, version="v3.100",
)
```

`prepare()` exige que el índice coincida con el inventario y sus decisiones antes de enviar
solo los candidatos a YOLO; también reserva val/test v1. Después verifica calidad, duplicados
cercanos y mínimo de fotos, y guarda el índice en la procedencia refinada. El cambio requiere
una nueva versión: no sustituye `v2.20`, `v2.50` ni `v2.100`. Esto reduce entradas a YOLO;
no evita por sí solo recuperar las imágenes originales en la preparación.

Los reportes persisten en `MyDrive/PetTrace/data-lookup`, en carpetas identificadas por hash
con recibo `lookup.json`. Otro runtime recupera el CSV y verifica/reutiliza el mismo reporte.
La revisión visual opcional necesita fotos en las rutas registradas; en una sesión nueva
sin fotos se omite. El conteo confía en los hashes registrados, no vuelve a leer las imágenes.

## Organización

```text
datasets/
  data_lookup.ipynb            # Inventario raw, duplicados exactos y conflictos; CPU, sin YOLO
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
