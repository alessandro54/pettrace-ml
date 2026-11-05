# E3 · Volumen de las fuentes públicas

La corrida actual de **E3 50%** usa PetTrace **0.2.14** y la base refinada **v3.50**:
26.968 fotos de entrenamiento, 6.364 identidades. Su notebook fija el fingerprint,
carga la copia local o la recupera desde Drive y reutiliza embeddings compatibles.
En la T4 usada para preparar esta versión, empieza por la celda de verificación.
El setup instala el paquete publicado y sus extras de entrenamiento directamente desde PyPI.
Esta curación incorpora ocho grupos revisados; compararla con v2.50 cambia también
la limpieza, por lo que no mide aisladamente el efecto del volumen.

## Revisión de cierre — 2026-11-03

El [análisis de cierre en MLflow](https://ml.pettrace.app/#/experiments/3/runs/de5b422b2f26462dab85e173149a486d)
conserva las comparaciones por consulta e identidad y el plan de repeticiones.
Con semilla 42, el cambio v2.50 → v3.50 reduce mAP de PetTrace de 0,6430 a 0,6198:
20 de las 31 identidades empeoran. Rank-1 permanece en 73,77%, pero gana 25 consultas
y pierde otras 25. En el test público, mAP pasa de 0,7475 a 0,7552.

Se fijaron las semillas 42, 43 y 44 antes de las nuevas ejecuciones. El runner
[`tools/repeat_e3.py`](../tools/repeat_e3.py) conserva manifiestos, configuración,
dependencias, E2 y los benchmarks; solo cambia la base curada y la semilla pareada.
Entrena las seis MLP antes de evaluar sus tests, reutiliza embeddings verificados y
comprueba la exportación ONNX. Publica doce evaluaciones separadas, conserva los
checkpoints y registra candidatos sin promover alias. El script queda identificado
por su propio SHA-256 además de la versión del paquete. Una repetición de la semilla
42 no constituye una cuarta observación independiente.

Las seis repeticiones terminaron y sus doce evaluaciones tienen recibo de publicación
verificado. Los seis modelos quedaron registrados como candidatos **6–11**, en estado
`READY`, sin alias. Resumen de las semillas 42/43/44 (media ± desviación estándar):

| Benchmark | Curación | Rank-1 | mAP |
|---|---|---:|---:|
| Público v1 | v2.50 | 81,23% ± 1,91 pp | 0,7539 ± 0,0124 |
| Público v1 | v3.50 | 81,83% ± 1,37 pp | 0,7548 ± 0,0019 |
| PetTrace v4 | v2.50 | 75,63% ± 2,67 pp | 0,6555 ± 0,0110 |
| PetTrace v4 | v3.50 | 75,96% ± 2,00 pp | 0,6401 ± 0,0191 |

El mAP privado disminuye en las tres parejas de semillas, tanto en gatos como en
perros; el cambio medio global es −0,0154. En público, el promedio global cambia
apenas +0,0009: gatos pasan de 0,6842 a 0,6737 y perros de 0,8521 a 0,8691.
Ambas curaciones superan el control E2 congelado en estos benchmarks, pero esta
comparación no acredita que las fusiones manuales mejoren la transferencia.
El run de análisis conserva cada resultado, diferencias pareadas, gráficos, hashes
y el inventario de candidatos; no se elige la mejor semilla por su test.

Las repeticiones describen variación del entrenamiento; tres semillas no proporcionan
incertidumbre poblacional sobre nuevas mascotas. El contraste sigue siendo exploratorio
porque ambas galerías ya se habían observado. No se seleccionan hiperparámetros ni
una curación ganadora con el test privado.

**La comparación de volumen todavía no está cerrada.** Falta el inventario completo
del 100%, aplicar una política común a 20/50/100, conservar sus manifiestos y ejecutar
los modelos pendientes. La política elegida para la nueva serie `v4.20/v4.50/v4.100`
es limpieza automática común, sin fusiones manuales; v3.50 queda como contraste separado.
Se mantienen mínimo de cuatro fotos, deduplicación exacta, cuarentena por copias
cercanas/conflictos, filtro YOLO y exclusiones reservadas. Los ocho grupos revisados
en el 50% no autorizan automáticamente fusiones con nuevas fotos o etiquetas. Las fuentes originales
contienen 694.594 fotos; esa cifra no es el número de fotos aptas para entrenamiento.

El inventario del 100% y su auditoría automática están en curso en Colab. El plan
y los scripts están archivados en `volume-plan/` del run de análisis. La continuación
prepara `v4.20`, `v4.50` y `v4.100` solo después de terminar esa auditoría, reutilizando
las cachés compatibles. Su estado queda en
`MyDrive/PetTrace/data-lookup/inventory100/preparation-status.json`; no inicia modelos
ni convierte una preparación parcial en un resultado experimental.

El notebook **E3 100%** ya usa PetTrace **0.2.16** y exige la nueva **v4.100**.
La fase 1 lee su entrada terminada de `data-lookup/inventory100/volume-series.json`,
verifica el fingerprint y el hash del índice de candidatos contra la base refinada,
y conserva ese recibo en el reporte. Si falta la entrada del 100%, se detiene antes
de extraer embeddings o entrenar. Reutiliza cachés compatibles y mantiene publicación
obligatoria y registro de candidato sin alias. Ejecuta la semilla 42; las repeticiones
43/44 siguen siendo necesarias para cerrar la serie. No tiene resultados propios aún.

Para ejecutarlo interactivamente en la sesión que está preparando los datos, empieza
por verificación y ejecuta las celdas siguientes. La verificación espera **antes de
importar Torch y las bibliotecas de ML**, mientras ese proceso de preparación siga
vivo, y exige su recibo exitoso antes de continuar. Usa un kernel sin datos/modelos
anteriores en memoria; la espera no libera objetos de una sesión ya utilizada. No inicia ni
repite la limpieza. Mantén desactivada la cola `run_e3_volume.py` durante esta ejecución
para evitar entrenamientos duplicados. Los embeddings locales se conservan cuando
son posteriores al último respaldo de Drive.

La primera celda de código ejecuta una sola instalación desde PyPI:
`pettrace-ml[training,notebooks]==0.2.16`. En la misma instancia de Colab, reiniciar
solo el kernel no elimina esa instalación; una instancia nueva sí requiere instalar de nuevo.

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

Para preparar cualquiera de los tres volúmenes usa [`datasets/prepare_training.ipynb`](../datasets/prepare_training.ipynb)
con PetTrace `0.2.14` desde PyPI (ver [instalación y cachés](../datasets/README.md)).
Solo una sesión puede escribir en la misma carpeta de respaldos de Drive.
La preparación comparte el algoritmo de limpieza,
y cambia únicamente `SOURCE_SAMPLE_PCT` entre 20, 50 y 100. El código compartido está
en `pettrace.data.workflow.TrainingPreparation`, `pettrace.data.preparation` y `pettrace.data.volume`.

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
La nueva serie automática usa `v4.20`, `v4.50`, `v4.100`; una versión solo está
disponible después de terminar su preparación y verificar el recibo.

## Paquete y ejecución

Los tres notebooks E3 instalan **pettrace-ml 0.2.16** y los extras
`training,notebooks` desde PyPI. No descargan requirements, ni construyen o suben
un wheel. La versión fija el código del paquete; `runtime.json` conserva las versiones
realmente resueltas por pip, el commit, el hash del código y la GPU. Las salidas
históricas conservan la versión con la que fueron producidas.

La presentación conserva visibles la configuración, el ajuste, las métricas, las
figuras y la interpretación. El paquete concentra únicamente el acceso seguro, la
recuperación verificada de runs publicados, la publicación obligatoria de ambos dominios
y el registro idempotente del candidato. Reejecutar la publicación no repite el entrenamiento.

El perfil E3 exige Linux/Python 3.13. PyPI resuelve las versiones compatibles declaradas
por el paquete; el entorno real queda registrado con la corrida. El runtime histórico
del piloto era distinto; sus métricas no acreditan resultados con este nuevo perfil.

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

El notebook 20 conserva la corrida `20261101T174014Z-a7a976eb`, con 138.918 fotos
procesadas, 1.099 fotos y 267 identidades de entrenamiento, y checkpoint de la época 2.
Sus recibos público `5655b80130324470a0a87815b330efb6` y privado
`64e3edcd48ad4d7eb5609ff401f92bda` acreditan publicación; el modelo es la versión 3.
Su fase 3 reutiliza esas métricas, corrigiendo las referencias desactualizadas al
piloto de 1.276 fotos. El notebook 50 conserva su ejecución v3.50 e incorpora las
repeticiones verificadas, con sus runs y gráfico. No se copiaron salidas de una
corrida a otra ni se atribuyeron resultados al 100% o a la nueva serie v4.

Para recuperar los modelos y resultados existentes, cada notebook 20/50 incluye
«Recuperar resultados y modelo publicados» en la fase 3. Con PetTrace instalado,
ejecuta la definición de `configure_private_access` y esa celda: deja disponibles
`published_model.pipeline` y `published_results` por dominio. Reutiliza la copia
local, verifica versión, recibos, hashes y golden set, y no crea una corrida.
Las versiones fijadas son 3 (20%) y 5 (50%, v3.50, semilla 42). Ejecutar todo el
notebook también ejecuta su protocolo de entrenamiento; no es necesario para consultar lo publicado.

Los resultados son descriptivos: una semilla en el 20% y tres en el contraste de
curación del 50%, fotos dependientes por identidad, PetTrace v4 ya observado y sin
garantía de igualdad bit a bit entre GPU. La metodología
y referencias siguen junto a las celdas; la adquisición se documenta en
[el pipeline de datos](e3_data_pipeline.md).

### Persistencia con Drive (v0.2.4)

Las corridas actuales instalan una versión exacta del wheel desde PyPI y registran
el runtime resuelto; los perfiles históricos permanecen en sus releases. Los ZIP
originales opcionales van en `MyDrive/PetTrace/raw/`. El notebook de preparación
restaura y respalda el trabajo local mediante bloques verificados. E3 restaura
las versiones preparadas `v2.20`, `v2.50` o `v2.100` cuando existen y no se ha
seleccionado una versión HF explícita; conserva los controles públicos/privados.

Los embeddings se respaldan entre lotes y los reportes durante/al terminar el
entrenamiento y al publicar. Una sola sesión puede escribir en cada carpeta de
caché compartida. Una interrupción brusca recupera el último respaldo confirmado,
no el estado del optimizador. Ver [detalle y límites](../datasets/README.md#drive-y-recuperación-entre-sesiones-paquete-v024).

### Serie automática 20/50/100

La serie automática v4.20/50/100 puede encadenarse a la preparación con
`tools/run_e3_volume.py`, junto a `repeat_e3.py` y el `volume-plan.json` prospectivo.
Espera los tres recibos completos y la salida del proceso de preparación. Verifica
la política común, reutiliza embeddings por hash y ejecuta las semillas 42/43/44;
publica cada benchmark por separado y registra candidatos sin promover alias.
El estado se guarda en `MyDrive/PetTrace/data-lookup/inventory100/experiment-status.json`.
Una desconexión del navegador no detiene el proceso independiente; la eliminación
de la VM sí. Drive conserva los checkpoints confirmados, no el optimizador de un
entrenamiento interrumpido. Los resultados históricos del 50% no sustituyen esta serie.

### Reintento de publicación tras HTTP 413

En un kernel que ya entrenó con v0.2.4, subir `src/pettrace/artifact_upload.py`
a `/content/artifact_upload.py` y ejecutar únicamente la última celda actualizada.
No reinstalar ni reiniciar: el helper adapta solo el transporte de artefactos y
el publicador conserva la clave del run para reanudarlo. Los JSON locales permanecen
intactos. En MLflow, los JSON grandes se guardan como partes gzip de hasta 512 KiB
y un `<nombre>.transport.json` con tamaños, orden y hashes SHA-256. Para recuperar
el original, concatenar las partes en ese orden, descomprimir gzip y verificar el
hash original del manifiesto. Los modelos y demás archivos conservan sus rutas.
