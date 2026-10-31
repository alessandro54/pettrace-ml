# E3 · Proyección métrica de identidades públicas a PetTrace v4

La pregunta de E3 es si una MLP 512→256→512 sobre E2 congelado mejora la recuperación de individuos locales. Solo la cabeza se entrena con Batch-Hard Triplet Loss; la validación pública macro por especie selecciona el checkpoint. No hay augmentación de imagen en este piloto. La preparación se ejecuta por separado y su fingerprint se conserva en cada corrida.

## Separación de datos y entrenamiento

La preparación se ejecuta en `datasets/prepare_training.ipynb`, separado de los notebooks experimentales, o mediante `make prepare-e3` / `pettrace-prepare-e3`. El pipeline produce una versión con imágenes propias, manifiesto, exclusiones, auditoría y hashes. Después `e3_mlp_head_triplet.ipynb` consume esa versión mediante `load_training` o `load_local_training`; no vuelve a adquirir ni filtrar las fuentes. El contrato completo está en [Pipeline de datos](e3_data_pipeline.md).

## Primera corrida con la versión publicada

Abre `notebooks/e3_mlp_head_triplet.ipynb`, selecciona un runtime GPU T4 limpio, instala la release v0.2.1 con la primera celda, reinicia el kernel y continúa desde la verificación del paquete. Véase [instalación versionada](releases.md). El notebook carga por defecto el dataset privado `alessandro54/pettrace-training@v1` (212 MB, 1.826 fotos, 402 identidades), fijado al commit `dd7a00cb8b3638759ffa0216251e62d334b5bf10` y al fingerprint `38ef098d26afa792a1712f9e37487d143f60a9a3bc647a5fdde5505f790f8c7d`. No es necesario ejecutar otra vez la preparación.

Los accesos se obtienen del entorno, Colab Secrets o campos ocultos. El extractor E2 calcula o reutiliza embeddings de las 1.549 fotos de train/val; después la MLP entrena hasta 30 épocas con parada temprana. La salida muestra pérdida y mAP de validación por época. El test público queda reservado. Al finalizar se exporta ONNX y se compara E3 con E1/E2 en el benchmark privado v4. La publicación en MLflow es obligatoria. `REGISTER_E3` controla por separado el registro de una versión del modelo.

Conserva la carpeta `reports/e3_v4/<fecha-id>` antes de cerrar el runtime: incluye `head.pt`, modelo ONNX, curvas y métricas. Colab no guarda esa carpeta en el repositorio automáticamente. Para usar una copia preparada local establece `HF_TRAINING_DATASET_REPO=""` y `E3_PREPARED_DATASET`.

## Fuentes y límite de adquisición

El notebook propone Dogs of the World v1 para perros y LCW v1 para gatos. El importador de perros lee `metadata/*.json`, `identities[0].identity` y el `path` explícito; descarta entradas con varias identidades. LCW usa `CatFullDataset/<ID>/<imagen>`. Un individuo no se deduce de su raza ni del nombre de su foto.

`allow_download=True` en `PrepareConfig` autoriza únicamente el piloto acotado por rangos HTTP: índices ZIP, un prefijo acotado de metadatos de perros y únicamente las fotos seleccionadas. El servidor debe responder 206 con el rango exacto; de lo contrario la adquisición se detiene sin descargar todo el archivo.

La selección fija 300 identidades por especie y hasta 10 fotos por individuo, y un presupuesto máximo de 3 GiB de red por fuente, incluidos índices y metadatos. Dogs World examina un prefijo de 100.000 JSON según su orden en el archivo; muestrea con semilla dentro de ese prefijo. No equivale a muestreo uniforme sobre toda la fuente. LCW selecciona carpetas elegibles a partir del índice.

El recibo registra configuración, versión, huella del índice ZIP, IDs elegidos, metadatos examinados, bytes de red y hashes de los archivos seleccionados. No registra una huella del archivo completo ni URLs firmadas temporales. Los rangos requieren caché de bloques para metadatos y prelectura por foto para evitar descargar bloques grandes por cada imagen dispersa. Las fotos se verifican por CRC del ZIP durante adquisición y por SHA-256 al reutilizar el piloto. También se verifican los JSON de identidad.

La misma configuración permite reanudar una interrupción y reutilizar una adquisición completa verificada. Para cambiar tamaño/semilla se usa otra raíz de datos. Pueden proporcionarse extracciones manuales con `E3_DOGS_WORLD_ROOT` y `E3_LCW_ROOT`. El pipeline independiente no utiliza el camino histórico de archivos completos.

Los accesos Kaggle opcionales siguen en el entorno; ninguna clave ni URL firmada entra al recibo. Las fuentes públicas inspeccionadas respondieron a solicitudes anónimas. El notebook instala la release v0.2.1 de GitHub con dependencias fijadas y hashes; no consume `main` ni wheels arbitrarios de `/content`.

## Control de calidad exploratorio

La primera corrida usa `PrepareConfig.review_policy = "exploratory"`. El piloto selecciona hasta 300 individuos por especie y 10 fotos por individuo con semilla 42. `review.csv` conserva rutas, individuos, fuente, especie y hashes originales. Las decisiones existentes se respetan; las filas pendientes mantienen etiquetas del publicador sin verificación visual individual, y no se reescriben como aprobadas.

El filtrado excluye individuos ambiguos conocidos mediante `EXCLUDED_ANIMALS`, incluyendo el grupo `dogs-world:dog:003e295e97401ebf` con dos perros sin objetivo anotado. Las identidades que comparten una imagen exacta se ponen en cuarentena, sin elegir una etiqueta. Luego se eliminan archivos ilegibles, duplicados y casi duplicados; se requieren cuatro fotos distintas retenidas por identidad antes de dividir train/val/test por individuo. El entrenamiento requiere al menos ocho identidades por especie y la validación al menos dos.

La auditoría registra la política, exclusiones y razones, conflictos, archivos ilegibles, duplicados, decisiones previas y fotos retenidas con etiquetas sin verificar. Antes de curar y dividir los datos, se reutiliza el detector congelado de E2, verificado por hash y golden set. Un paso separado conserva todas las detecciones de perros y gatos, con confianza 0.25 y NMS entre clases con IoU 0.5. E2 conserva su comportamiento original de recorte; el filtro no cambia su pipeline. Solo pasan fotos con exactamente un animal detectado de la especie esperada. Cero detecciones, varios animales, especie inesperada y archivos ilegibles se excluyen por ruta, sin modificar fotos ni decisiones manuales. El requisito de cuatro fotos y los splits se calculan después de estas exclusiones.

En el dataset preparado, `animal-screening.json` registra cajas, puntuaciones, motivos, hashes y parámetros. La caché por foto se vincula a sus bytes, hash del detector, código, tamaño de entrada y versiones del runtime. Los errores de inferencia detienen el proceso en lugar de tratarse como fotos ilegibles. Una sola detección no demuestra identidad: puede haber oclusiones, animales no detectados y falsos positivos. Los umbrales se fijan para este piloto y no se presentan como calibrados.

Después del filtrado se genera una muestra reproducible de cuatro identidades por especie; `visual-sample.json` conserva la selección. Esta muestra no garantiza que las demás etiquetas sean correctas, no detecta todos los casos con varios animales y no demuestra sesiones independientes. La primera corrida es exploratoria bajo posible ruido de etiquetas.

`PrepareConfig.review_policy = "strict"` exige decisiones explícitas en `review.csv`; el visor opcional de `pettrace.data.review` permite completarlas. El pipeline exploratorio no solicita decisiones foto por foto. La revisión no modifica etiquetas, rutas ni hashes. Una selección o imagen distinta requiere volver a generar la revisión. El test público no entra al optimizador ni a la selección. La nueva fase 3 lo evalúa después de fijar el checkpoint, con una galería separada de PetTrace.

## Modelo, entrenamiento y comparación

Se verifican hashes del encoder/detector de E2 y su golden set. Se cachean embeddings con identidad del modelo, datos y splits. PyTorch entrena en CUDA con P=8, K=4, margen 0.2, AdamW lr=1e-3, weight decay=1e-4, hasta 30 épocas y paciencia 7; se alternan lotes de gatos y perros. Las curvas y el checkpoint seleccionado quedan en el reporte.

La referencia privada es `pettrace-reid@v4`, commit `8b12db0dce41040a821f9792bea5b352342263ce`. Su etiqueta se verifica y la descarga se fija al commit. Ninguna mascota privada participa del entrenamiento o selección. Se comprueba solapamiento por hashes exactos con los datos públicos; esto no detecta todas las copias transformadas.

El grafo ONNX fusionado se valida y E1/E2 se recalculan sobre la misma muestra y las mismas consultas/galerías. El protocolo local es zero-shot sobre todas las fotos elegibles de v4, sin restringir al split test. La EER es descriptiva sobre los mismos pares y no valida un umbral de producción. v4 ya se observó durante E1/E2 y es un benchmark exploratorio; no debe describirse como test final intacto.

## Cobertura del dataset en MLflow

`dataset.train_pct` expresa `100 × fotos de train / fotos originales de las dos fuentes`. Para v1 el máximo es 694.594 entradas de imagen: Dogs of the World 313.671 y LCW 380.923. Se contaron los índices ZIP completos y se comprobaron sus hashes contra los recibos del dataset preparado; no fue necesario descargar las fotos originales completas. La referencia queda en `datasets/source_capacity_v1.json` y en la celda de cobertura del notebook para que funcione de forma independiente en Colab.

La primera corrida E3 usa 1.276 fotos para entrenar (**0,1837 %**), y 1.549 incluyendo validación (**0,2230 %**). Los 277 archivos del test público no entran en ninguno de esos numeradores. `dataset.retained_pct` describe las 1.826 fotos conservadas en la versión preparada; no implica que todas hayan entrenado el modelo. El denominador incluye fotos antes de limpieza, no representa el máximo elegible para reidentificación ni un total de individuos.

El reporte `dataset-coverage.json` conserva métricas, conteos y referencias por fuente (versión y hash). En todas las corridas se registran como métricas `dataset.train_pct`, `dataset.train_val_pct` y sus variantes `dataset.cat.*` / `dataset.dog.*`. Para agregar cobertura a una sesión E3 ya completada basta ejecutar la nueva celda de cobertura; no es necesario repetir el ajuste. La publicación obligatoria requiere que esa celda se haya ejecutado.

## Estructura de investigación

E1/E2/E3 comparten tres fases: datos, representación/entrenamiento y evaluación/análisis definidas en [el índice de notebooks](../notebooks/README.md). E3 distingue la selección pública del checkpoint de la evaluación privada, explicita la galería cerrada y conserva `experiment-design.json`. El análisis por identidad se guarda en `analysis.json`; las curvas de pérdida/validación y los hiperparámetros también se registran como métricas y parámetros consultables de MLflow.

Las celdas añadidas durante la estandarización no alteran las salidas históricas. Para generar el nuevo análisis en una sesión ya entrenada se ejecuta de nuevo la comparación de líneas base y las celdas de análisis/publicación; no se repite la optimización de la MLP. Cada reproducción completa crea una carpeta nueva por corrida.

## Evidencia y publicación

Cada corrida crea un directorio nuevo bajo `reports/e3_v4/<fecha-id>`. Contiene fuentes, revisión, manifiesto, auditoría, configuración de entrenamiento, huellas, historia/checkpoint, curvas, artefactos ONNX, líneas base, métricas, comparación, hallazgos y `runtime.json`. Hojas de contacto quedan en el directorio local del piloto y no se adjuntan al reporte.

Todas las corridas de E3 y los demás experimentos deben publicar sus resultados en MLflow. `LOG_TO_MLFLOW = True` es obligatorio y la celda final rechaza `False`. No hay desactivación mediante variables del entorno ni opción CLI `--no-mlflow`. Si una subida falla, se conserva el reporte y se reintenta la publicación sin repetir entrenamiento; la corrida no está completa hasta que exista el recibo de publicación. `REGISTER_E3` controla el registro del modelo por separado y no promueve un alias. Las credenciales permanecen en el entorno, Secrets o campos ocultos; la preparación de datos conserva su flujo independiente.

El notebook requiere el paquete actualizado instalado; una sesión de Colab no puede importar cambios sin publicar del equipo local. Las pruebas sintéticas verifican el wiring y los límites de datos, no constituyen entrenamiento real ni resultados de E3. La corrida `20261030T153523Z-e258882a` ya conserva entrenamiento y evaluación reales: checkpoint de época 3, Rank-1 privado 72,79 % y mAP 0,5682. Su última salida guardada no acredita una publicación; esa fase debe completarse con el recibo correspondiente. Una reproducción necesita el dataset preparado, accesos de lectura/escritura y runtime GPU. La instalación versionada está documentada en [releases.md](releases.md).

## Fuentes primarias

- Dogs of the World: https://www.kaggle.com/datasets/lextoumbourou/dogs-world
- LCW: https://www.kaggle.com/datasets/dseidli/lcwlabeled-cats-in-the-wild

MPDD y Cat Individual Images siguen disponibles en el importador histórico `pettrace.data.acquisition.prepare`, pero no son las fuentes activas de este piloto. El runner extendido `pettrace.experiments.legacy` conserva un protocolo distinto; sus pruebas/calibración públicas no se atribuyen a este notebook.
