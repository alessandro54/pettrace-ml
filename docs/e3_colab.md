# E3 · Proyección métrica de identidades públicas a PetTrace v4

La pregunta de E3 es si una MLP 512→256→512 sobre E2 congelado mejora la recuperación de individuos locales. Solo la cabeza se entrena con Batch-Hard Triplet Loss; la validación pública macro por especie selecciona el checkpoint. No hay augmentación de imagen en este piloto.

## Fuentes y límite de adquisición

El notebook propone Dogs of the World v1 para perros y LCW v1 para gatos. El importador de perros lee `metadata/*.json`, `identities[0].identity` y el `path` explícito; descarta entradas con varias identidades. LCW usa `CatFullDataset/<ID>/<imagen>`. Un individuo no se deduce de su raza ni del nombre de su foto.

`ALLOW_DATA_DOWNLOAD = False` evita los archivos completos de unos 96 GB declarados. `ALLOW_PILOT_DOWNLOAD = True` adquiere automáticamente un piloto por rangos HTTP: índices ZIP, un prefijo acotado de metadatos de perros y únicamente las fotos seleccionadas. El servidor debe responder 206 con el rango exacto; de lo contrario la adquisición se detiene sin descargar todo el archivo.

La selección fija 300 identidades por especie y hasta 10 fotos por individuo, y un presupuesto máximo de 3 GiB de red por fuente, incluidos índices y metadatos. Dogs World examina un prefijo de 100.000 JSON según su orden en el archivo; muestrea con semilla dentro de ese prefijo. No equivale a muestreo uniforme sobre toda la fuente. LCW selecciona carpetas elegibles a partir del índice.

El recibo registra configuración, versión, huella del índice ZIP, IDs elegidos, metadatos examinados, bytes de red y hashes de los archivos seleccionados. No registra una huella del archivo completo ni URLs firmadas temporales. Los rangos requieren caché de bloques para metadatos y prelectura por foto para evitar descargar bloques grandes por cada imagen dispersa. Las fotos se verifican por CRC del ZIP durante adquisición y por SHA-256 al reutilizar el piloto. También se verifican los JSON de identidad.

La misma configuración permite reanudar una interrupción y reutilizar una adquisición completa verificada. Para cambiar tamaño/semilla se usa otra raíz de datos. Pueden proporcionarse extracciones manuales con `E3_DOGS_WORLD_ROOT` y `E3_LCW_ROOT`. Desactivar la adquisición parcial permite usar el camino histórico de archivos completos, que requiere activar explícitamente `ALLOW_DATA_DOWNLOAD`.

Los accesos Kaggle opcionales siguen en el entorno; ninguna clave ni URL firmada entra al recibo. Las fuentes públicas inspeccionadas respondieron a solicitudes anónimas. El notebook instala el código publicado en GitHub o la fuente explícita `PETTRACE_PACKAGE_SOURCE`; los wheels viejos en `/content` no se seleccionan automáticamente.

## Revisión antes de entrenamiento

El piloto selecciona como máximo 300 identidades por especie y 10 fotos por identidad con semilla 42. Se generan hojas de contacto por individuo y `review.csv`, donde cada foto requiere `keep` o `drop`. El visor de la sección 1.1 permite decidir con las imágenes visibles, guardar y avanzar por identidad; al reabrir retoma la primera pendiente. No hay aprobaciones preseleccionadas. Guarda decisiones y notas sin modificar etiquetas, rutas ni hashes, y rechaza guardados si el CSV fue editado en otra ventana o cambió una foto. La celda de curación muestra cuántas decisiones faltan y dirige al visor antes de cualquier acceso privado o entrenamiento. El notebook no aprueba fotos automáticamente. Un archivo que no puede decodificarse se muestra con un recuadro de error; `image-readability.json` registra su ruta y el error. La generación continúa con las demás fotos, sin cambiar decisiones: esos archivos deben marcarse `drop` en la revisión. La selección queda vinculada a path, individuo, fuente, especie y hash; una selección o imagen distinta requiere nueva revisión.

Hay que comprobar objetivo inequívoco, identidad consistente, cuerpo útil y variedad observable. LCW contiene también fotos con varios gatos; la caja mayor de un detector no valida el individuo objetivo. La diversidad de fondos no demuestra sesiones diferentes y no se inventan fechas.

Después de revisar se eliminan duplicados y casi duplicados. Se requieren cuatro fotos distintas retenidas por identidad antes de dividir train/val/test por individuo. El entrenamiento requiere al menos ocho identidades por especie y la validación al menos dos. El test público no entra al optimizador ni a la selección; este notebook no publica métricas de ese test.

## Modelo, entrenamiento y comparación

Se verifican hashes del encoder/detector de E2 y su golden set. Se cachean embeddings con identidad del modelo, datos y splits. PyTorch entrena en CUDA con P=8, K=4, margen 0.2, AdamW lr=1e-3, weight decay=1e-4, hasta 30 épocas y paciencia 7; se alternan lotes de gatos y perros. Las curvas y el checkpoint seleccionado quedan en el reporte.

La referencia privada es `pettrace-reid@v4`, commit `8b12db0dce41040a821f9792bea5b352342263ce`. Su etiqueta se verifica y la descarga se fija al commit. Ninguna mascota privada participa del entrenamiento o selección. Se comprueba solapamiento por hashes exactos con los datos públicos; esto no detecta todas las copias transformadas.

El grafo ONNX fusionado se valida y E1/E2 se recalculan sobre la misma muestra y las mismas consultas/galerías. El protocolo local es zero-shot sobre todas las fotos elegibles de v4, sin restringir al split test. La EER es descriptiva sobre los mismos pares y no valida un umbral de producción. v4 ya se observó durante E1/E2 y es un benchmark exploratorio; no debe describirse como test final intacto.

## Evidencia y publicación

Cada corrida crea un directorio nuevo bajo `reports/e3_v4/<fecha-id>`. Contiene fuentes, revisión, manifiesto, auditoría, configuración de entrenamiento, huellas, historia/checkpoint, curvas, artefactos ONNX, líneas base, métricas, comparación, hallazgos y `runtime.json`. Hojas de contacto quedan en el directorio local del piloto y no se adjuntan al reporte.

`LOG_TO_MLFLOW = False` y `REGISTER_E3 = False` son valores literales en la configuración. El entorno no los activa implícitamente. La adquisición pública y las hojas de contacto no solicitan credenciales privadas. Después de completar la revisión, `configure_private_access()` solicita los accesos necesarios para leer modelos MLflow y el benchmark privado de Hugging Face; se mantienen en Secrets, entorno o campos ocultos. Solo cambiar explícitamente `LOG_TO_MLFLOW` permite publicar; registrar requiere también `REGISTER_E3`. No se promueve ningún alias.

El notebook requiere el paquete actualizado instalado; una sesión de Colab no puede importar cambios sin publicar del equipo local. Las pruebas sintéticas verifican el wiring y los límites de datos, no constituyen entrenamiento real ni resultados de E3. La primera corrida real sigue pendiente de los datos revisados y el runtime GPU. `PETTRACE_PACKAGE_SOURCE` puede seleccionar explícitamente una wheel actual si se utiliza ese mecanismo.

## Fuentes primarias

- Dogs of the World: https://www.kaggle.com/datasets/lextoumbourou/dogs-world
- LCW: https://www.kaggle.com/datasets/dseidli/lcwlabeled-cats-in-the-wild

MPDD y Cat Individual Images siguen disponibles en el importador histórico `pettrace.acquisition.prepare`, pero no son las fuentes activas de este piloto. El runner extendido `pettrace.experiments` conserva un protocolo distinto; sus pruebas/calibración públicas no se atribuyen a este notebook.
