# E3 · Proyección métrica de identidades públicas a PetTrace v4

La pregunta de E3 es si una MLP 512→256→512 sobre E2 congelado mejora la recuperación de individuos locales. Solo la cabeza se entrena con Batch-Hard Triplet Loss; la validación pública macro por especie selecciona el checkpoint. No hay augmentación de imagen en este piloto.

## Fuentes y límite de adquisición

El notebook propone Dogs of the World v1 para perros y LCW v1 para gatos. El importador de perros lee `metadata/*.json`, `identities[0].identity` y el `path` explícito; descarta entradas con varias identidades. LCW usa `CatFullDataset/<ID>/<imagen>`. Un individuo no se deduce de su raza ni del nombre de su foto.

`ALLOW_DATA_DOWNLOAD = False` evita descargar por defecto archivos completos cuyo volumen declarado conjunto ronda 96 GB. Se pueden preparar extracciones bajo `E3_DATA_ROOT/dogs-world` y `E3_DATA_ROOT/lcw`, o indicar `E3_DOGS_WORLD_ROOT` y `E3_LCW_ROOT`. La descarga explícita usa una versión fijada, registra SHA-256 y licencia declarada y extrae con el helper seguro. No es una descarga remota por subconjunto: el piloto se selecciona tras la extracción.

Una extracción proporcionada manualmente se registra como tal: su versión de archivo no queda autenticada por un hash del ZIP. La API de Kaggle puede requerir `KAGGLE_USERNAME` y `KAGGLE_KEY` en el entorno; nunca se incluyen claves en el notebook.

## Revisión antes de entrenamiento

El piloto selecciona como máximo 300 identidades por especie y 10 fotos por identidad con semilla 42. Se generan hojas de contacto por individuo y `review.csv`, donde cada foto requiere `keep` o `drop`. El notebook no aprueba fotos automáticamente. La selección queda vinculada a path, individuo, fuente, especie y hash; una selección o imagen distinta requiere nueva revisión.

Hay que comprobar objetivo inequívoco, identidad consistente, cuerpo útil y variedad observable. LCW contiene también fotos con varios gatos; la caja mayor de un detector no valida el individuo objetivo. La diversidad de fondos no demuestra sesiones diferentes y no se inventan fechas.

Después de revisar se eliminan duplicados y casi duplicados. Se requieren cuatro fotos distintas retenidas por identidad antes de dividir train/val/test por individuo. El entrenamiento requiere al menos ocho identidades por especie y la validación al menos dos. El test público no entra al optimizador ni a la selección; este notebook no publica métricas de ese test.

## Modelo, entrenamiento y comparación

Se verifican hashes del encoder/detector de E2 y su golden set. Se cachean embeddings con identidad del modelo, datos y splits. PyTorch entrena en CUDA con P=8, K=4, margen 0.2, AdamW lr=1e-3, weight decay=1e-4, hasta 30 épocas y paciencia 7; se alternan lotes de gatos y perros. Las curvas y el checkpoint seleccionado quedan en el reporte.

La referencia privada es `pettrace-reid@v4`, commit `8b12db0dce41040a821f9792bea5b352342263ce`. Su etiqueta se verifica y la descarga se fija al commit. Ninguna mascota privada participa del entrenamiento o selección. Se comprueba solapamiento por hashes exactos con los datos públicos; esto no detecta todas las copias transformadas.

El grafo ONNX fusionado se valida y E1/E2 se recalculan sobre la misma muestra y las mismas consultas/galerías. El protocolo local es zero-shot sobre todas las fotos elegibles de v4, sin restringir al split test. La EER es descriptiva sobre los mismos pares y no valida un umbral de producción. v4 ya se observó durante E1/E2 y es un benchmark exploratorio; no debe describirse como test final intacto.

## Evidencia y publicación

Cada corrida crea un directorio nuevo bajo `reports/e3_v4/<fecha-id>`. Contiene fuentes, revisión, manifiesto, auditoría, configuración de entrenamiento, huellas, historia/checkpoint, curvas, artefactos ONNX, líneas base, métricas, comparación, hallazgos y `runtime.json`. Hojas de contacto quedan en el directorio local del piloto y no se adjuntan al reporte.

`LOG_TO_MLFLOW = False` y `REGISTER_E3 = False` son valores literales en la configuración. El entorno no los activa implícitamente. Las credenciales MLflow siguen siendo necesarias para leer los modelos existentes. Solo cambiar explícitamente `LOG_TO_MLFLOW` permite publicar; registrar requiere también `REGISTER_E3`. No se promueve ningún alias.

El notebook requiere el paquete actualizado instalado; una sesión de Colab no puede importar cambios sin publicar del equipo local. Las pruebas sintéticas verifican el wiring y los límites de datos, no constituyen entrenamiento real ni resultados de E3. La primera corrida real sigue pendiente de los datos revisados y el runtime GPU. Una wheel local `pettrace_ml-0.1.0-py3-none-any.whl` subida a `/content` se detecta en la celda de instalación; `PETTRACE_PACKAGE_SOURCE` puede seleccionar otra fuente explícitamente.

## Fuentes primarias

- Dogs of the World: https://www.kaggle.com/datasets/lextoumbourou/dogs-world
- LCW: https://www.kaggle.com/datasets/dseidli/lcwlabeled-cats-in-the-wild

MPDD y Cat Individual Images siguen disponibles en el importador histórico `pettrace.acquisition.prepare`, pero no son las fuentes activas de este piloto. El runner extendido `pettrace.experiments` conserva un protocolo distinto; sus pruebas/calibración públicas no se atribuyen a este notebook.
