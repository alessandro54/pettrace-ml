# E4 · Adaptación LoRA del encoder

Estado: **E4 v1 (semilla 42) está publicado; E4 v2 prueba la semilla 46 sobre el E3 seleccionado**.

## Pregunta y diseño

- **Objetivo:** medir el beneficio adicional de adaptar el encoder visual después de E3.
- **Pregunta:** ¿LoRA sobre la atención de CLIP mejora la recuperación respecto del E3-100 fijado sin perder transferencia a PetTrace?
- **Hipótesis:** una actualización de bajo rango puede capturar señales de identidad que una MLP sobre embeddings congelados no representa.
- **Intervención:** LoRA `r=8`, `alpha=16` únicamente en `q_proj` y `v_proj` de CLIP ViT-B/32.
- **Parámetros congelados:** detector E2, pesos base de CLIP, proyección visual y cabeza MLP de E3.
- **Control:** E3-100 semilla 46, seleccionado entre 45–49 solo por validación antes de evaluar sus tests.

## Procedencia fijada

| Elemento | Valor |
|---|---|
| Datos de entrenamiento | LCW refined `v2` commit `581e44a04fd461ef1ba248177b84aff61a2ada21` + Dogs refined `v2` commit `9fdd6e5f7eaa752f16bd4dae54cd32b88c7e09ba` |
| Equivalencia | Mismos bytes, identidades, splits y fingerprint `d0ca5c0b1bed8844f943b9fd355ce912524797d0dc71b50c7a1624e39f49fe58` de `v4.100`; v2 añade solo cajas YOLO |
| Encoder base | `openai/clip-vit-base-patch32` revisión `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| Cabeza E3 | `report/head.pt` del run privado `cf84059462574151835a7542223a00a4` |
| SHA-256 de la cabeza | `70aa5a8423b3f6a5c36b8c68d32b2417f1e20699f31b510de8747be7509e26aa` |
| Control público E3 | run `922c9e67d53b4c6f924d1339673e059f` |
| Modelo registrado de control | `pettrace-embedder` versión 18, sin promoción de alias |

Antes de entrenar, el notebook verifica commits, manifests, shards y cajas, demuestra que los adaptadores iniciales no cambian la salida del control y que la cabeza descargada coincide con el hash anterior. Drive no es la fuente de datos; se usa únicamente para respaldar una corrida reanudable.

## Entrenamiento y selección

- Pérdida batch-hard triplet, margen `0.2`, lotes de 8 identidades × 4 fotografías y alternancia por especie.
- AdamW, tasa `1e-4`, weight decay `1e-4`, máximo 30 épocas × 100 pasos y paciencia 7.
- Esta corrida exploratoria usa la semilla 46 tanto para el control E3 como para la inicialización y el muestreo de LoRA.
- El checkpoint se selecciona solo por mAP macro por especie en la validación pública.
- El test público y PetTrace no participan del ajuste ni de la selección.
- No se añade aumento de imagen en esta primera comparación para aislar el efecto de LoRA.

## Evaluación y cierre

Se mantienen las galerías separadas del test público v1 y `pettrace-reid@v4`. Se reportan Rank-1/2/5/10, mAP, EER descriptiva, resultados por especie y comparación pareada con E3-100. Cada dominio debe publicarse en MLflow; un fallo de publicación deja la corrida incompleta pero conserva el checkpoint para reintentar sin entrenar otra vez. El registro de candidato es posterior y no mueve alias.

LoRA sigue a Hu et al. (2022). La elección de capas, hiperparámetros, control y protocolo de transferencia es una decisión de esta investigación. Las pruebas unitarias de equivalencia y merge validan implementación, no rendimiento.

## Limitaciones previstas

- El test público y PetTrace ya fueron observados en experimentos anteriores; no son benchmarks vírgenes.
- Una corrida de LoRA no estima variabilidad de optimización ni incertidumbre poblacional.
- No hay garantía de separación entre sesiones fotográficas.
- La comparación no demuestra rendimiento en búsqueda abierta ni a escala de ciudad.
