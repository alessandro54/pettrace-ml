# E4 · Adaptación LoRA del encoder

Estado: **protocolo prospectivo; todavía no hay resultados de E4**.

## Pregunta y diseño

- **Objetivo:** medir el beneficio adicional de adaptar el encoder visual después de E3.
- **Pregunta:** ¿LoRA sobre la atención de CLIP mejora la recuperación respecto del E3-100 fijado sin perder transferencia a PetTrace?
- **Hipótesis:** una actualización de bajo rango puede capturar señales de identidad que una MLP sobre embeddings congelados no representa.
- **Intervención:** LoRA `r=8`, `alpha=16` únicamente en `q_proj` y `v_proj` de CLIP ViT-B/32.
- **Parámetros congelados:** detector E2, pesos base de CLIP, proyección visual y cabeza MLP de E3.
- **Control:** E3-100 semilla 42, elegido por el protocolo y no por sus métricas de test.

## Procedencia fijada

| Elemento | Valor |
|---|---|
| Datos de entrenamiento | `pettrace-training@v4.100` |
| Encoder base | `openai/clip-vit-base-patch32` revisión `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| Cabeza E3 | `report/head.pt` del run privado `a58f3a42642648f2ab85ab0f83fa8e4b` |
| SHA-256 de la cabeza | `aeccdfb0d91e404c3abe1d5f17ac1429d5a2d2497941e0d68f661c4891a1ea06` |
| Control público E3 | run `82f4c031abc5403ebcdcd3eea1093b36` |
| Modelo registrado de control | `pettrace-embedder` versión 12, sin promoción de alias |

Antes de entrenar, la implementación debe demostrar que los adaptadores iniciales no cambian la salida del control y que la cabeza descargada coincide con el hash anterior.

## Entrenamiento y selección

- Pérdida batch-hard triplet, margen `0.2`, lotes de 8 identidades × 4 fotografías y alternancia por especie.
- AdamW, tasa `1e-4`, weight decay `1e-4`, máximo 30 épocas × 100 pasos y paciencia 7.
- Semillas planificadas: 42, 43 y 44; la primera corrida será 42.
- El checkpoint se selecciona solo por mAP macro por especie en la validación pública.
- El test público y PetTrace no participan del ajuste ni de la selección.
- No se añade aumento de imagen en esta primera comparación para aislar el efecto de LoRA.

## Evaluación y cierre

Se mantienen las galerías separadas del test público v1 y `pettrace-reid@v4`. Se reportan Rank-1/2/5/10, mAP, EER descriptiva, resultados por especie y comparación pareada con E3-100. Cada dominio debe publicarse en MLflow; un fallo de publicación deja la corrida incompleta pero conserva el checkpoint para reintentar sin entrenar otra vez. El registro de candidato es posterior y no mueve alias.

LoRA sigue a Hu et al. (2022). La elección de capas, hiperparámetros, control y protocolo de transferencia es una decisión de esta investigación. Las pruebas unitarias de equivalencia y merge validan implementación, no rendimiento.

## Limitaciones previstas

- El test público y PetTrace ya fueron observados en experimentos anteriores; no son benchmarks vírgenes.
- Tres semillas describen variabilidad de optimización, no incertidumbre poblacional.
- No hay garantía de separación entre sesiones fotográficas.
- La comparación no demuestra rendimiento en búsqueda abierta ni a escala de ciudad.
