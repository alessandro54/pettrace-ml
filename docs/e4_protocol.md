# E4 · Adaptación LoRA del encoder

Estado: **congelado**. E4 v2 (semilla 46, época 9) fue publicado en ambos dominios y el
candidato 19 quedó registrado sin promoción de alias. E5 debe cargar esa versión numérica y
verificar sus hashes mediante `pettrace.experiments.e5.load_frozen_e4(model_uri, output)`; no puede seleccionar otro
E4 usando resultados de test ni un alias mutable.

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

Antes de entrenar, el notebook verifica commits, manifests, shards y cajas, demuestra que los adaptadores iniciales no cambian la salida del control y que la cabeza descargada coincide con el hash anterior. Hugging Face es la fuente inmutable de datos y MLflow conserva los resultados, artefactos y modelos publicados; E4 no usa Google Drive.

## Entrenamiento y selección

- Pérdida batch-hard triplet, margen `0.2`, lotes de 8 identidades × 4 fotografías y alternancia por especie.
- AdamW, tasa `1e-4`, weight decay `1e-4`, máximo 30 épocas × 100 pasos y paciencia 7.
- Esta corrida exploratoria usa la semilla 46 tanto para el control E3 como para la inicialización y el muestreo de LoRA.
- El checkpoint se selecciona solo por mAP macro por especie en la validación pública.
- El test público y PetTrace no participan del ajuste ni de la selección.
- No se añade aumento de imagen en esta primera comparación para aislar el efecto de LoRA.

## Evaluación y cierre

Se mantienen las galerías separadas del test público v1 y `pettrace-reid@v4`. Se reportan Rank-1/2/5/10, mAP, EER descriptiva, resultados por especie y comparación pareada con E3-100. Cada dominio debe publicarse en MLflow; un fallo de publicación deja la corrida incompleta, pero puede reintentarse desde el reporte local mientras el runtime siga activo. El registro de candidato es posterior y no mueve alias.

| Dominio | E3 Rank-1 | E4 Rank-1 | E3 mAP | E4 mAP | E3 EER | E4 EER |
|---|---:|---:|---:|---:|---:|---:|
| Test público v1 | 80.1 % | **85.6 %** | 0.764 | **0.813** | 9.5 % | **8.4 %** |
| PetTrace v4 | 78.4 % | **83.6 %** | 0.688 | **0.722** | 12.8 % | **11.7 %** |

Recibos: público `2302b302772b4349894e1d1b58d438ed`; PetTrace `5bc216d03afc4a31a15c53fc3bfbc747`.

## Diagnóstico de sensibilidad en inferencia (validación)

Con E4 congelado se compararon seis configuraciones de representación y agregación usando
exclusivamente el split de validación. `single + max` fue el control; no hubo reentrenamiento ni
acceso al test público v1 o a PetTrace v4.

| Representación | Agregación por identidad | Delta Rank-1 | Delta mAP |
|---|---|---:|---:|
| `single` | `max` | 0.000000 | 0.000000 |
| `horizontal_flip_tta` | `max` | -0.007613 | +0.001724 |
| `single` | `top2_mean` | -0.019771 | 0.000000 |
| `horizontal_flip_tta` | `prototype` | -0.021472 | +0.001724 |
| `single` | `prototype` | -0.023062 | 0.000000 |
| `horizontal_flip_tta` | `top2_mean` | -0.028974 | +0.001724 |

El control obtuvo el mejor Rank-1 macro por especie. Por tanto, la vista única y el máximo por
identidad quedan defendidos empíricamente frente al *pooling* esférico, `top2_mean` y el flip
horizontal. Aunque TTA elevó el mAP macro en 0.001724, redujo Rank-1 en todas sus combinaciones; no
cumplió el criterio de selección predefinido. Las variantes se descartaron en validación y no se
evaluaron en ninguno de los dos tests, evitando seleccionar el protocolo con evidencia de test.

Evidencia MLflow: run [`8d3549cfa0114529972ea1b42b90b1e5`](https://ml.pettrace.app/#/experiments/3/runs/8d3549cfa0114529972ea1b42b90b1e5).

LoRA sigue a Hu et al. (2022). La elección de capas, hiperparámetros, control y protocolo de transferencia es una decisión de esta investigación. Las pruebas unitarias de equivalencia y merge validan implementación, no rendimiento.

## Limitaciones

- El test público y PetTrace ya fueron observados en experimentos anteriores; no son benchmarks vírgenes.
- Una corrida de LoRA no estima variabilidad de optimización ni incertidumbre poblacional.
- No hay garantía de separación entre sesiones fotográficas.
- La comparación no demuestra rendimiento en búsqueda abierta ni a escala de ciudad.
