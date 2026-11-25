# Un pipeline de adaptación, dos encoders

Estado verificado al 20 de noviembre de 2026: la secuencia DINO completó seis entrenamientos y 16 evaluaciones publicadas; selección MLP semilla 46 y LoRA semilla 47. [Resultados y límites](dino_sequential_results_20261119.md). CLIP E1/E2 se completó sobre PetTrace v6 el 19 de noviembre; ambos runs FINISHED, con controles públicos reproducidos. Los resultados lineales anteriores se mantienen separados del MLP. La contribución es una comparación empírica de adaptación, no la invención de CLIP, DINO, YOLO o LoRA.

## Matriz principal

| Etapa | CLIP ViT-B/32 | DINOv2-L/14 | Contraste |
|---|---|---|---|
| E1 | Imagen completa, encoder congelado | Imagen completa, encoder congelado | Referencia sin adaptación |
| E2 | YOLO → encoder congelado | Mismo YOLO → encoder congelado | E2 − E1: localización |
| E3 | E2 → MLP 512→256→512, ReLU y L2 | E2 → MLP 1024→256→1024, ReLU y L2 | E3 − E2: adaptación métrica |
| E4 | MLP E3 congelada; LoRA Q/V + proyección visual nativa entrenables | MLP E3 congelada; LoRA query/value entrenable | E4 − E3: adaptación del encoder |

La proyección visual de CLIP es parte de su arquitectura; DINO no tiene un equivalente que deba inventarse. Utilizar la ablación CLIP solo-LoRA existente para aislar este componente. Mismo flujo no significa igual capacidad, preentrenamiento, resolución, presupuesto de búsqueda o coste. Aquí «sin adaptación» significa recuperación visual con pesos congelados, no clasificación por prompts de texto.

## Implementación compartida y entradas

- `training/triplet.py`: misma Head y entrenador E3; dimensión explícita, 512 por defecto para mantener compatibilidad CLIP. DINO usa 1024 y ancho oculto fijo 256.
- `training/lora.py`: mismo muestreo, pérdida, optimizador, selección y carga de adaptadores; implementaciones de encoder y targets específicas por backbone.
- `training/dinov2.py`: modos nuevos `mlp` y `mlp_lora`. Ambos requieren una cabeza entrenada; E4 la congela y los adaptadores iniciales reproducen E3.
- CLIP conserva `e1_clip_zero_shot.ipynb`, `e2_yolo_crop_zero_shot.ipynb`, `e3_mlp_head_triplet.ipynb`, `e4_clip_lora.ipynb`, `e4_sampling_multiseed.ipynb` y `e4_component_ablation.ipynb`.
- DINO secuencial: [notebook](../notebooks/dino_sequential_pipeline.ipynb), exactamente las tres fases. Los cuadernos `dino_pipeline.ipynb` y `dino_projection_test.ipynb` siguen siendo las ablaciones lineales originales.

## Protocolo fijado para DINO secuencial

193.276 fotos train y 273 val del corpus preparado v2; fingerprint `d0ca5c0b1bed8844f943b9fd355ce912524797d0dc71b50c7a1624e39f49fe58`. Selección únicamente por mAP macro de especies en validación. E3: semillas 45,46,47; seleccionar el mejor MLP, desempate por semilla menor. E4: tres semillas sobre **ese único padre E3 fijado**, guardando hash y run de procedencia; época cero elegible. Esta variación E4 es condicional al padre elegido, no tres réplicas completas E3→E4.

MLP: AdamW lr 1e-3, hidden 256. LoRA: rango 8, alpha 16, lr 1e-5. Ambas etapas: margen 0,2, P8×K4, máximo 30 épocas × 100 pasos, paciencia 7, clipping 5. Registrar parámetros y tiempos: no confundir pasos máximos iguales con igual coste. El lr de DINO no iguala necesariamente al CLIP histórico: presentar el protocolo de cada uno y no afirmar tuning emparejado.

Evaluar **todas** las semillas, no solo la ganadora, y los dos controles: ocho condiciones × dos dominios, 16 publicaciones. Público: test v1 inmutable, 277 fotos/60 identidades, snapshot compuesto `a724c1099a5432609c853958e88b68c4cf82d871`. Privado: PetTrace v6, 502 fotos/50 identidades, commit `52bfe26384c4230008fe4707aa0cb145988766a8`. Nunca mezclar galerías. Las cajas v2 preservan imágenes/identidades/splits v1.

Closed-set de misma especie, leave-one-out por fotografía, Rank-k por máximo de identidad, mAP por fotografía; EER descriptiva. Reportar por dominio y especie, media/DE por semilla, diferencias pareadas, rescates/daños, intervalos por identidad y costes. El notebook calcula E2→E3 y E3→E4. No interpretar los intervalos como validación entre sesiones.

## Ejecución y recuperación

Instalar el código de esta revisión en una sesión CUDA con `python tools/setup_dino_colab.py`; abrir el notebook secuencial. No instalar código nuevo encima de un entrenamiento activo ni sustituir sus salidas. La carpeta nueva es `reports/dino-sequential/20261119-v1`; el diseño y la identidad del código quedan fijados antes de entrenar. Una configuración diferente exige otro directorio.

Publicación incondicional a https://ml.pettrace.app después de cada entrenamiento y evaluación. Reejecutar carga resultados completos y reintenta publicación sin reentrenar. Un checkpoint parcial sin resultado final bloquea la sobrescritura: conservarlo y abrir otro estudio; no hay recuperación de optimizador a mitad del entrenamiento. Guardar los reportes fuera de `/content` antes de destruir un runtime. Registro `REGISTER_MODEL=True` se conserva, separado de publicación y sin promoción de aliases. El bundle E4 incluye **head.pt y adapter.pt**; el adaptador aislado no es el modelo completo.

## Estado de las verificaciones del paper

1. Completado: secuencia DINO publicada; los resultados lineales no se trasladan al MLP.
2. Completado: controles E1/E2 público/v6 con hashes verificados. Runs público `b26c5d8494054beab283f0bf985cb756` y privado `bea86db8772042b0a69714a14534dbc0`; la tabla del paper ya incluye las cuatro etapas en ambos dominios.
3. Alinear y declarar presupuesto/semillas de CLIP y DINO; reportar la ablación de proyección nativa de CLIP. No presentar el historial de búsquedas CLIP como presupuesto igual al de DINO.
4. Evaluación con consultas y galería de sesiones distintas y confirmación independiente. El test ya influyó en esta investigación: el estudio actual es retrospectivo.
5. E5 morfología queda fuera de la contribución visual principal y se reporta como exploratorio. No presuponer que E4 mejora E3.

Fuentes primarias: [CLIP](https://arxiv.org/abs/2103.00020), [DINOv2](https://arxiv.org/abs/2304.07193), [LoRA](https://arxiv.org/abs/2106.09685), [batch-hard triplet](https://arxiv.org/abs/1703.07737), [YOLOv8](https://docs.ultralytics.com/models/yolov8/). La arquitectura MLP, la secuencia y las reglas de selección son decisiones de este estudio.
