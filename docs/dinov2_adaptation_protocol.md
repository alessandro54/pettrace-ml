# Adaptación sobre DINOv2: siguiente etapa experimental

Fecha: 18 de noviembre de 2026. **Estado actualizado: D1 tiene un piloto reducido ejecutado y publicado con tres semillas; D2 y la combinación tienen comprobación técnica ejecutada, mientras el estudio completo sigue pendiente.** [Resultados del piloto](d1_dinov2_pilot_results_20261118.md): 1.040 fotos train y 273 val; mAP macro medio 0,8692 frente a 0,8387 de C1, Rank-1 macro 92,06 % frente a 89,47 %. Son resultados de validación, no de test. El protocolo siguiente describe el estudio completo prospectivo. La decisión de investigar DINO responde a resultados ya observados; no convierte los benchmarks actuales en confirmación independiente.

## Decisión y evidencia

Priorizar DINOv2-L/14 como base de la siguiente investigación. Conservar E1–E5 como evidencia histórica identificable y CLIP como comparación externa de esta nueva etapa. La ablación de componentes CLIP preparada queda en segundo plano; E5 continúa como exploración. No reasignar sus nombres, pesos o resultados a modelos DINO.

| Dominio | E4 histórico: Rank-1 / mAP | C1 DINO congelado: Rank-1 / mAP |
|---|---:|---:|
| Público, 277 consultas | 237/277 · 85,56 % / 0,8345 | 255/277 · 92,06 % / 0,8383 |
| PetTrace v6, 502 consultas | 433/502 · 86,25 % / 0,7333 | 476/502 · 94,82 % / 0,8650 |

[Evidencia, incertidumbre, especies y recibos C1](c1_dinov2_results_20261118.md). El nuevo E4 elegido por validación obtuvo 238/277 y 425/502; el [estudio de seis corridas](e4_sampling_results_20261118.md) no aportó una mejora general.

La ventaja de DINO es clara en Rank-1 observado. En público, la diferencia global de mAP es solo 0,0039 y su intervalo descriptivo incluye cero; para gatos públicos mAP baja de 0,7984 a 0,7834. Hay 22 errores públicos y 26 privados restantes, pero estos conteos no predicen cuántos puede corregir una adaptación. Luna 8/10 es un diagnóstico posterior, no un objetivo de selección.

Adrián pidió incorporar un competidor independiente competitivo con protocolo comparable (C36). Su comentario estaba anclado al párrafo que mencionaba DINOv2 como trabajo futuro; no ordenó abandonar CLIP ni avaló estos resultados posteriores. El giro se fundamenta en la evidencia nueva.

## Pregunta, hipótesis y aporte posible

**Pregunta:** ¿la adaptación supervisada por identidad mejora la recuperación de mascotas de DINOv2 congelado y conserva esa mejora al separar sesiones de captura?

**Hipótesis:** una adaptación pequeña puede ordenar mejor fotografías de identidades parecidas. También puede sobreajustar o deteriorar la representación; conservar C1 debe ser un resultado válido de selección.

[DINOv2](https://arxiv.org/abs/2304.07193), [LoRA](https://arxiv.org/abs/2106.09685) y [batch-hard triplet](https://arxiv.org/abs/1703.07737) son métodos externos. Usarlos no constituye por sí solo novedad metodológica. La configuración, comparación controlada y validación en el problema de mascotas serían parte de nuestro estudio; una contribución de método exige evidencia adicional y contraste con antecedentes. La aplicación es el contexto de uso.

## Fase 1 · Preparación y verificación de datos

- Cargar el mismo corpus preparado de train/val que el estudio E4: 193.276 fotos de entrenamiento y 273 de validación, con identidades disjuntas. Verificar el manifiesto y fingerprint `d0ca5c0b1bed8844f943b9fd355ce912524797d0dc71b50c7a1624e39f49fe58`; registrar el commit de preparación resuelto. No limpiar datos dentro del experimento.
- Mantener el test público v1 inmutable (277 fotos, 60 identidades). El snapshot compuesto `a724c1099a5432609c853958e88b68c4cf82d871` aporta las cajas de v2 sin redefinir el test.
- PetTrace v6: commit `52bfe26384c4230008fe4707aa0cb145988766a8`, 502 fotos, 50 identidades. No usarlo para entrenamiento, selección de hiperparámetros o checkpoints.
- Mantener detector, margen y política de recorte de C1. Mantener el procesador DINO fijado: lado corto 256, centro 224 y normalización ImageNet. No sustituirlo por el procesador CLIP.
- Verificar ausencia de intersección de identidades y hashes entre train, val y test; investigar coincidencias antes de entrenar, sin eliminar silenciosamente casos del benchmark.
- Para una confirmación posterior, reunir fotografías con sesiones documentadas y congelar una nueva cohorte antes de evaluarla. Las fotos privadas actuales carecen de fecha EXIF de captura; no asignar sesiones por nombre de archivo o semejanza visual.

## Fase 2 · Representación y entrenamiento

Todos los brazos parten de `facebook/dinov2-large`, revisión `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`, CLS de 1.024 dimensiones y normalización L2. Primero medir C1 en validación, con exactamente el mismo evaluador que las variantes.

| Condición | Intervención | Control que permite interpretar |
|---|---|---|
| C1 | Encoder congelado, sin cabeza aprendida | Referencia publicada; también candidato sin intervención |
| D1 | Encoder congelado + proyección lineal 1.024→1.024, sin bias, inicializada como identidad; normalizar salida | Efecto de aprender la métrica sin modificar el encoder |
| D2 | LoRA en las proyecciones de consulta y valor del encoder; salida CLS normalizada, sin cabeza añadida | Efecto de adaptar el encoder sin mezclarlo con D1 |

La identidad inicial de D1 y los adaptadores de salida cero de D2 deben reproducir C1 antes de entrenar. Verificar esto numéricamente, los módulos exactos intervenidos, el conjunto de parámetros entrenables y el flujo de gradientes. La cabeza E3 y los adaptadores CLIP existentes no son pesos transferibles a estos brazos.

Configuración inicial propuesta, pendiente de quedar registrada en el notebook ejecutable antes de entrenar:

- Semillas 45, 46 y 47; fijarlas antes de construir cualquier capa. Tres corridas por brazo, sin escoger semillas por test.
- P8×K4, alternancia de especies y batch-hard triplet con margen 0,2, reutilizando el sampler y la pérdida existentes. Sin aumentos nuevos en esta primera comparación.
- AdamW; tasa D1 `1e-4`, D2 `1e-5`, weight decay `1e-4`. D2: rango 8 y alpha 16. Son decisiones iniciales de este protocolo, no hiperparámetros demostrados óptimos ni una comparación a igual número de parámetros.
- Máximo 30 épocas × 100 pasos, paciencia 7 y clipping de gradiente 5. Es presupuesto de pasos común; no equivale a recorrer todo el corpus ni a igualar tiempo GPU. Registrar pasos, fotos presentadas, fotos e identidades distintas, tiempo, memoria y parámetros entrenables.
- Métrica de checkpoint: media de mAP por especie en validación. Incluir época 0 (C1); desempatar a favor de la época más temprana. Así, un brazo sin mejora conserva su control inicial y se reporta como tal.
- Elegibilidad de familia: media de mAP de validación superior a C1 y media de Rank-1 macro por especie no inferior a C1. Elegir entre familias elegibles por la primera métrica; si ninguna cumple, conservar C1. Para un único candidato de despliegue, elegir la semilla por esa misma mAP de validación, desempate por semilla menor. Reportar las tres semillas, aunque el candidato sea solo una.
- Cachear una sola vez las representaciones congeladas de train/val para D1, con hashes de imágenes, pesos, recorte, procesador y código. D2 debe recalcularlas porque cambia el encoder. No reutilizar embeddings CLIP ni embeddings anteriores al ajuste como salida D2.

Primero realizar un piloto técnico limitado a train/val para medir extracción, memoria, velocidad y consumo real de Colab. Registrar ese piloto y cualquier cambio al protocolo; si cambian presupuesto o precisión, fijarlos antes del estudio y mantenerlos entre semillas. El saldo observado de 295,58 correspondía a unidades de cómputo, no horas ni saldo actual verificado. No comprometer todo el saldo sin medir el coste de DINO.

Actualización al implementar el pipeline: se incorpora D3 como tercera familia, ajustando proyección y LoRA conjuntamente desde C1 (lr 1e-5 para ambos), con tres semillas. Es una intervención distinta de la cadena CLIP histórica y no prueba aún una mejora. [Pipeline y comprobación técnica](dino_pipeline.md). Aumentos, otro tamaño DINO y un comparador local E6 siguen como hipótesis posteriores. La primera comparación debe permitir atribuir el resultado a una intervención concreta.

## Fase 3 · Evaluación, publicación y decisión

Congelar la selección antes de consultar tests. Evaluar las nueve corridas y C1 en los dos dominios por separado, con recuperación cerrada de la misma especie y leave-one-out por fotografía. Rank-k usa máximo por identidad y AP ordena fotografías. Reportar Rank-1/2/5 y mAP globales, por especie y por animal; EER es descriptiva y no establece un umbral operativo para mascotas ausentes.

Publicar medias y desviación entre semillas, diferencias por consulta frente a C1, rescates y daños, e intervalos descriptivos pareados por animal con galería fija. Separar esa incertidumbre de la variación por entrenamiento. Medir latencia y memoria en hardware y lotes comparables antes de recomendar integración a la aplicación.

Los tests actuales ya influyeron en la decisión de investigar DINO: reconocer esa exposición. No seleccionar posteriores cambios mediante el test ni presentar una nueva pasada por las mismas fotos como confirmación independiente. Una evaluación entre sesiones y una cohorte nueva siguen pendientes; tampoco se ha descartado completamente solapamiento con el preentrenamiento externo.

Publicación obligatoria de cada corrida, incluidos resultados negativos y pilotos, en `https://ml.pettrace.app`. Guardar configuración, hashes, métricas por consulta, checkpoints, costes y salidas del notebook; preservar evidencia ante errores y reintentar publicación sin reentrenar. No introducir un interruptor para omitir MLflow. Registro de modelos y promoción de aliases siguen separados; conservar la elección de registro del usuario. Ninguna corrida se declara completa sin recibo verificado.

Una mejora de adaptación debe medirse contra C1, no solo contra CLIP. Si mejora una métrica y empeora otra, reportar la compensación; si C1 gana, conservarlo. No afirmar que DINO adaptado mejora hasta disponer de resultados publicados.

## Alcance técnico comprobado

Reutilizables: `seed_all`, `pk_batch`, `batch_hard_triplet`, `inject_lora` con nombres de módulos verificados, evaluación y publicación existentes. `PreparedPreprocessor.image()` permite reutilizar la política de recorte; `pixels()` aplica CLIP y no debe copiarse a DINO.

Cambios necesarios antes de ejecutar: cabeza y entrenamiento para 1.024 dimensiones, caché DINO separada con procedencia completa, constructor DINO y selección/registro de sus adaptadores, orquestación del estudio, notebook de tres fases y pruebas de paridad inicial, selección sin test, congelación y publicación recuperable. `triplet.Head`, `triplet.fit` y `features.feature_cache` están fijados a 512; el constructor/exportador LoRA actual compone CLIP y la cabeza E3. No cambiar sus contratos históricos al incorporar DINO.

El piloto posterior añade `training/dino_metric.py` y `notebooks/d1_dinov2_pilot.ipynb`, con informe y recibo de publicación propios. Conserva los resultados históricos y no modifica el modelo servido. D2 y la combinación ya tienen comprobación técnica real publicada. La extracción del corpus completo, sus entrenamientos completos y la integración de despliegue siguen pendientes.
