# E4: seis corridas de muestreo y reproducibilidad — 18 de noviembre de 2026

**Ejecutado y publicado.** Se verificaron las doce evaluaciones FINISHED, los snapshots, las métricas, el head de control, los seis archivos de selección idénticos y el registro READY de la versión 24. El notebook descargado contiene sus ocho celdas de código ejecutadas sin errores y conserva el mismo código que el protocolo inicial.

**Resultado principal:** cambiar P8×K4 por P16×K2 no produjo una mejora general que justifique reemplazar el E4 histórico. El candidato elegido por validación gana un acierto público y pierde ocho privados; su mAP baja en ambos dominios. DINOv2-L congelado sigue por encima de ese candidato en las dos métricas globales.

## Selección realizada antes de evaluar test

La media de la mejor mAP macro de validación de las tres semillas fue 0.833935 para P8×K4 y 0.838791 para P16×K2. La regla fijada eligió P16×K2 y después la semilla 46 dentro de esa familia. No se cambió la decisión tras observar test. El head E3 de semilla 46 se mantuvo fijo: esta dispersión no representa volver a entrenar toda la cadena E3→E4.

| Dominio | Modelo | Aciertos Rank-1 | Rank-1 | mAP |
|---|---|---:|---:|---:|
| Público | E4 histórico | 237/277 | 85.56% | 0.8345 |
| Público | E4 seleccionado P16×K2 / 46 | 238/277 | 85.92% | 0.8336 |
| Público | C1 DINOv2-L congelado | 255/277 | 92.06% | 0.8383 |
| PetTrace v6 | E4 histórico | 433/502 | 86.25% | 0.7333 |
| PetTrace v6 | E4 seleccionado P16×K2 / 46 | 425/502 | 84.66% | 0.7092 |
| PetTrace v6 | C1 DINOv2-L congelado | 476/502 | 94.82% | 0.8650 |

El mayor Rank-1 público de los seis candidatos es 243/277 (P16×K2, semilla 45), pero no fue el seleccionado por validación. Elegirlo ahora por ese resultado sería selección con test.

## Resultados completos por semilla

★ identifica al candidato seleccionado por validación. No se omiten corridas negativas.

### Público

| P×K | Semilla | mAP val. macro | Aciertos | Rank-1 | Rank-2 | Rank-5 | mAP | MLflow |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 8×4 | 45 | 0.817579 | 231/277 | 83.39% | 96.03% | 99.64% | 0.7966 | [Run](https://ml.pettrace.app/#/experiments/3/runs/115e2ae3410e477c900af9c230ca9987) |
| 8×4 | 46 | 0.847558 | 240/277 | 86.64% | 96.75% | 99.64% | 0.8320 | [Run](https://ml.pettrace.app/#/experiments/3/runs/efb248ea8c78426cac99e4228314b0cf) |
| 8×4 | 47 | 0.836668 | 240/277 | 86.64% | 95.31% | 100.00% | 0.8337 | [Run](https://ml.pettrace.app/#/experiments/3/runs/2389af6098c043dd8ab1417a604c833e) |
| 16×2 | 45 | 0.840604 | 243/277 | 87.73% | 96.75% | 100.00% | 0.8221 | [Run](https://ml.pettrace.app/#/experiments/3/runs/a22b04a4332a4ce2a210fbc9abac0025) |
| 16×2 ★ | 46 | 0.844367 | 238/277 | 85.92% | 94.95% | 100.00% | 0.8336 | [Run](https://ml.pettrace.app/#/experiments/3/runs/7978ff37e44c48a49047ce676e61873b) |
| 16×2 | 47 | 0.831400 | 237/277 | 85.56% | 96.39% | 99.64% | 0.8002 | [Run](https://ml.pettrace.app/#/experiments/3/runs/a448e17949a64ce19a345ea3fdb45099) |

### PetTrace v6

| P×K | Semilla | mAP val. macro | Aciertos | Rank-1 | Rank-2 | Rank-5 | mAP | MLflow |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 8×4 | 45 | 0.817579 | 417/502 | 83.07% | 93.23% | 99.20% | 0.7007 | [Run](https://ml.pettrace.app/#/experiments/3/runs/aede0c3f2dae4678910e0e4abece0ec9) |
| 8×4 | 46 | 0.847558 | 433/502 | 86.25% | 94.22% | 99.80% | 0.7274 | [Run](https://ml.pettrace.app/#/experiments/3/runs/9cf78b21b3e04334bf738a29bcbcc167) |
| 8×4 | 47 | 0.836668 | 419/502 | 83.47% | 93.03% | 99.60% | 0.7253 | [Run](https://ml.pettrace.app/#/experiments/3/runs/7f7ca187076641ab8873707aaa704822) |
| 16×2 | 45 | 0.840604 | 422/502 | 84.06% | 95.42% | 99.00% | 0.7340 | [Run](https://ml.pettrace.app/#/experiments/3/runs/dc06dcf0d43a44478d6272265a25945e) |
| 16×2 ★ | 46 | 0.844367 | 425/502 | 84.66% | 93.63% | 99.20% | 0.7092 | [Run](https://ml.pettrace.app/#/experiments/3/runs/c230a966311144f68a26522858a8c3d4) |
| 16×2 | 47 | 0.831400 | 421/502 | 83.86% | 94.82% | 99.60% | 0.7198 | [Run](https://ml.pettrace.app/#/experiments/3/runs/b1ba20f90b304ccd98ef4f906bfd4b96) |

### Media y dispersión entre tres semillas

| Dominio | P×K | Rank-1 medio ± DE (puntos porcentuales) | mAP media ± DE |
|---|---|---:|---:|
| Público | 8×4 | 85.56 ± 1.88 | 0.8208 ± 0.0210 |
| Público | 16×2 | 86.40 ± 1.16 | 0.8186 ± 0.0170 |
| PetTrace v6 | 8×4 | 84.26 ± 1.74 | 0.7178 ± 0.0149 |
| PetTrace v6 | 16×2 | 84.20 ± 0.41 | 0.7210 ± 0.0125 |

La DE es muestral (n=3), describe optimización y muestreo en este protocolo; no es un intervalo de confianza entre animales. P16 mejora el promedio de Rank-1 público, pero baja ligeramente su mAP; en privado, el promedio de Rank-1 es casi igual y la mAP sube ligeramente. El efecto no es uniforme.

## Diferencias emparejadas del seleccionado frente al E4 histórico

| Dominio | Rescates | Daños | Neto | Δ Rank-1 (puntos) | IC descriptivo 95 % (puntos) | Δ mAP | IC descriptivo 95 % mAP |
|---|---:|---:|---:|---:|---|---:|---|
| PetTrace v6 | 28 | 36 | -8 | -1.59 | [-4.89, 1.83] | -0.0241 | [-0.0412, -0.0078] |
| Público | 10 | 9 | +1 | +0.36 | [-2.27, 2.94] | -0.0008 | [-0.0218, 0.0209] |

Bootstrap pareado de 10.000 réplicas, animales como conglomerados dentro de especie, semilla 20261118. Galería y checkpoint permanecen fijos. No incluye incertidumbre por optimización, galerías nuevas ni sesiones distintas. Es exploratorio; no corrige las múltiples comparaciones ni convierte estos benchmarks ya observados en una confirmación independiente. El intervalo de Δ mAP privado del seleccionado queda por debajo de cero.

Frente a C1, el seleccionado pierde 17 aciertos públicos y 51 privados. Las diferencias emparejadas y sus intervalos están en `verified-study.json`, campo `selected_vs_c1`. DINOv2 cambia backbone, dimensión, preentrenamiento y procesador: esta comparación no aísla una decisión arquitectónica ni iguala capacidad.

## Entrenamiento y exposición real

| P×K | Semilla | Épocas / mejor | Pasos | Presentaciones | Fotos distintas vistas | Minutos de entrenamiento |
|---|---:|---:|---:|---:|---:|---:|
| 8×4 | 45 | 16 / 9 | 1600 | 51200 | 44081 | 16.32 |
| 8×4 | 46 | 30 / 23 | 3000 | 96000 | 73199 | 30.12 |
| 8×4 | 47 | 30 / 25 | 3000 | 96000 | 73154 | 29.98 |
| 16×2 | 45 | 30 / 26 | 3000 | 96000 | 73301 | 28.78 |
| 16×2 | 46 | 28 / 21 | 2800 | 89600 | 69347 | 26.36 |
| 16×2 | 47 | 14 / 7 | 1400 | 44800 | 39373 | 13.20 |

Entrenamiento total medido: 2.41 horas en NVIDIA L4. No incluye preparación, exportación, evaluación o publicación. Evaluación de los modelos exportados en ONNX CPU. No es una comparación de latencia con C1, ejecutado en MPS.

El presupuesto máximo fue común: 30 épocas, 100 pasos/época, 32 imágenes/paso, margen 0,2, learning rate 1e-4, weight decay 1e-4, paciencia 7 y LoRA rango 8/alpha 16. La parada temprana produce distinta exposición efectiva. P16×K2 cambia conjuntamente negativos y positivos por identidad; no aísla solo el número de negativos. El 100 % del corpus estuvo disponible, pero cada corrida solo vio entre 39.373 y 73.301 fotos distintas.

Se entrenaron 688.128 parámetros: LoRA q/v y proyección visual; el backbone base y el head E3 quedaron congelados. Se fijó la semilla antes de construir los adaptadores. El E4 histórico no garantizaba ese orden, por lo que repetir su número de semilla no reproduce necesariamente su inicialización.

## Qué hacer con estos resultados

Conservar el E4 histórico como referencia y presentar esta ronda como prueba de variabilidad y una intervención de muestreo sin mejora general. No sustituirlo por el mejor test de esta ronda. La versión 24 quedó registrada como candidata por la configuración autorizada; no se promovió ningún alias.

Antes de añadir E6, probar adaptación del control DINOv2 y un presupuesto de pasos mayor con selección por validación. Son intervenciones futuras, sin resultados todavía. La ablación LoRA/proyección está preparada pero no entrenó porque falta autorizar los secretos del nuevo notebook. La evaluación entre sesiones requiere anotaciones reales; no hay fechas EXIF disponibles en las 502 fotos privadas.

## Trazabilidad y límites

- Mismo test público inmutable v1 (277 fotos, 60 identidades); v2 aporta recortes preparados. Commit compuesto `a724c1099a5432609c853958e88b68c4cf82d871`.
- PetTrace v6 separado: 502 fotos, 50 identidades; commit `52bfe26384c4230008fe4707aa0cb145988766a8`.
- Corpus preparado: fingerprint `d0ca5c0b1bed8844f943b9fd355ce912524797d0dc71b50c7a1624e39f49fe58`.
- Head E3 modelo 20, semilla 46; SHA-256 `4ff76c5f79bd1acaf6a8732905ee69bcaf9c93983d1e3332142965f18c2cb4ed`.
- CLIP fijado a `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; paquete Colab 0.2.46. Inicialización sembrada explícitamente en el notebook.
- Recuperación cerrada de la misma especie, leave-one-out por fotografía. Rank-k ordena identidades por máximo de similitud; AP/mAP ordena fotos. EER descriptiva, sin umbral de despliegue validado.
- Tres semillas y benchmarks ya observados; no se garantiza separación por sesiones ni búsqueda de animales ausentes de la galería.

[Notebook ejecutado](../notebooks/e4_sampling_multiseed.ipynb) · [Datos verificados](../reports/e4-sampling-session/completed/verified-study.json) · [Figura](../reports/e4-sampling-session/completed/seed-comparison.pdf).

Métodos: [LoRA](https://arxiv.org/abs/2106.09685), [batch-hard y P×K](https://arxiv.org/abs/1703.07737), [CLIP](https://proceedings.mlr.press/v139/radford21a.html), [YOLOv8](https://docs.ultralytics.com/models/yolov8/), [bootstrap pareado](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) y [conglomerados/estratos](https://www.stata.com/manuals/rbootstrap.pdf). Familias, semillas, controles y regla de selección son decisiones de este estudio.
