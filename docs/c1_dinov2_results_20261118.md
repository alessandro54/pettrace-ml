# C1: comparación independiente DINOv2-L/14 · 18 de noviembre de 2026

**Estado: ejecutado y publicado en MLflow.** Ambos runs se verificaron mediante API: estado FINISHED, mismos commits/fingerprints que E4 y métricas idénticas a la evidencia local. Este resultado es una referencia externa; no se atribuye a la adaptación LoRA de PetTrace.

| Dominio | Modelo | Aciertos Rank-1 | Rank-1 | Rank-2 | mAP |
|---|---|---:|---:|---:|---:|
| Público | E4 histórico | 237/277 | 85.56% | 96.03% | 0.8345 |
| Público | DINOv2-L congelado | 255/277 | 92.06% | 96.75% | 0.8383 |
| PetTrace v6 | E4 histórico | 433/502 | 86.25% | 94.22% | 0.7333 |
| PetTrace v6 | DINOv2-L congelado | 476/502 | 94.82% | 98.80% | 0.8650 |

## Desglose por especie

| Dominio | Especie | E4 Rank-1 | C1 Rank-1 | E4 mAP | C1 mAP |
|---|---|---:|---:|---:|---:|
| Público | Gatos | 83.33% | 90.12% | 0.7984 | 0.7834 |
| Público | Perros | 88.70% | 94.78% | 0.8853 | 0.9158 |
| PetTrace | Gatos | 86.31% | 90.48% | 0.7642 | 0.8161 |
| PetTrace | Perros | 86.23% | 97.01% | 0.7177 | 0.8897 |

Rank-1 mejora en los cuatro grupos, pero la mAP de gatos públicos baja 0,0151. La ganancia no es uniforme entre métricas ni especies; el agregado global no debe ocultar esta regresión.

## Qué cambia en la investigación

El comparador independiente supera al E4 histórico en Rank-1 en ambos dominios. En público, la diferencia de mAP es pequeña (0,0039); en PetTrace es 0,1318. La mejora previa E2→E3→E4 sigue siendo una ganancia real dentro de CLIP, pero no demuestra superioridad frente a un encoder independiente competitivo.

Antes de añadir una capa E6, la prioridad científica pasa a comparar adaptación sobre DINOv2 con su propio control congelado y a evaluar entre sesiones. Una nueva capa debe mejorar al control fuerte con selección exclusivamente por validación. Añadir complejidad sobre CLIP no queda justificado solo porque el E4 supere a E3.

## Cambios por consulta e incertidumbre

| Dominio | Errores corregidos | Aciertos perdidos | Ganancia neta | Δ Rank-1 (puntos porcentuales) | Δ mAP |
|---|---:|---:|---:|---:|---:|
| Público | 28 | 10 | +18 | +6.50 | +0.0039 |
| PetTrace | 57 | 14 | +43 | +8.57 | +0.1318 |

Intervalos descriptivos del 95 %: bootstrap pareado de 10.000 réplicas, con animales como conglomerados y estratificación por especie; semilla 20261118. Se conserva la galería fija y se remuestrean conjuntamente todas las consultas de cada animal.

- Público: Δ Rank-1 [1.40, 11.85] puntos porcentuales; Δ mAP [-0.0321, 0.0388].
- PetTrace: Δ Rank-1 [4.99, 12.45] puntos porcentuales; Δ mAP [0.0984, 0.1649].

Estos intervalos condicionan en las imágenes de galería y los checkpoints ya fijados. No abarcan variación de optimización, dependencia por nuevas galerías ni generalización entre sesiones. Son exploratorios, no una prueba confirmatoria de superioridad poblacional. El intervalo de mAP público incluye cero.

## Caso Luna: análisis posterior

Con la correspondencia de identidades ya declarada en el notebook Luna, C1 acierta
8/10 consultas de Luna frente a 2/10 de E4. Para Sunset pasa de 4/10 a 5/10 y
para Cat-8 baja de 8/10 a 7/10. Luna todavía presenta dos errores: uno tiene como
rival mejor puntuado a Sunset y otro a Cat-8. Es una inspección posterior de los
resultados congelados, no una condición de selección ni evidencia de que el caso
esté completamente resuelto. El registro está en `luna-posthoc.json`.

## Protocolo y trazabilidad

- DINOv2-L/14, 304.368.640 parámetros congelados; token CLS de 1.024 dimensiones normalizado L2.
- Pesos/procesador HF `facebook/dinov2-large`, revisión `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`.
- Detector y regla de recorte del E2 verificado. Procesador oficial DINOv2: resize del lado corto a 256, centro 224 y normalización ImageNet.
- Sin ajuste, barrido ni selección basada en test. Mismo benchmark cerrado, misma especie, leave-one-out por fotografía; dominios separados.
- Público: 277 fotos, 60 identidades, hashes contrastados con el recibo E4; snapshot compuesto `a724c1099a5432609c853958e88b68c4cf82d871`.
- Privado: 502 fotos, 50 identidades; v6 `52bfe26384c4230008fe4707aa0cb145988766a8`.
- EER solo descriptiva sobre los mismos pares, sin umbral de despliegue validado.
- Inferencia local MPS con fallback CPU para una interpolación no implementada. Paridad en 12 imágenes sintéticas: coseno mínimo 0,99999994 y error absoluto máximo 0,00002448. Se preservó el intento previo fallido antes de evaluar.
- Los pesos originales, configuración y detector tienen hashes registrados. Se conservaron el notebook ejecutado y resultados por consulta; no se registró este control para servir ni se promovieron aliases.

## Límites de la comparación

El backbone, dimensión, procesador y preentrenamiento son distintos; no es una ablación a igual capacidad ni una comparación de latencia. No se ha descartado todo solapamiento entre el preentrenamiento externo y las fotos públicas. La evaluación no garantiza separación por sesión ni contempla mascotas ausentes de la galería. El [estudio multisemilla de E4 ya terminó](e4_sampling_results_20261118.md): el nuevo seleccionado obtiene 85,92 % público y 84,66 % privado de Rank-1, sin mejora general sobre el histórico. Esta tabla conserva su comparación con el E4 histórico fijado; el informe E4 compara además el nuevo seleccionado con C1.

## Evidencia y fuentes

[Notebook ejecutado](../notebooks/c1_dinov2_reference.ipynb). [Comparación gráfica](../reports/c1-dinov2-reference/20261118T063806Z-cc261f74/comparison-vs-e4.pdf). [Diferencias por consulta y bootstrap](../reports/c1-dinov2-reference/20261118T063806Z-cc261f74/comparison-vs-e4.json).

- [MLflow public](https://ml.pettrace.app/#/experiments/3/runs/3e604ebd78254956a2bc3f9a9f9a199c): `3e604ebd78254956a2bc3f9a9f9a199c`.
- [MLflow private](https://ml.pettrace.app/#/experiments/3/runs/39e0a0fc04904406893ca2adbac98e6b): `39e0a0fc04904406893ca2adbac98e6b`.

Método y pesos: [artículo DINOv2](https://arxiv.org/abs/2304.07193), [repositorio oficial](https://github.com/facebookresearch/dinov2), [conversión fijada de Hugging Face](https://huggingface.co/facebook/dinov2-large/tree/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c). Contexto del análisis de incertidumbre: [bootstrap pareado](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) y [remuestreo por conglomerados/estratos](https://www.stata.com/manuals/rbootstrap.pdf).
