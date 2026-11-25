# Sensibilidad al número de distractores

Ejecutado y publicado el 20 de noviembre de 2026. Se analizaron CLIP y DINOv2 E1–E4 en ambos dominios (16 condiciones), conservando las consultas y checkpoints publicados.

**Alcance:** valor esperado exacto de Rank-1 al elegir uniformemente identidades rivales de la galería existente, sin reemplazo. No se añadieron animales externos; no se midió separación entre sesiones ni mAP nuevo. No hubo entrenamiento ni selección de modelos.

| Dominio | Especie | Modelo | 5 distractores | 10 distractores | Todos los distractores |
|---|---|---|---:|---:|---:|
| public | cat | CLIP E4 | 96.54% | 93.63% | 83.33% (33) |
| public | cat | DINOv2 E4 | 98.99% | 98.05% | 94.44% (33) |
| public | dog | CLIP E4 | 96.80% | 94.18% | 88.70% (25) |
| public | dog | DINOv2 E4 | 99.83% | 99.65% | 99.13% (25) |
| private | cat | CLIP E4 | 94.53% | 90.20% | 86.31% (16) |
| private | cat | DINOv2 E4 | 97.77% | 95.54% | 92.86% (16) |
| private | dog | CLIP E4 | 96.11% | 93.40% | 86.23% (32) |
| private | dog | DINOv2 E4 | 99.05% | 98.17% | 95.21% (32) |

Para rango r entre N identidades y d rivales, P(Rank-1)=C(N−r,d)/C(N−1,d). La fórmula se verificó contra enumeración exhaustiva para N=2…8; cada extremo de galería completa reproduce la métrica publicada. No hay error Monte Carlo. Las cifras son expectativas sobre subconjuntos, no intervalos de confianza ni nuevos aciertos observados.

La degradación al añadir rivales es estructural; lo informativo es su magnitud para cada representación. DINOv2 E4 conserva mayor Rank-1 esperado que CLIP E4 en los puntos resumidos. Esta evidencia no establece robustez ante identidades externas o galerías más grandes que las disponibles.

[Run publicado](https://ml.pettrace.app/#/experiments/3/runs/b85b07563d0341ffbf4672ad2433274c) · [Notebook ejecutado](../notebooks/distractor_sensitivity.ipynb) · [Curvas completas](../reports/distractor-sensitivity-20261120/curves.json).
