# Distractores externos Oxford-IIIT Pet — 20 de noviembre de 2026

Evaluación sin entrenamiento de los ocho checkpoints E1–E4, con 277 consultas públicas y 502 privadas. Oxford aporta 500 fotografías por especie, añadidas a cada galería por separado. Las 1.000 fotos nunca son consultas ni datos de entrenamiento.

Fuente: [Oxford-IIIT Pet, Parkhi et al., CVPR 2012](https://www.robots.ox.ac.uk/~vgg/data/pets/). Mirror `timm/oxford-iiit-pet@089695c834a7deb60505b7cc506672db1c31a6aa`, partición train usada exclusivamente para evaluación. Licencia CC BY-SA 4.0.

## Resultados con 500 distractores de la misma especie

| Modelo | Público Rank-1 | Público mAP | PetTrace Rank-1 | PetTrace mAP |
|---|---:|---:|---:|---:|
| CLIP-E1 | 66.79% | 51.77% | 64.94% | 37.17% |
| CLIP-E2 | 65.70% | 49.08% | 69.12% | 38.40% |
| CLIP-E3 | 77.98% | 71.57% | 78.29% | 61.69% |
| CLIP-E4 | 84.84% | 80.10% | 85.06% | 68.00% |
| DINOv2-E1 | 89.89% | 82.03% | 94.02% | 81.20% |
| DINOv2-E2 | 91.70% | 81.14% | 93.63% | 81.41% |
| DINOv2-E3 | 91.70% | 85.72% | 91.63% | 79.52% |
| DINOv2-E4 | 93.86% | 89.18% | 92.03% | 81.93% |

## E4 frente a E2 con 500 distractores

Diferencias en puntos porcentuales. Intervalos descriptivos de 10.000 remuestreos pareados de identidades, estratificados por especie; galería y checkpoint fijos. No representan variabilidad entre pools de distractores ni entre entrenamientos.

| Encoder / conjunto | Δ Rank-1 [IC 95%] | Δ mAP [IC 95%] | Rescates / daños |
|---|---:|---:|---:|
| CLIP/public | +19.13 [+11.61, +26.76] | +31.01 [+25.12, +36.76] | 66 / 13 |
| CLIP/private | +15.94 [+10.69, +21.22] | +29.60 [+24.05, +35.05] | 110 / 30 |
| DINOv2/public | +2.17 [-3.09, +7.48] | +8.04 [+4.32, +12.13] | 18 / 12 |
| DINOv2/private | -1.59 [-3.90, +0.59] | +0.51 [-2.01, +3.10] | 18 / 26 |

## Interpretación

CLIP E4 conserva una ventaja amplia sobre E2 en ambos dominios. DINOv2 E4 conserva la mejora de mAP público, pero no demuestra una ventaja de transferencia sobre DINOv2 congelado. En PetTrace, E4 tiene menor Rank-1 observado que E2; ambos intervalos incluyen cero. Esta prueba refuerza la evaluación y también revela un límite del framework. No respalda afirmar que cada etapa mejora todo.

La tasa de primera foto externa en DINOv2 aumenta de E2 a E4: 0,36% a 2,89% en público, y 1,39% a 2,59% en PetTrace. Mejorar el ranking medio no implica ser menos vulnerable a todos los rivales externos.

## Controles y límites

- Protocolo y orden de muestreo fijados antes de medir; un solo pool anidado de 0/100/250/500 fotos por especie, idéntico para todos los checkpoints. No hubo selección usando resultados Oxford.
- SHA-256 contrastados con 193.826 registros públicos y 502 fotos privadas; hashes de píxeles y dHash (distancia <= 4) frente a las 779 fotos del benchmark y distractores previamente retenidos. Cero exclusiones entre las 1.000 seleccionadas. No certifica animales distintos ni elimina todo duplicado transformado.
- Las 16 condiciones originales reprodujeron Rank-1 exactamente y mAP dentro de 0,01 puntos porcentuales. Cachés por lote y evidencia de la primera interrupción MPS preservados; se habilitó respaldo CPU para interpolación bicúbica.
- Oxford tiene categorías de raza/especie, no identidades individuales verificadas. Se reportan fotografías distractoras, Rank-1 y AP por fotografía; no un número certificado de nuevas mascotas ni Rank-5 por identidad externa.
- Todos los queries conservan un positivo. No es una prueba open-set, de rechazo, ni entre sesiones. Solapamiento con preentrenamiento desconocido.
- Checkpoints seleccionados del estudio previo; evidencia retrospectiva exploratoria. No se repite entrenamiento ni se cambia el registro de modelos.

## Evidencia

[Notebook](../notebooks/external_distractors.ipynb), [protocolo](external_distractors_protocol.md), [preparación](../datasets/prepare_oxford_distractors.py). Resultados por consulta, fuentes, hashes, reproducción e intervalos: `reports/external-distractors-20261120/`.

Publicación completada y verificada: [MLflow · daba7785305f4da08200005788e9bc5e](https://ml.pettrace.app/#/experiments/3/runs/daba7785305f4da08200005788e9bc5e), FINISHED; 576 métricas y notebook ejecutado. Recibo `publication.json` con `published=true`.
