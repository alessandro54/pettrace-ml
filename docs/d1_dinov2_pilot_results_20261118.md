# D1: piloto de adaptación DINOv2 · 18 noviembre 2026

**Estado: tres semillas entrenadas y publicadas; solo validación.**

DINOv2-L/14 congelado con la misma revisión, procesador y recorte E2 que C1. Se entrena únicamente una matriz 1024×1024 sin bias, inicializada como identidad, y se normaliza la salida. No se ha ajustado LoRA ni evaluado test en este piloto.

| Condición | Época seleccionada | mAP macro por especie (val) | Rank-1 macro por especie (val) |
|---|---:|---:|---:|
| C1 congelado | 0 | 0.838703 | 89.467 % |
| D1 semilla 45 | 3 | 0.867272 | 91.751 % |
| D1 semilla 46 | 11 | 0.869427 | 92.058 % |
| D1 semilla 47 | 18 | 0.870904 | 92.365 % |
| D1 media ± DE muestral | — | 0.869201 ± 0.001827 | 92.058 ± 0.307 % |

Estos son valores macro de validación, no los valores globales de test 92,06 % / 94,82 % de C1. Los checkpoints se seleccionaron sobre estas mismas 273 fotos; la mejora es una señal de desarrollo, no confirmación independiente.

## Muestra y protocolo

- 1.040 fotografías train, 228 identidades, recuperadas del caché local y verificadas por SHA-256, especie y split contra las dos fuentes v2 fijadas. Las 236 fotos train restantes del caché antiguo no están en el corpus vigente; el listado de exclusiones y el fallo previo se conservan. No se redefinió el benchmark.
- Validación completa: 273 fotos, identidades disjuntas de train. El estudio completo contempla 193.276 fotos train; este piloto no permite extrapolar su rendimiento.
- Semillas 45–47, P8×K4 alternando especies, batch-hard triplet margen 0,2; AdamW lr 1e-4, weight decay 1e-4; máximo 30×100 pasos, paciencia 7, clipping 5. Época cero elegible, selección por mAP macro.
- Recuperación cerrada por especie, leave-one-out por fotografía. Rank-1 agrega por identidad usando máximo; AP ordena fotos. Sin separación acreditada de sesiones.
- Extracción MPS con fallback CPU, entrenamiento de la proyección CPU. Caché con checksums; publicación interrumpida recuperada sin reentrenar semilla 45. No consumo de Colab en este piloto.

## Interpretación y siguiente prueba

Las tres semillas deben interpretarse como replicas del mismo piloto y validación. Corresponde probar el protocolo con el corpus completo y comparar la familia por validación contra C1 antes de abrir los tests. Mantener D2 (LoRA sin cabeza) como intervención separada. No afirmar superioridad en público, PetTrace o entre sesiones con estos datos.

## Evidencia

[Notebook ejecutado](../notebooks/d1_dinov2_pilot.ipynb). [Módulo de entrenamiento](../src/pettrace/training/dino_metric.py). [Resultados por consulta](../reports/d1-dino-pilot/20261118-local-v1/validation-per-query.json).

[MLflow: piloto y tres semillas](https://ml.pettrace.app/#/experiments/3/runs/c28ea6cec2bc41a4a12435c5f7e08493). Incluye checkpoints, historias, configuración, manifiesto, exclusiones y notebook.

Fuentes de método: [DINOv2](https://arxiv.org/abs/2304.07193), [batch-hard triplet](https://arxiv.org/abs/1703.07737). La métrica lineal y este protocolo no se presentan como una técnica nueva.
