# Exposición de entrenamiento y costes registrados

Actualización: 20 de noviembre de 2026. Lectura de artefactos existentes; no son nuevas corridas ni un benchmark de velocidad. Los tiempos publicados abarcan distinto trabajo según el entrenador; CPU de MLP no es comparable directamente con CUDA de LoRA. No incluyen necesariamente descarga, extracción inicial, exportación o publicación.

| Encoder | Condición | Semilla | Pasos | Presentaciones | Fotos únicas | Parámetros entrenables | Segundos registrados | Backend |
|---|---|---:|---:|---:|---:|---:|---:|---|
| CLIP | p16-k2-seed45 | 45 | 3000 | 96000 | 73301 | 688128 | No consolidado | cuda (L4) |
| CLIP | p16-k2-seed46 | 46 | 2800 | 89600 | 69347 | 688128 | No consolidado | cuda (L4) |
| CLIP | p16-k2-seed47 | 47 | 1400 | 44800 | 39373 | 688128 | No consolidado | cuda (L4) |
| CLIP | p8-k4-seed45 | 45 | 1600 | 51200 | 44081 | 688128 | No consolidado | cuda (L4) |
| CLIP | p8-k4-seed46 | 46 | 3000 | 96000 | 73199 | 688128 | No consolidado | cuda (L4) |
| CLIP | p8-k4-seed47 | 47 | 3000 | 96000 | 73154 | 688128 | No consolidado | cuda (L4) |
| DINOv2 | combined | 45 | 3000 | 96000 | 73233 | 1835008 | 5985.04 | cuda |
| DINOv2 | lora | 46 | 3000 | 96000 | 73199 | 786432 | 5990.88 | cuda |
| DINOv2 | lora | 47 | 3000 | 96000 | 73154 | 786432 | 5986.89 | cuda |
| DINOv2 | mlp | 45 | 1800 | 57600 | 48805 | 525568 | 329.95 | cpu |
| DINOv2 | mlp | 46 | 2200 | 70400 | 57561 | 525568 | 405.41 | cpu |
| DINOv2 | mlp | 47 | 1800 | 57600 | 48759 | 525568 | 327.82 | cpu |
| DINOv2 | mlp_lora | 45 | 2200 | 70400 | 57808 | 786432 | 4394.99 | cuda |
| DINOv2 | mlp_lora | 46 | 1100 | 35200 | 31884 | 786432 | 2194.64 | cuda |
| DINOv2 | mlp_lora | 47 | 2900 | 92800 | 71333 | 786432 | 5706.16 | cuda |

Los contadores son acumulados hasta la parada, no hasta el checkpoint seleccionado. Fotos únicas se cuenta dentro de cada corrida; no se deben sumar como imágenes diferentes entre semillas.

La MLP DINO tiene 525.568 parámetros; sus adaptadores LoRA, 786.432. El conjunto combinado lineal tiene 1.835.008 y no debe confundirse con E4 secuencial. CLIP E4 combina 294.912 parámetros LoRA y 393.216 de proyección nativa.

La selección histórica de CLIP E3 y el coste completo de inferencia todavía deben incorporarse para cerrar C16/C36. Las diferencias de hardware, caché y carga de validación impiden presentar estos tiempos como una comparación de eficiencia entre encoders.

Fuentes por fila:
- DINOv2 mlp_lora seed 47: `e75c53869b0e4b3fbca2f548dbb379d8`.
- DINOv2 lora seed 46: `5e1909f4e9ae47f0a9551df350072c7c`.
- DINOv2 lora seed 47: `eaade411fa2141dc97137e421d405d2d`.
- DINOv2 mlp seed 45: `1826964031ee46469b32771a91e8f397`.
- DINOv2 mlp seed 46: `0b46d1e179a1491897b2bbbe9130b85d`.
- DINOv2 combined seed 45: `e132bb1349604498aa36df8bd793ed15`.
- DINOv2 mlp_lora seed 45: `59c75a8f5b88421985b08bbd296be09b`.
- DINOv2 mlp_lora seed 46: `c5a1d8eef6a140f0bcf0b072077cd08a`.
- DINOv2 mlp seed 47: `3dccefa2a7564d7e9dcf42693e2cd1c1`.
- CLIP p16-k2-seed45 seed 45: `reports/e4-sampling-session/completed/p16-k2-seed45/private/training-summary.json`.
- CLIP p8-k4-seed46 seed 46: `reports/e4-sampling-session/completed/p8-k4-seed46/private/training-summary.json`.
- CLIP p8-k4-seed47 seed 47: `reports/e4-sampling-session/completed/p8-k4-seed47/private/training-summary.json`.
- CLIP p8-k4-seed45 seed 45: `reports/e4-sampling-session/completed/p8-k4-seed45/private/training-summary.json`.
- CLIP p16-k2-seed46 seed 46: `reports/e4-sampling-session/completed/p16-k2-seed46/private/training-summary.json`.
- CLIP p16-k2-seed47 seed 47: `reports/e4-sampling-session/completed/p16-k2-seed47/private/training-summary.json`.
