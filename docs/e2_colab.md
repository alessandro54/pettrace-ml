# E2 headless en Colab

E2 se ejecuta desde el paquete instalado y el notebook de tesis solo documenta y presenta el
flujo. No requiere Drive ni un checkout. Mantiene CLIP ViT-B/32 congelado e introduce un detector
YOLO congelado: caja gato/perro de mayor confianza, margen de 10 % y fallback a imagen completa.

Evalúa por separado el test público compuesto desde las fuentes refinadas v2 y el snapshot
inmutable `pettrace-reid@v5`, commit
`1777734dbffda0df0bf43305233809ee73157d94`. Para aislar la intervención, recalcula E1 sobre las
mismas consultas y galerías y verifica que los embeddings fallback coincidan.

## Ejecución

Abre [`notebooks/e2_yolo_crop_zero_shot.ipynb`](../notebooks/e2_yolo_crop_zero_shot.ipynb) en
Colab con Python 3.11–3.13, habilita los secretos `HF_TOKEN`, `MLFLOW_TRACKING_USERNAME` y
`MLFLOW_TRACKING_PASSWORD`, y ejecuta las tres fases. El equivalente no interactivo es:

```bash
python -m pip install "pettrace-ml[tracking]==0.2.35"
pettrace run e2 --output reports/e2-v5
```

El reporte incluye auditoría por foto, ejemplos del preprocesamiento real, E1/E2 pareados, CMC,
matrices de confusión, métricas por especie, comparación entre dominios y los dos recibos MLflow.
Una corrida solo está completa cuando `publication.json` contiene `published: true`.

## Recuperación

Si falla únicamente la publicación, conserva el reporte y reintenta sin repetir descargas ni
inferencia:

```bash
pettrace run e2 --resume reports/e2-v5
```

Las salidas ejecutadas de la antigua evaluación pública v1/PetTrace v4 permanecen en
[`notebooks/archive/e2_yolo_crop_zero_shot_v1_v4.ipynb`](../notebooks/archive/e2_yolo_crop_zero_shot_v1_v4.ipynb)
y no se mezclan con una reproducción actual.
