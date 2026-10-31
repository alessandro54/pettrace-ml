# E1 en Colab

Abre [`e1_clip_zero_shot.ipynb`](../notebooks/e1_clip_zero_shot.ipynb) en una sesión nueva con Python 3.11–3.13.

1. Ejecuta setup. Instala el wheel v0.2.2 y el [perfil de dependencias](../requirements/e1-runtime.txt) con hashes, sin reinstalar las herramientas de Jupyter de Colab.
2. Continúa desde la verificación. **Reinicia la misma sesión solo si setup lo solicita** porque cambió una biblioteca cargada. No elimines el runtime.
3. Proporciona `HF_TOKEN`, `MLFLOW_TRACKING_USERNAME` y `MLFLOW_TRACKING_PASSWORD` mediante Secrets o campos ocultos. La `.env` local no se transfiere a Colab.
4. Ejecuta las tres fases en orden. Las tablas y figuras aparecen debajo de sus celdas.

Los errores de instalación quedan en `e1-install.log`. «Ejecutar todo» se detiene en la verificación si necesita un reinicio.

## Resultados y recuperación

Cada corrida guarda reportes en `REPORT`, con el test público y PetTrace separados. La última celda publica ambos en `https://ml.pettrace.app`; `publication.json` debe contener `published: true`, `run_id` y `public_run_id`. Guarda el notebook para conservar sus salidas.

Si aparece `413` al subir una lámina de fotografías, ejecuta solo la última celda actualizada: prepara vistas JPEG de hasta 512 KiB, conserva los PNG originales junto a `REPORT` (fuera de la carpeta subida) y reintenta la publicación. No recalcula embeddings ni métricas.

Si falla únicamente la publicación, conserva los reportes y reintenta sin repetir inferencia:

```python
from pettrace.experiments.e1 import resume_publication
resume_publication(REPORT)
```

Las salidas históricas no acreditan una nueva corrida. El entorno de referencia es Python 3.11.15: las versiones, datasets y modelo quedan registrados, pero la igualdad de métricas entre plataformas debe comprobarse con corridas completas. [Versiones de Colab](https://research.google.com/colaboratory/runtime-version-faq.html).

El lanzador opcional `tools/run_e1.py` conserva la ejecución histórica aislada; descarga el notebook de la release y produce `e1.executed.ipynb`. El notebook interactivo no lo utiliza.
