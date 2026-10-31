# E1 en Colab

**Baseline congelado: `e1-v1` (2026-10-31).** [Runs, versiones, métricas y hashes](e1-frozen.json). El tag conserva el notebook ejecutado; las mejoras operativas posteriores no cambian esa evidencia. Un cambio de modelo, datos o protocolo requiere otra versión del baseline.

Abre [`e1_clip_zero_shot.ipynb`](../notebooks/e1_clip_zero_shot.ipynb) en una sesión nueva con Python 3.11–3.13.

1. Ejecuta setup. Instala el wheel v0.2.2 y el [perfil de dependencias](../requirements/e1-runtime.txt) con hashes, sin reinstalar las herramientas de Jupyter de Colab.
2. Continúa desde la verificación. **Reinicia la misma sesión solo si setup lo solicita** porque cambió una biblioteca cargada. No elimines el runtime.
3. Proporciona `HF_TOKEN`, `MLFLOW_TRACKING_USERNAME` y `MLFLOW_TRACKING_PASSWORD` mediante Secrets o campos ocultos. La `.env` local no se transfiere a Colab.
4. Ejecuta las tres fases en orden. Las tablas y figuras aparecen debajo de sus celdas.

La última celda también sube los recibos finales a ambos runs. Para archivar el notebook completo, **guárdalo primero con sus salidas** y ejecuta en el equipo donde está ese archivo:

```bash
python tools/attach_notebook.py notebooks/e1_clip_zero_shot.ipynb \
  --run-id RUN_PRIVADO --public-run-id RUN_PUBLICO --env-file .env
```

Sustituye los IDs por los del recibo. El comando comprueba ambos runs, sube el notebook por partes, lo descarga para verificarlo y sincroniza los recibos. Puede reintentarse sin inferencia; rechaza reemplazar un notebook diferente ya archivado. El kernel remoto no puede guardar por sí solo el archivo abierto en VS Code.

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
