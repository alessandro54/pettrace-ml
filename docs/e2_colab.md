# E2 en Colab

Abre [`notebooks/e2_yolo_crop_zero_shot.ipynb`](../notebooks/e2_yolo_crop_zero_shot.ipynb)
en una sesión nueva con Python 3.11–3.13. Ejecuta setup: instala
`pettrace-ml[notebooks]==0.2.31` directamente desde PyPI y continúa desde la
verificación del wheel. Crea las credenciales en Colab Secrets, habilita el acceso
del notebook y ejecuta la celda de credenciales. Nunca se guardan en el código.

Ejecuta las tres fases en orden. E2 mantiene CLIP congelado e incorpora el detector
YOLO registrado: caja gato/perro de mayor confianza, margen del 10 % y fallback a
imagen completa. No entrena ni descarta fotos cuando falla la detección.

El notebook muestra la auditoría del detector, imágenes antes/después del recorte,
métricas y curvas, ejemplos de recuperación y cambios frente a E1. Evalúa el test
público v1 y PetTrace v4 por separado. Recalcula E1 sobre las mismas consultas y
galerías y guarda esos controles en `baseline_e1/`; no modifica los runs congelados.
Las salidas históricas conservadas no son resultados de una nueva ejecución.

La última celda publica obligatoriamente **dos runs E2**, uno por dominio, y sus
recibos finales. Si falla, repite únicamente esa celda en la misma sesión, sin
recalcular embeddings. Conserva el directorio `REPORT` si debes abandonar la sesión.

Guarda el notebook ejecutado con sus salidas. Desde este checkout actualizado,
adjúntalo a los dos runs usando los IDs de `REPORT/publication.json`:

```bash
uv run --locked python tools/attach_notebook.py notebooks/e2_yolo_crop_zero_shot.ipynb \
  --experiment E2 --run-id RUN_PRIVADO --public-run-id RUN_PUBLICO
```

El adjunto se divide en partes verificadas por SHA-256 para respetar el límite del
servidor. La herramienta rechaza reemplazar un notebook distinto ya archivado.
El kernel remoto de VS Code no puede leer el archivo guardado en tu equipo.
