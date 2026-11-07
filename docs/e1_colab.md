# E1 headless

**Baseline histórico congelado: `e1-v1` (2026-10-31).** [Runs, versiones, métricas y hashes](e1-frozen.json). La nueva ejecución headless no altera esa evidencia.

E1 se ejecuta desde el paquete instalado, sin notebook, Drive ni checkout del repositorio. Mantiene CLIP ViT-B/32 congelado y sin detector. Evalúa por separado:

- el test inmutable de `pettrace-training@v1`;
- una versión y commit explícitos del benchmark privado `pettrace-reid`.

El runner genera `summary.json`, `scores.json`, CMC, matriz de confusión, comparación Rank-1/Rank-2/mAP, diagnóstico por identidad y comparación entre dominios. Después publica obligatoriamente ambos reportes en `https://ml.pettrace.app`.

## Ejecución

Instala una release exacta con tracking y exporta las credenciales sin guardarlas en código:

```bash
python -m pip install "pettrace-ml[tracking]==VERSION"
export HF_TOKEN=...
export MLFLOW_TRACKING_URI=https://ml.pettrace.app
export MLFLOW_TRACKING_USERNAME=...
export MLFLOW_TRACKING_PASSWORD=...
```

La condición privada predeterminada es `pettrace-reid@v5`, fijada al commit
`1777734dbffda0df0bf43305233809ee73157d94`:

```bash
pettrace run e1 --output reports/e1-v5
```

`--private-version` y `--private-commit` permiten auditar otra release inmutable, pero deben
proporcionarse juntos y registrarse como una condición de benchmark diferente.

El runner comprueba que la etiqueta resuelva exactamente al commit solicitado y que no haya imágenes idénticas entre los dominios. Nunca mezcla galerías. La diferencia entre ambos resultados es descriptiva porque cambian identidades, tamaño de galería y condiciones de captura.

## Recuperación

La evidencia numérica y los recibos se guardan antes de completar la publicación. Si falla únicamente MLflow, conserva el reporte y reintenta sin descargar imágenes ni recalcular embeddings:

```bash
pettrace run e1 --resume reports/e1-v5
```

`publication.json` debe terminar con `published: true`, `run_id` y `public_run_id`. Una corrida no está completa hasta que ambos runs finalicen correctamente.
