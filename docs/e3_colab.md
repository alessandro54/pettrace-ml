# E3 en Colab · estudio multisemilla

El notebook canónico es [`notebooks/e3_mlp_head_triplet.ipynb`](../notebooks/e3_mlp_head_triplet.ipynb).
Se ejecuta desde la UI de Colab con una GPU T4 o superior y no repite preparación de datos.

## Entradas fijadas

- Paquete: `pettrace-ml[training,notebooks]==0.2.26` desde PyPI.
- Entrenamiento: unión pública refinada `v2` de Dogs of the World y LCW, fijada por commits y fingerprint.
- Control: E2 congelado, incluido detector YOLO y encoder CLIP.
- Selección: validación pública por identidades, nunca los tests.
- Evaluación: test público `pettrace-training@v1` y PetTrace `pettrace-reid@v4`, siempre separados.
- Semillas prospectivas: 45, 46, 47, 48 y 49. Las semillas 42–44 son evidencia histórica.

El notebook necesita los secretos de Colab `HF_TOKEN`, `MLFLOW_TRACKING_USERNAME` y
`MLFLOW_TRACKING_PASSWORD`. No guarda credenciales en el archivo.

## Ejecución

1. Abre el notebook desde GitHub en Colab y selecciona GPU.
2. Ejecuta la instalación y la configuración. La primera descarga deja caché efímera en `/content`.
3. La fase 1 carga las releases refinadas v2 y verifica los dos benchmarks.
4. La fase 2 calcula una sola vez las representaciones E2 congeladas y entrena las cinco semillas.
5. Cada semilla selecciona su mejor época por mAP macro-especie de validación.
6. E3 selecciona la semilla ganadora por esa misma métrica; un empate exacto favorece la semilla menor.
7. Solo después evalúa ambos tests y publica dos runs MLflow por semilla.
8. La ganadora se registra como candidata, sin promover alias, y produce `e3-handoff.json`.

Las curvas superponen las cinco semillas y resaltan la ganadora. La tabla final muestra
validación, test público y PetTrace por separado, además de media y desviación estándar.
Esta dispersión describe variabilidad del entrenamiento, no incertidumbre poblacional.

## Fallos y reanudación

Un entrenamiento no está completo hasta que sus dos runs estén publicados. Los modelos,
evaluaciones y recibos quedan en el reporte local de cada semilla, por lo que un fallo de
MLflow se reintenta desde esos artefactos sin volver a entrenar. No se registra ni entrega
un modelo si falta cualquiera de las dos publicaciones.

## Evidencia histórica

Los estudios 20/50/100 y las semillas 42–44 permanecen en MLflow como análisis históricos.
No participan en la selección prospectiva 45–49 y sus notebooks antiguos fueron retirados
para que exista un único protocolo vigente. El E4 ya ejecutado conserva el control que fijó
en su momento; los próximos E4/E5 deberán consumir el handoff nuevo sin reinterpretar aquel run.
