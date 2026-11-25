# Pipeline experimental DINOv2

La nueva [secuencia MLP → LoRA, comparable por etapas con CLIP](two_encoder_framework.md) está separada de estas ablaciones lineales. Su [notebook](../notebooks/dino_sequential_pipeline.ipynb) ya completó seis entrenamientos y 16 evaluaciones; véanse los [resultados verificados del 19 de noviembre](dino_sequential_results_20261119.md). Los estados fechados del estudio lineal que siguen son históricos, no el estado actual de la secuencia MLP → LoRA.

Estado al 18 de noviembre de 2026: las tres proyecciones completas están entrenadas y publicadas; sus seis evaluaciones de test también están publicadas. LoRA semilla 45 seguía activo en L4 al verificar el terminal (20 épocas completadas); la desconexión de la interfaz no había detenido el proceso. La selección entre familias y el registro siguen pendientes.

La semilla 46, fijada por validación antes del test de proyecciones, obtuvo 270/277 aciertos públicos (97,47 %, mAP 0,9096) y 480/502 privados (95,62 %, mAP 0,8949). C1 obtuvo 255/277 y 476/502. En PetTrace el intervalo descriptivo de la diferencia de Rank-1 incluye cero. [Notebook de evaluación ejecutado](../notebooks/dino_projection_test.ipynb); evidencia completa en `reports/dino-projection-test/20261118-projection-test-v1/`. Es un análisis intermedio retrospectivo; se fijó la configuración pendiente de LoRA antes de observar estos tests y no debe ajustarse con ellos.

| Condición | Intervención | Parámetros entrenables verificados |
|---|---|---:|
| C1 / frozen | YOLO → procesador DINO → CLS normalizado | 0 |
| D1 / projection | C1 → proyección identidad 1024×1024 → normalización | 1.048.576 |
| D2 / lora | Adaptadores query/value, rango 8, alpha 16; sin cabeza | 786.432 |
| D3 / combined | Proyección y adaptadores entrenados conjuntamente desde C1 | 1.835.008 |

D3 es una ablación conjunta, no la reproducción de E4 histórico: en CLIP la cabeza E3 estaba congelada y la proyección visual propia de CLIP también se ajustaba. D1 es lineal, mientras E3 utilizó una MLP. Las comparaciones emparejadas con CLIP siguen pendientes; no atribuir sus diferencias actuales exclusivamente al backbone. E5 permanece exploratorio y no se ha incorporado a estas condiciones.

## Evidencia ejecutada

[Comprobación técnica](../notebooks/dino_pipeline_smoke.ipynb): ocho fotos train verificadas, 273 val, semilla 45, dos pasos P2×K2 por condición. Comprueba paridad inicial con C1, gradientes, entrenamiento, selección de época cero, recuperación y publicación. Proyección conserva época cero; LoRA y combinación seleccionan época uno. Es una prueba funcional, no evidencia de mejora de rendimiento.

[MLflow verificado](https://ml.pettrace.app/#/experiments/3/runs/3fcd5e4922a749768d41fdb283d9b0f5): checkpoints de las tres condiciones descargados y contrastados por hash; notebook ejecutado publicado y verificado. Los tiempos de 27–29 segundos por condición incluyen validación en MPS y no predicen el coste del estudio completo.

El [piloto anterior D1](d1_dinov2_pilot_results_20261118.md) sí entrenó tres semillas con 1.040 fotos train. Sus mejoras son de validación, no test.

## Estudio completo preparado

[Notebook](../notebooks/dino_pipeline.ipynb): 193.276 train y 273 val, tres semillas (45–47) para cada una de tres familias. Features congeladas cacheadas una sola vez para D1; D2/D3 recalculan la representación. Época cero elegible. Familia por media de mAP macro de validación, con media Rank-1 macro no inferior al control; empate entre familias por nombre y entre semillas por número menor. Si ninguna mejora, se conserva C1.

Se publican los nueve entrenamientos y se congela la selección antes del test. Después se evalúan nueve candidatos y C1: 20 evaluaciones separadas, público v1 inmutable de 277 fotos y PetTrace v6 de 502. No se mezclan galerías. Se generan medias/DE entre semillas, diferencias pareadas y bootstrap descriptivo por animal/especie con galerías fijas. Esa incertidumbre no establece generalización entre sesiones.

La combinación usa lr 1e-5 para ambos componentes. No representa un presupuesto de tuning idéntico al historial CLIP. La variación entre familias mide distintas intervenciones y cantidades de parámetros; el máximo de pasos es común, no el tiempo GPU.

## Ejecución en Colab

[Sesión en L4](https://colab.research.google.com/drive/1tIEfyxCqAKUOO2c0yTv9XeFZuIPpb97x): datos verificados, proyecciones terminadas y LoRA en ejecución. El recibo local está en `reports/dino-pipeline-session/session.json`. Se recuperaron los tres checkpoints de proyección desde MLflow y se verificaron sus hashes; las salidas del notebook de Colab se conservaron antes de reconectar.

El paquete `reports/dino-pipeline-package/pettrace-dino.zip` contiene código, configuración, notebook y manifiesto SHA-256, sin imágenes, credenciales ni resultados privados. Extraerlo en `/content/pettrace-dino` y ejecutar desde el terminal:

```sh
python /content/pettrace-dino/tools/setup_dino_colab.py
```

Abrir `notebooks/dino_pipeline.ipynb`, conectar CUDA y ejecutar sus tres fases. El notebook obtiene HF_TOKEN y las credenciales MLflow del entorno o de Colab Secrets. Su acceso requiere la autorización que Colab solicite. No escribir secretos en celdas ni salidas.

Si se instala el paquete editable después de conectar el kernel, reiniciar la sesión para que Python cargue su ruta. El instalador conserva las versiones de `ipykernel` y `jupyter-client` de Colab. En la primera instalación se restauraron explícitamente las versiones originales 6.17.1 y 7.4.9 antes de ejecutar; el paquete corregido omite esa actualización innecesaria.

Los reportes se crean bajo el directorio de trabajo en `reports/dino-components/20261118-full-v1`. Conservarlos fuera del almacenamiento efímero al terminar y entre sesiones. Los entrenamientos completos se publican individualmente; si falla publicación, la repetición carga el resultado y checkpoint sin entrenarlo otra vez. Un entrenamiento interrumpido antes de guardar su resultado final conserva evidencia; no se reanuda su optimizador automáticamente ni se sobrescribe un adaptador parcial. Usar otra carpeta de estudio para un nuevo intento.

El notebook rechaza reutilizar una carpeta con otro código/configuración. Para una corrida nueva cambiar explícitamente la carpeta del estudio; nunca editar resultados anteriores. El caché comprueba sus hashes.

## Registro e integración

`LOG_TO_MLFLOW=True` y la publicación es incondicional. `REGISTER_MODEL=True` registra solo el candidato seleccionado después de completar evaluaciones, en `pettrace-dinov2-research`; no promueve aliases. El bundle conserva los adaptadores/proyección, revisión HF, procesador, detector fijado y código. Es un artefacto de investigación PyTorch, **no el modelo ONNX compatible con el servicio actual**.

La aplicación sigue usando su modelo publicado. La integración de DINO exige exportación/paridad, medir latencia/memoria y revisar el índice: DINO produce 1.024 dimensiones frente a 512 de CLIP; no deben mezclarse vectores de ambos espacios.

Fuentes: [DINOv2](https://arxiv.org/abs/2304.07193), [LoRA](https://arxiv.org/abs/2106.09685), [triplet batch-hard](https://arxiv.org/abs/1703.07737). No se reivindican estos métodos como nuevos.
