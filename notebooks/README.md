# Cuadernos de investigación · PetTrace

Estos notebooks presentan la investigación sobre re-identificación visual de mascotas realizada
para PetTrace. El objetivo es recuperar **la identidad de un animal**, entre otros individuos de
su especie, a partir de una fotografía. La raza no es la etiqueta de recuperación.

Cada cuaderno articula una pregunta, una intervención experimental y la evidencia necesaria para
evaluarla. La secuencia permite recorrer las decisiones del proyecto: partir de una representación
preentrenada, estudiar la localización del animal y plantear adaptaciones aprendidas. Las secciones
Markdown están escritas en español; identificadores y herramientas compartidas siguen las
convenciones del paquete.

## Recorrido experimental

| Etapa | Pregunta de investigación | Cuaderno y evidencia |
|---|---|---|
| E1 · Línea base | ¿Qué capacidad de recuperación ofrece CLIP sin ajuste específico? | [Notebook](e1_clip_zero_shot.ipynb) · [ejecución v4](https://ml.pettrace.app/#/experiments/3/runs/8601747ecd384d948d429b3b490449c9) |
| E2 · Localización | ¿Cómo cambia la recuperación al recortar la mascota con YOLO y conservar el mismo CLIP? | [Notebook](e2_yolo_crop_zero_shot.ipynb) · [ejecución v4](https://ml.pettrace.app/#/experiments/3/runs/9117658f1945404ca9d1058257bf97f0) |
| E3 · Aprendizaje métrico | ¿Una proyección MLP entrenada con identidades públicas transfiere a las mascotas de PetTrace? | [Notebook](e3_mlp_head_triplet.ipynb) · implementación disponible; entrenamiento real pendiente |
| E4 · Adaptación del encoder | ¿Qué aporta adaptar el backbone mediante LoRA? | Etapa propuesta, sin notebook ni resultados registrados en este índice |
| E5 · Información textual | ¿Puede la descripción de un avistamiento mejorar el orden de candidatos visuales? | Etapa propuesta, sin notebook ni resultados registrados en este índice |

E1 y E2 en v4 comparten el snapshot y el encoder. E2 recalcula su propia línea base y verifica
consultas y galerías antes del contraste. Sus tablas pareadas, auditoría del detector y diagnósticos
por identidad e iluminación permiten examinar beneficios y pérdidas más allá del promedio.

| Indicador registrado en v4 | E1 | E2 |
|---|---:|---:|
| Rank-1 | 65.90 % | 67.87 % |
| Rank-5 | 94.10 % | 92.13 % |
| mAP | 0.3751 | 0.3960 |
| EER descriptiva | 36.23 % | 33.83 % |

Los valores son una síntesis del [índice versionado](../experiments.yaml); MLflow conserva los
artefactos completos. E2 mejora tres de estos indicadores, mientras Rank-5 disminuye. El contraste
es descriptivo y no prueba significancia ni dependencia causal del fondo. Leave-one-out separa
fotografías, no sesiones de captura; la EER se estima y reporta sobre los mismos pares.

E3 configura ahora el mismo commit de v4 y recalcula E1/E2 para la comparación. El piloto
propuesto utiliza Dogs of the World (perros) y LCW (gatos), con revisión explícita de fotos,
auditoría de duplicados y particiones por identidad. La adquisición parcial del piloto está activa; archivos completos, publicación
en MLflow y registro del modelo permanecen desactivados por defecto. El entrenamiento real sigue
pendiente; v4 es un benchmark local exploratorio ya observado, no un test final intacto.
Las ejecuciones [E1 v3](https://ml.pettrace.app/#/experiments/3/runs/af8a88b1e2a6434c9eeb31b62c9010e4)
y [E2 v3](https://ml.pettrace.app/#/experiments/3/runs/e0d3e49aec554a4588efa3bce99955e1) permanecen
como antecedentes sobre otra muestra.

## Estructura de la evidencia

Los cuadernos siguen el recorrido: pregunta e hipótesis → muestra → modelo → entrenamiento cuando
corresponde → evaluación → resultados → discusión → trazabilidad. Las tablas y figuras se generan
con la implementación reutilizable del paquete `pettrace`; no se necesita ejecutar otro notebook
ni disponer de sus reportes. E1/E2 conservan salidas de ejecuciones reales; E3 describe un protocolo
implementado y separa expresamente lo pendiente.

Las fotografías privadas no se incorporan a la exposición de resultados. El análisis usa códigos
seudónimos y registra el commit del dataset, hashes de modelos, configuración y procedencia del
código. Los artefactos locales de `reports/` son salidas de trabajo fuera de Git. MLflow es el
registro completo; `experiments.yaml` permite localizar las ejecuciones desde el repositorio.

## Reproducción de los análisis

Las primeras celdas contienen instalación y configuración, como procedimiento reproducible,
separado del argumento científico. Un kernel local utiliza `uv sync --group dev --extra training`;
un kernel remoto instala `pettrace-ml` si no está presente. `PETTRACE_PACKAGE_SOURCE` permite fijar
una fuente o wheel publicada. El código local sin publicar no está disponible en Colab y no hay
bundles de código codificados dentro de los notebooks.

`HF_TOKEN`, `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` y `MLFLOW_TRACKING_PASSWORD` se
proporcionan mediante entorno, Colab Secrets o campos ocultos. El endpoint de registro es
`https://ml.pettrace.app`; el experimento debe existir. Las configuraciones guardadas de E1 y E2
activan la publicación; E3 la mantiene desactivada por defecto. E1/E2 permiten configurar
`LOG_TO_MLFLOW` o sus variables `E1_LOG_TO_MLFLOW` / `E2_LOG_TO_MLFLOW`. En E3,
`LOG_TO_MLFLOW = False` es literal: solo una edición explícita habilita la publicación.
El registro opcional del modelo E3 es otra configuración y no promueve un alias de servicio.

El mecanismo compartido reutiliza runs coincidentes o reanuda subidas interrumpidas. La publicación
de un mismo resultado no debe ejecutarse desde varios kernels simultáneamente. Los reportes se
escriben bajo `reports/eN_<dataset>`; `PETTRACE_OUTPUT_DIR` permite otra ubicación.

`make notebook NB=notebooks/e1_clip_zero_shot.ipynb` reproduce y guarda un cuaderno en Docker.
`tools/attach_notebook.py` permite adjuntarlo con un recibo de publicación que evita asociarlo a un
run antiguo. En VS Code, el notebook ejecutado se guarda desde el equipo local: el kernel remoto
no puede leer el archivo abierto en el Mac. Los reportes sin publicar y el notebook ejecutado se
conservan antes de cerrar un runtime temporal.

Después de una publicación, `make manifest` actualiza localmente el índice de ejecuciones mediante
una consulta de lectura a MLflow. El índice incluye también runs fallidos o en curso etiquetados
E1–E5; por ello, su estado debe leerse junto con las métricas. El manifiesto se versiona, los reportes
se mantienen fuera de Git.

Detalles operativos: [E1 en Colab](../docs/e1_colab.md) y [protocolo E3](../docs/e3_colab.md).
