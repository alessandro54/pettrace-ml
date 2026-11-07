# Experimentos de investigación · PetTrace

Los notebooks se reservan para experimentos y preparación/preprocesamiento de datos. Los lanzadores y herramientas de instalación se mantienen como scripts Python, sin notebooks adicionales.

Los experimentos consumen releases refinadas e inmutables de Hugging Face. La capa v2 agrega
localización YOLO sin cambiar fotos, identidades o splits.

Los tres cuadernos documentan una secuencia de investigación de tesis sobre recuperación visual de identidades individuales de perros y gatos. Cada condición modifica un componente del pipeline y conserva el mismo benchmark privado de evaluación. La raza no es la etiqueta del problema.

| Experimento | Intervención | Parámetros ajustados | Comparación |
|---|---|---|---|
| [E1 histórico](e1_clip_zero_shot.ipynb) | CLIP ViT-B/32 sin localización previa | Ninguno | Presentación congelada; las nuevas corridas usan `pettrace run e1` headless |
| [E2](e2_yolo_crop_zero_shot.ipynb) | Recorte YOLOv8n antes del mismo CLIP | Ninguno | E1 recalculado con las mismas consultas y galerías |
| [E3](e3_mlp_head_triplet.ipynb) | MLP 512→256→512 sobre E2 congelado; barrido 45–49 | Solo la MLP | E1/E2 congelados; ganador elegido solo con validación pública |
| [E4](e4_clip_lora.ipynb) | LoRA en `q_proj`/`v_proj` de CLIP sobre un E3 fijado | Solo adaptadores LoRA | Control E3 declarado antes de entrenar |

El [protocolo de E4](../docs/e4_protocol.md) congela la semilla 46, época 9 y candidato 19 como
control visual de E5. E5 aún no está implementado: falta fijar una fuente de atributos observables
que no revele identidad antes de crear su notebook de tres fases.

## Tres fases de la investigación

1. **Preparación y preprocesamiento de datos.** Las releases públicas refinadas conservan la limpieza, deduplicación, splits y localización auditados; los experimentos solo cargan y verifican snapshots. PetTrace v4 procede de su flujo privado de curación/exportación.
2. **Representación y entrenamiento.** Se verifica el modelo y se calculan representaciones. E1/E2 mantienen todos los pesos congelados. E3 entrena la MLP y selecciona su checkpoint únicamente con validación pública; muestra curvas de pérdida y validación.
3. **Evaluación y análisis de resultados.** Cada condición se evalúa por separado en el test público de v1 y PetTrace v4. Incluye tablas por especie, Rank-1/Rank-2/mAP, CMC, matrices de confusión, diagnósticos por identidad, discusión, referencias y publicación obligatoria en MLflow.

Las fuentes externas se citan junto al método utilizado, además de la lista de referencias: CLIP (Radford et al., 2021), YOLOv8 (Ultralytics), batch-hard triplet (Hermans et al., 2017), contexto de CMC/mAP (Zheng et al., 2015) y errores biométricos (Fiumara/NIST, 2022). Las reglas de galería, muestreo, filtros e hiperparámetros se identifican como decisiones propias. [references.bib](../docs/references.bib) conserva las entradas para la tesis. Los resultados y gráficos calculados por el proyecto no se atribuyen a estos trabajos.

## Muestra y protocolo comparables

El benchmark es `pettrace-reid@v4`, commit `8b12db0dce41040a821f9792bea5b352342263ce`: 305 fotos de 31 mascotas. La descarga se fija a ese commit después de comprobar la etiqueta. E1/E2 no entrenan con estas fotos. E3 entrena con identidades públicas y selecciona el checkpoint únicamente mediante validación pública; v4 es una referencia exploratoria ya observada.

La recuperación es de galería cerrada y de la misma especie anotada, con evaluación leave-one-out por fotografía. Rank-k utiliza el máximo score por identidad; AP/mAP ordena fotografías. `zero-shot` es la etiqueta técnica del protocolo compartido, no clasificación mediante prompts de texto. No existe separación garantizada entre sesiones. La EER se ajusta y reporta sobre los mismos pares y es descriptiva.

## Comparación pública y privada en las nuevas ejecuciones

La muestra pública de evaluación es el test ya publicado de `pettrace-training@v1`: **277 fotos, 60 identidades** (34 gatos y 26 perros). Es el 15,2 % de las fotos preparadas y el 14,9 % de sus identidades; no se cambia v1 para forzar un 20 %. Los datos proceden de Dogs of the World y LCW, ambos distribuidos en Kaggle. Train/val no se incorporan a la galería pública de evaluación.

E1, E2 y E3 usan exactamente este mismo test y el mismo snapshot PetTrace v4. Se producen `public_test/` y los reportes privados del directorio principal, `benchmark-comparison.json` y `benchmark-comparison.png`. Se publica un run MLflow por dominio con su input de dataset; `publication.json` enlaza `run_id` privado y `public_run_id` público. La diferencia entre dominios no es una estimación causal: cambian la galería y las condiciones de captura. Las salidas históricas no incluyen aún esa comparación.

## Evidencia histórica de referencia

E1 tiene un baseline congelado, **`e1-v1`**, con las corridas pública y privada verificadas. [Registro de cierre](../docs/e1-frozen.json): incluye el commit y hash del notebook ejecutado, modelo, datasets y métricas para las comparaciones posteriores.

| Indicador global | E1 | E2 | E3 |
|---|---:|---:|---:|
| Rank-1 | 65,90 % | 67,87 % | 72,79 % |
| Rank-5 | 94,10 % | 92,13 % | 96,39 % |
| mAP | 0,3751 | 0,3960 | 0,5682 |
| EER descriptiva | 36,23 % | 33,83 % | 16,76 % |

E1: [run publicado](https://ml.pettrace.app/#/experiments/3/runs/8601747ecd384d948d429b3b490449c9). E2: [run publicado](https://ml.pettrace.app/#/experiments/3/runs/9117658f1945404ca9d1058257bf97f0). E3: corrida `20261030T153523Z-e258882a`, conservada en las salidas del notebook; su última salida de publicación guardada indica que no se publicó en ese momento. Una configuración actualizada no acredita una publicación retrospectiva. El recibo `publication.json` de la sesión confirma su estado real.

E3 seleccionó la época 3 (mAP macro de validación pública 0,6603) y se detuvo en la época 10. Entrenó con 1.276 fotos de 282 identidades; validó con 273 fotos de 60 identidades. En esa corrida histórica, las 277 fotos de 60 identidades del test público no se evaluaron. Las nuevas reproducciones las evaluarán después de fijar el checkpoint, separadas de PetTrace. Las fuentes son Dogs of the World v1 y LCW v1, combinadas y limpiadas en `pettrace-training@v1`.

La métrica `dataset.train_pct = 0,1837 %` expresa cobertura de las 694.594 entradas de imagen originales, antes de limpieza. No es el split de entrenamiento ni el porcentaje de individuos elegibles. Versiones y hashes del denominador están en [source_capacity_v1.json](../datasets/source_capacity_v1.json).

Los resultados describen una muestra pequeña y una sola semilla de E3. No se reportan significancia, intervalos de confianza ni generalización entre sesiones; tampoco equivalen a precisión en una búsqueda abierta de mascotas perdidas.

## Conservación y reproducción

Los notebooks se ejecutan celda a celda en la UI de Colab e instalan una versión exacta de `pettrace-ml` desde PyPI. E1/E2 usan el extra `[notebooks]`; E3/E4 añaden `[training]`. No copian perfiles de requirements, construyen wheels ni montan Google Drive. Hugging Face aporta los datasets inmutables y MLflow conserva la evidencia publicada y los modelos. Una celda visible obtiene las credenciales con `google.colab.userdata`; el paquete solo consume las variables de entorno resultantes. Una `.env` local no se transfiere automáticamente. `runtime.json` conserva el entorno real de cada corrida. Véanse las guías de [E1](../docs/e1_colab.md), [E2](../docs/e2_colab.md) y [E3](../docs/e3_colab.md).

Cada nueva corrida crea `reports/eN_v4/<fecha-id>`. Conserva `experiment-design.json`, `summary.json`, `scores.json`, `analysis.json`, `findings.md`, figuras y procedencia. E3 añade manifiesto/auditoría, configuración, historia/checkpoint, curvas, cobertura y modelo ONNX. Las salidas históricas del notebook se preservan al editar el método: las celdas nuevas o modificadas deben ejecutarse para producir evidencia actualizada.

Cada experimento conserva primero su evidencia completa en `reports/` y después la publica obligatoriamente en `https://ml.pettrace.app`. La URI y credenciales se cargan desde Colab Secrets; nunca se incluyen en el notebook. Un fallo de publicación conserva el reporte y se reintenta sin volver a entrenar ni evaluar. El registro de versiones y la promoción de aliases siguen siendo decisiones separadas.

E1 se relanza con `pettrace run e1` de forma headless; su notebook queda como evidencia histórica. `tools/attach_notebook.py` adjunta notebooks interactivos usando su recibo; desde VS Code deben guardarse primero en el equipo local. `make manifest` actualiza el índice `experiments.yaml` mediante lectura de MLflow, después de una publicación real. El índice puede incluir runs fallidos o incompletos y debe interpretarse junto a su estado y artefactos.

Las fotos privadas no se incorporan a la presentación de resultados; se emplean códigos de identidad. Los reportes permanecen fuera de Git. Las ejecuciones históricas v3 describen otra muestra y no deben mezclarse con esta comparación.

## Paquete para relanzar en Colab

E1 instala `pettrace-ml[tracking]` y se ejecuta con `pettrace run e1`; E2 conserva su notebook interactivo. Python 3.11–3.13 está admitido. La evidencia histórica conserva el runtime con el que fue producida. [La guía E1](../docs/e1_colab.md) describe PetTrace v5, resultados y recuperación.

El notebook E3 instala **`pettrace-ml[training,notebooks]==0.2.26` desde PyPI** con un solo comando. Carga el corpus refinado v2 desde Hugging Face, calcula una vez las representaciones E2 congeladas y entrena las semillas 45–49. No requiere Drive, requirements copiados ni construir un wheel. Véase [la guía E3](../docs/e3_colab.md).

El código está en `src/pettrace/`, organizado por datos, modelos, entrenamiento y evaluación. E1/E2 no entrenan; E3 requiere GPU y entrena su MLP en la fase 2. La migración de imports conserva las salidas históricas: no constituye otra ejecución ni publica métricas nuevas.

### E3 multisemilla

Las semillas 45–49 comparten exactamente datos, features e hiperparámetros. Cada una
selecciona su checkpoint con validación pública; después se elige la semilla ganadora
por esa misma métrica, con desempate por la semilla menor. Recién entonces se evalúan
los tests público v1 y PetTrace v4. Todas las semillas publican ambos dominios en MLflow,
pero solo la ganadora se registra y produce el handoff para E4/E5. Las semillas 42–44 y
los estudios 20/50/100 son evidencia histórica publicada, no candidatos de esta selección.
