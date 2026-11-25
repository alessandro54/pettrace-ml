# Evidencia DINOv2 para revisión del asesor

Publicación documental: 25/11/2026. Los registros conservan sus fechas originales; esta incorporación no es una nueva ejecución.

- `source/`: contenido textual íntegro del paquete `pettrace-sequential.zip` conservado para el estudio 20261119-v1. Incluye el notebook y el paquete Python con sus dependencias declaradas en `pyproject.toml`.
- `evaluated-model-source/`: código y configuración conservados junto al modelo evaluado. Se mantienen separados del paquete de lanzamiento para que puedan contrastarse; no se afirma que el árbol actual del repositorio sea idéntico a la versión ejecutada.
- `results/sequential/`: registros de E1 (imagen completa), E2 (recorte), E3 (MLP) y E4 (MLP seleccionado + LoRA). La selección por validación fue MLP semilla 46 y LoRA semilla 47. Resultado público E4: Rank-1 96,39 %, mAP 92,17 %; PetTrace v6: Rank-1 94,42 %, mAP 87,29 %.
- `lora-only-evaluated.ipynb` y `results/lora-only/`: control independiente sin MLP. No debe confundirse con E4 secuencial. Es el origen del modelo de servicio seleccionado, semilla 45: PetTrace v6 Rank-1 95,82 %, mAP 90,40 %.
- Los comprobantes `publication.json` identifican los runs de MLflow. El asesor necesita acceso autorizado a https://ml.pettrace.app para consultar sus artefactos remotos.
- `SHA256SUMS.json`: integridad de los archivos publicados y hash del archivo original de empaquetado.

No se distribuyen imágenes privadas, pesos ni credenciales. Las métricas privadas pueden contrastarse con los registros, pero repetir la inferencia requiere acceso autorizado a las fotografías. Los resultados no acreditan generalización entre sesiones de captura ni un piloto independiente con usuarios. La repetición de E4 está condicionada al mismo MLP seleccionado; no equivale a repetir independientemente toda la secuencia.

Para inspeccionar el protocolo, comenzar por `source/notebooks/dino_sequential_pipeline.ipynb`, `source/docs/two_encoder_framework.md`, los módulos en `source/src/pettrace/training/` y los comprobantes de evaluación. El notebook LoRA-only conserva la copia recuperada del run evaluado, no una sustitución por el notebook de trabajo actual.
