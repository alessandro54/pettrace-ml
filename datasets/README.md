# Datasets de PetTrace

Esta carpeta conserva la documentación y la evidencia pequeña que debe permanecer en Git. Los
experimentos de `notebooks/` consumen releases inmutables; no vuelven a descargar los ZIP
originales ni repiten la limpieza.

## Releases canónicas

| Fuente refinada | Entrada inmutable | Uso |
|---|---|---|
| Gatos | [`labeled-cats-in-the-wild-refined@v1`](https://huggingface.co/datasets/alessandro54/labeled-cats-in-the-wild-refined/tree/v1), commit `089518b5cffcf71a7f55dbc7d47d0f2269f98a2b` | Identidades públicas de LCW |
| Perros | [`dogs-of-the-world-refined@v1`](https://huggingface.co/datasets/alessandro54/dogs-of-the-world-refined/tree/v1), commit `26a361971d5a77976fadd4aeae5e8294d40333c7` | Identidades públicas de Dogs of the World |

`v1` conserva imágenes completas después de verificar archivos, deduplicar, separar identidades
y excluir escenas ilegibles, sin mascota, con varias mascotas o con una especie inesperada. Las
decisiones de preparación y sus límites están documentados en las tarjetas de ambos datasets.

El paquete descarga y verifica esas releases directamente:

```python
from pettrace import data

dogs = data.dogs_of_the_world(version="v1")
cats = data.labeled_cats_in_the_wild(version="v1")
training = data.training(version="v2")
```

`pettrace-training` es el corpus público para entrenamiento y validación. `pettrace-reid@v4` es
el benchmark privado e inmutable de evaluación; sus galerías nunca se mezclan.

## Capa YOLO v2

Las releases `v2` agregan la caja reproducible del detector E2 en
`versions/v2/screening.csv.gz`. No cambian identidades, splits ni bytes de imagen y no duplican
los TAR. El consumidor aplica el mismo margen de recorte declarado por el modelo.

Su generación fue una migración one-shot ya completada. Las etiquetas, commits y tarjetas de
los datasets en Hugging Face son la evidencia canónica; el repositorio no conserva un launcher
de publicación obsoleto.

## Evidencia conservada

- [`source_capacity_v1.json`](source_capacity_v1.json) fija conteos, alcance y hashes de los
  archivos fuente usados como denominadores de cobertura.
- `reviews/` conserva decisiones humanas vinculadas al fingerprint del reporte original. Son
  evidencia histórica; no se aplican silenciosamente a nuevas versiones.

Los antiguos notebooks basados en ZIP y restauración completa desde Drive se retiraron una vez
publicadas y verificadas las releases. Su código, salidas ejecutadas y evolución permanecen en el
historial de Git. `pettrace.data` conserva únicamente los loaders y verificadores necesarios para
consumir las releases; la preparación y publicación one-shot no forman parte de la API OSS.
