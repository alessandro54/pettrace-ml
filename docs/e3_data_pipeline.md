# Datos de E3

E3 consume datos preparados; no descarga las fuentes raw ni ejecuta limpieza dentro del experimento.

## Releases públicas

La entrada canónica es `pettrace.data.training(version="v2")`, unión verificada de:

- `alessandro54/labeled-cats-in-the-wild-refined@v2`
- `alessandro54/dogs-of-the-world-refined@v2`

Cada release v2 conserva las imágenes e identidades refinadas de v1 y añade el recorte YOLO
auditable usado por E2. El cargador fija commits, verifica hashes, splits y fingerprint, y
comprueba equivalencia con la preparación histórica `v4.100`. Hugging Face es el transporte
canónico; Drive no forma parte del protocolo de E3.

La creación fue un flujo one-shot ya finalizado; las releases y sus tarjetas en Hugging Face son
la fuente canónica. Los experimentos únicamente las descargan y verifican.
Cambiar imágenes, identidades, splits o cajas requiere una versión nueva; nunca se modifica v2.

## Uso en el experimento

El [notebook canónico de E3](../notebooks/e3_mlp_head_triplet.ipynb):

1. carga y verifica v2;
2. separa `train` y `val` por identidad;
3. calcula una sola vez embeddings E2 congelados;
4. reutiliza exactamente esas features para todas las semillas;
5. selecciona checkpoints y semilla únicamente con validación;
6. evalúa después Public v2 y PetTrace v5 como galerías independientes.

El test público no comparte identidades con train/val. PetTrace no interviene en ajuste,
early stopping ni selección. Los caches locales solo aceleran la corrida y no son evidencia:
los commits, fingerprints, configuración y hashes del modelo se guardan en cada reporte y run.

## Alcance histórico

Las preparaciones 20/50/100 permitieron explorar el efecto del volumen y permanecen publicadas
en MLflow. El protocolo vigente usa el corpus completo v2 y estudia variación entre semillas;
no mezcla aquellas condiciones históricas en la selección del modelo entregado a E4/E5.
