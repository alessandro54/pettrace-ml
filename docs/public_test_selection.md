# Selección de las 60 identidades del test público

Verificación ejecutada el 20 de noviembre de 2026. El test no se seleccionó entre las 42.113 identidades del corpus ampliado ni procede de la recolección entre conocidos de PetTrace v6.

El piloto público v1, commit `dd7a00cb8b3638759ffa0216251e62d334b5bf10`, retuvo 402 identidades después de curación: 229 gatos y 173 perros, con mínimo de cuatro fotos por identidad. Para cada fuente se ordenaron las etiquetas, se barajaron con `random.Random("42:<fuente>")` y se asignó aproximadamente 15% a validación, el siguiente 15% a test y el resto a entrenamiento. El test contiene 34 gatos y 26 perros; validación tiene los mismos conteos de identidades. El piloto previo había muestreado candidatos y fotos de las fuentes, por lo que este no es un muestreo aleatorio de todas las identidades del corpus posterior.

Se reprodujo el conjunto exacto de etiquetas de test mediante el algoritmo histórico y se contrastó con el manifiesto conservado: coincidencia completa para ambas fuentes. Al ampliar entrenamiento a v4.100 y después publicar las fuentes refinadas, preparación reutilizó las filas v1 de validación/test y asignó las nuevas filas a train. La capa v2 añade localización, sin reemplazar ese test.

Esto preserva comparabilidad histórica y separación por identidad, pero limita la representatividad del pequeño test y no garantiza independencia entre sesiones. Los resultados son retrospectivos porque las evaluaciones previas influyeron en decisiones posteriores.

Evidencia verificable: [recibo](../reports/advisor-closure-20261120/public-test-selection.json). Código histórico: `git show 17a7728^:src/pettrace/data/curation.py` (`audit_and_split`) y `git show 17a7728^:src/pettrace/data/preparation.py` (reutilización de `reference_rows`). La semilla 42 corresponde al split, no a las semillas de entrenamiento 45–49.
