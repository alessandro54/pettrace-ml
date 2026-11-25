# Auditoría de E5 y del caso Luna — 18 de noviembre de 2026

Se recontaron los registros publicados de E5 y se contrastó cada líder visual con
la evaluación histórica de E4, para las mismas consultas, identidades y especies.
No se ejecutó inferencia nueva, no se ajustaron umbrales y no se seleccionaron modelos.
Esta evidencia confirma las reservas de Adrián sobre la magnitud del aporte de E5.

## Elegibilidad y cambios reales

| Conjunto | Consultas | Primer filtro abierto | Líder distinto | Rescates | Daños | Incorrecto → otro incorrecto | Aciertos E4 → E5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Público | 277 | 38 (13,72 %) | 8 (2,89 %) | 4 | 3 | 1 | 237 → 238 |
| PetTrace v6 | 502 | 64 (12,75 %) | 13 (2,59 %) | 8 | 2 | 3 | 433 → 439 |

Los porcentajes 13,72 % y 12,75 % describen únicamente la condición sobre el margen
visual. No son el porcentaje de consultas cuyo líder cambió. Tampoco basta sumar
rescates y daños: existen cambios entre dos respuestas incorrectas.

En validación, el registro de calibración seleccionado documenta 16/273 consultas
elegibles, un cambio, cero rescates y un daño. Para validación se verificó ese registro
agregado; no se recuperaron decisiones individuales de sus 273 consultas.

La tabla publicada de la variante *ungated* registra 4 rescates/3 daños en público y
10 rescates/3 daños en PetTrace. No incluye el total de líderes distintos. Aplicar la
fusión a todas las consultas no demuestra que todas cambien de líder. Ese conteo
sigue pendiente; no se ha deducido de los rescates y daños.

Evidencia: [E5 público](https://ml.pettrace.app/#/experiments/3/runs/c8a0d12a59f44bea9fc719d96556c418)
y [E5 PetTrace](https://ml.pettrace.app/#/experiments/3/runs/57ffea8d48a64561be3651fad9b0d914),
artefactos `report/evaluation-input.json`, `report/calibration.json` y `report/comparison.json`.

## Luna sí tiene evidencia publicada

El [estudio de caso ya publicado](https://ml.pettrace.app/#/experiments/3/runs/f7d92f64f66a469e82360484fe817cf0)
está finalizado. Sus métricas y `case-study/case-results.json` coinciden con el
recuento independiente de los registros oficiales de E5.

| Identidad | Consultas | E4 correcto | E5 correcto | Rescates | Daños |
|---|---:|---:|---:|---:|---:|
| Luna (`cat-b8d7189e`) | 10 | 2 | 4 | 2 | 0 |
| Sunset (`cat-54af8927`) | 10 | 4 | 4 | 0 | 0 |
| Cat-8 (`cat-09843e72`) | 10 | 8 | 8 | 0 | 0 |

Los nombres siguen el mapeo del notebook del caso. El alias «cat-01» del manuscrito
todavía requiere reconciliación explícita; no se ha inferido su identidad por orden alfabético.

| Consulta de Luna (índice desde cero) | Líder E4 | Líder E5 | Margen visual | Ventaja textual |
|---|---|---|---:|---:|
| 52 | Sunset | Sunset | 0,018339 | 0 |
| 53 | Luna | Luna | 0,046831 | 0 |
| 54 | Luna | Luna | 0,042935 | 0 |
| 55 | Sunset | Luna | 0,006074 | 0,461538 |
| 56 | Sunset | Luna | 0,004491 | 0,461538 |
| 57 | Sunset | Sunset | 0,004783 | 0 |
| 58 | Sunset | Sunset | 0,017157 | 0 |
| 59 | Sunset | Sunset | 0,012027 | 0 |
| 60 | Sunset | Sunset | 0,007942 | 0 |
| 61 | Cat-8 | Cat-8 | 0,009112 | 0 |

Los umbrales congelados son τv=0,01, λ=0,02 y τt=0,05. Solo las consultas 55 y 56
se rescatan. El margen visual compara los dos primeros candidatos, no necesariamente
Luna y su rival cuando Luna está por debajo del segundo puesto.

**Corrección del 9/10:** Sunset es la identidad incorrecta con mayor puntuación visual
en nueve consultas de Luna, pero supera a Luna en siete. En dos de esas nueve, Luna
sigue primera. El décimo rival es Cat-8. Se comprobó esta distinción en las filas de E4.

El resultado es explicativo y posterior al benchmark. Seis fotos de Luna siguen
fallando con E5. La anotación del ojo ausente es humana: estos registros no demuestran
que Qwen la haya detectado, ni identifican por sí solos qué atributo causó cada rescate.
Las fotos de Sunset y Cat-8 de la figura publicada son representativas, no necesariamente
las fotos exactas que determinaron la puntuación máxima de la galería.

## Evidencia reproducible y límites

El bootstrap pareado de 10.000 réplicas remuestreando animales dentro de cada especie,
con semilla 20261118 y galería fija, produce estos intervalos descriptivos del 95 %:

| Dominio | Δ Rank-1 (puntos) | Intervalo Δ Rank-1 (puntos) | Δ mAP | Intervalo Δ mAP |
|---|---:|---:|---:|---:|
| Público | +0,3610 | [−1,1029; +1,8868] | +0,000151 | [−0,003629; +0,003908] |
| PetTrace | +1,1952 | [0,0000; +2,5898] | +0,001275 | [−0,000389; +0,003117] |

Todos incluyen cero; el extremo inferior de Rank-1 privado coincide con cero. Este
análisis exploratorio no demuestra superioridad poblacional de E5. Condiciona en
galerías y modelos fijados y no abarca variación de entrenamiento, galerías nuevas
ni sesiones distintas. Se reutiliza el mismo analizador que en C1, documentado junto
a los métodos de [bootstrap pareado](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
y [conglomerados/estratos](https://www.stata.com/manuals/rbootstrap.pdf).

En `reports/e5-evidence-audit/` se conservaron los originales descargados, sus SHA-256,
`audit.json`, `luna-queries.csv`, `paired-cluster-analysis.json` y `audit_records.py`. El script comprueba que los
indicadores de cambio, rescate y daño coincidan con las identidades antes/después,
que las consultas coincidan con E4 y que los totales del caso coincidan con MLflow.
Puede repetirse con `.venv/bin/python reports/e5-evidence-audit/audit_records.py`.

No se volvió a ejecutar el modelo ni Qwen; esta es una auditoría de evidencia existente.
Faltan el conteo exacto de cambios *ungated*, la atribución por campos del VLM y una
evaluación entre sesiones distintas. Estos resultados no convierten el caso de Luna
en prueba de una mejora general o estadísticamente concluyente de E5.
