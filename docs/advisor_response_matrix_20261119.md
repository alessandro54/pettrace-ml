# Cobertura de los 38 comentarios del asesor

Fecha: 19 de noviembre de 2026. Documento evaluado: short paper ENG v2 CLIP–DINOv2, con correcciones adicionales de esta revisión. Fuente primaria: los 38 comentarios de `P202620066 - Short Paper ENG v1.0 - Annotated.docx`, releídos directamente de Word; informe previo en [plan de revisión](adrian_revision_plan.md).

**No están todos cerrados.** El nuevo enfoque cubre la comparación de encoders y mejora la transparencia, pero siguen pendientes la validación entre sesiones, documentación de datos y acceso reproducible para revisores. Retirar E5 del short paper elimina afirmaciones de ese alcance; no corrige su implementación ni completa sus experimentos.

Estados: **Cubierto** = cambio verificable en el nuevo alcance; **Parcial** = hay avance pero falta parte del requisito; **Pendiente** = falta evidencia; **Fuera de alcance** = no se presenta E5/sensibilidad histórica, sin afirmar que su problema esté resuelto.

| Comentario | Estado | Por qué y cómo se atiende | Qué falta |
|---|---|---|---|
| C01 Aporte y título | Cubierto | Título, resumen y conclusión se centran en adaptación CLIP/DINOv2, E1–E4. E5 no se propone como aporte principal. | La decisión Q1 sigue condicionada a evaluación adicional. |
| C02 Oración del reranking | Fuera de alcance | El resumen se reescribió sin esa oración ni descripción de E5. | Corregirla si E5 vuelve al manuscrito largo. |
| C03 Resultado en resumen | Cubierto | Incluye Rank-1/mAP de ambos encoders, punto de partida y E4, además de la limitación de transferencia. | — |
| C04 Cronología y exposición a test | Parcial | III-B declara estudio retrospectivo y selección por validación; distingue checkpoints históricos y nuevos. | Cronología completa por etapa y cohorte confirmatoria nueva. |
| C05 Controles arquitectónicos | Parcial | II-B/IV explican capacidades y presupuestos distintos; V-C incluye alternativa lineal. Se reduce el aporte a comparación empírica de una secuencia. | LoRA sin MLP y controles de arquitectura/entrenamiento con presupuesto igual si se quiere una afirmación causal fuerte. |
| C06 Acceso del revisor | Pendiente | Hay recibos y registros; el manuscrito reconoce restricciones privadas. | Procedimiento probado de acceso, enlaces inmutables e índice actualizado; autenticar nosotros no prueba acceso del asesor. |
| C07 Pérdida y numeración | Cubierto | Se cita batch-hard triplet de Hermans y se describe la pérdida usada; se retira la atribución a quadruplet. | Revisión editorial final al incorporar nuevas referencias. |
| C08 Novedad de fusión | Fuera de alcance | No se atribuye novedad a VLM ni a fusión; E5 no forma parte de la comparación. | Comparador de fusión si se recupera esa contribución. |
| C09 Estado del arte | Parcial | II sustituye el catálogo por trabajos de mascotas, CLIP, DINO/DINOv2 y adaptación; no compara porcentajes de protocolos incompatibles. | Comparación más explícita de galerías, supervisión y particiones de los antecedentes animales. |
| C10 Flujo de exclusiones | Pendiente | III-A resume curación y conserva totales. | Conteos reconciliados por motivo y orden, incluidas identidades excluidas. |
| C11 Procedencia PetTrace | Parcial | v6 fijado; III-A incorpora la recolección por conveniencia y la contribución voluntaria informada declarada por el autor, con acceso restringido. Auditoría visual de 502 fotos preparada. | Adjudicación independiente de identidad, alcance verificable de permisos, diferencia v5/v6 y sesiones confirmadas. |
| C12 Selección de 60 identidades | Cubierto | Se reprodujo exactamente el split del piloto v1: semilla 42, 34 gatos/26 perros entre 402 identidades retenidas, conservado al ampliar entrenamiento. Véase `public_test_selection.md`; III-A incorpora la explicación y su límite de representatividad. | — |
| C13 Acceso por fuente | Cubierto | Tabla I y III-A distinguen fuentes públicas de PetTrace restringido. | — |
| C14 Unidades de ranking | Cubierto | III-B distingue Rank-k por identidad, AP por fotografía, ponderación por consulta y galerías por especie. | — |
| C15 Variabilidad | Cubierto en alcance declarado | IV-C/V-C reportan cinco semillas E3 CLIP (80,00 ± 2,23% Rank-1), repeticiones E4 CLIP y tres semillas por etapa DINO; tabla V muestra media/DE. Se aclara que E4 comparte padre E3. | No es variabilidad de cadenas completas independientes. |
| C16 Entrenamiento efectivo | Parcial | Dimensiones MLP, P×K, margen, tasas, pasos, parada y selección descritos; se añaden rangos de exposición efectiva CLIP/DINO, parámetros y tiempos registrados con límites de comparabilidad. `training_exposure_20261120.md` consolida 15 corridas. | Completar detalles del E3 CLIP histórico y el coste total comparable de inferencia. |
| C17 Exactitud/fallos de atributos | Fuera de alcance | No se usan atributos en E1–E4. | Evaluación ciega por campo y fallos por dominio/especie para E5. |
| C18 Missingness de atributos | Fuera de alcance | El short paper no afirma manejo neutral de atributos faltantes. | Corregir scorer y repetir E5 antes de recuperar esa afirmación. |
| C19 Control sin intervención | Fuera de alcance | No se presenta la regla E5 como segura ni como mejora validada. | Selección con abstención total si se vuelve a estudiar E5. |
| C20 Diagrama y parámetros | Cubierto para E1–E4 | Se retira el esquema incorrecto; tabla II y IV-B indican MLP congelado en E4, LoRA y proyección nativa CLIP entrenable. | Esquema correcto de fusión solo si E5 vuelve. |
| C21 Unidades y legibilidad | Cubierto | Métricas expresadas como porcentajes y notas de 8 pt; revisión visual de todas las páginas. | Revalidar paginación tras nuevas tablas. |
| C22 Rank-5 CLIP E3 | Cubierto tras detectar regresión editorial | Se corrige 98,90 a 274/277 = 98,92%. La primera entrega v2 todavía contenía el error; se corrigió en esta revisión. | — |
| C23 Reproducción E4 | Cubierto | Se conserva 237/277, Rank-5 99,28 y mAP 83,45; se etiqueta como histórico y se conserva evidencia del asesor. | — |
| C24 Explicación del recorte | Cubierto | Se habla de localización/encuadre y resultados observados, sin afirmar causalidad del fondo. | Un control de máscara solo si se quiere aislar la causa. |
| C25 Sensibilidad de checkpoint anterior | Fuera de alcance | Se retira esa sección y no se atribuyen variantes antiguas al ganador final. | Reejecutar solo si se decide incluirlas. |
| C26 Rank-1 privado E3 | Cubierto | Tabla IV usa 402/502 = 80,08%; diferencia histórica E4 de 6,18 puntos. | — |
| C27 mAP privado E3 | Cubierto | Tabla IV usa 68,43% desde 0,6843190361. | — |
| C28 E5 sin compuertas | Fuera de alcance | No hay tabla E5 ni argumento de Rank-5 como mejora independiente. | Ambas condiciones completas si E5 vuelve. |
| C29 Intentos/cambios reales | Fuera de alcance | Se elimina la afirmación de que ungated cambia todo líder. | Completar conteos de ungated en el estudio E5. |
| C30 Elegibilidad/intervención | Fuera de alcance | Se retiran 13,72/12,75% como intervención. Auditoría E5 previa distingue cambios reales. | Mantener exportación y desglose si se publica E5. |
| C31 Dependencia entre fotos | Cubierto para nuevos contrastes DINO | III-B y V usan bootstrap pareado por identidad estratificado por especie, con galería fija; límites explícitos. | No equivale a incertidumbre sobre galerías/sesiones nuevas. |
| C32 Luna y alias | Fuera de alcance | Luna no sustenta la conclusión del nuevo short paper. La auditoría previa verifica rescates 55/56 y distingue rival máximo de rival que supera a Luna. | Reconciliar alias cat-01 si se recupera el caso. |
| C33 Rescate/daño con fotos reales | Fuera de alcance | Se retiró la figura representativa; no se usa como evidencia experimental. | Ejemplos medidos y permiso de imágenes para E5. |
| C34 Sesiones y distractores | Parcial | Sensibilidad interna y nueva expansión Oxford de 0/100/250/500 fotos por especie para CLIP/DINOv2 E1–E4; `external_distractors_results_20261120.md`. Protocolo fijado, checkpoints inmóviles, comprobación de reproducción y resultados por consulta. | La expansión externa está publicada en MLflow (`daba7785305f4da08200005788e9bc5e`); aún falta evaluación entre sesiones confirmadas. Oxford no certifica identidades externas ni rechazo open-set. |
| C35 Una de cada ocho consultas | Fuera de alcance | Conclusión reescrita sin esa afirmación. | — para E1–E4; corregir si E5 vuelve. |
| C36 Competidor fuerte | Parcial | DINOv2 congelado y adaptado se evalúa con mismo test y definición de retrieval; tablas III–V y recibos MLflow. | Costes/parámetros y presupuesto comparables; comparador animal especializado cuando sea viable. |
| C37 Disponibilidad y versiones | Parcial | Manuscrito separa datos públicos/privados, fija commits y llama a la evaluación closed-set. | Reconciliar README, model cards, índices y acceso externo; no todos se actualizaron en esta revisión. |
| C38 Implementación YOLO | Cubierto en descripción | IV-A cita Ultralytics YOLOv8n y explicita confianza 0,25, margen 10%, fallback; artefacto fijado en evidencia. | Mantener enlaces inmutables del artefacto en suplemento de reproducibilidad. |

## Evaluaciones autorizadas en esta revisión

CLIP E1/E2 v6: ejecutor existente E2, que calcula también el control E1 sobre exactamente las mismas consultas. Se detuvo una primera preparación que descargaba entrenamiento innecesario; se relanzó usando las imágenes de test verificadas de C1. Protocolo, código de ejecución y evidencia local en `reports/clip-e1-e2-v6-20261119-control/` y `reports/clip-e1-e2-v6-20261119-verified/`. Ambos runs están FINISHED y se verificó que los artefactos incluyen también el control E1. Público: `b26c5d8494054beab283f0bf985cb756`; privado: `bea86db8772042b0a69714a14534dbc0`. No hay entrenamiento ni promoción de aliases.

## Revisión visual para sesiones

Se inspeccionaron las 50 hojas de contacto (502 fotos). Se guardó una propuesta de agrupación por contexto visual en `reports/session-visual-audit-v6/visual_groups_draft.csv`. Siete identidades quedaron conservadoramente en un solo grupo visual: no deben partirse artificialmente para aparentar sesiones distintas. Las imágenes sin pareja clara tienen estado desconocido, no una sesión nueva automáticamente. No se consultaron aciertos/errores de modelos para agrupar.

El usuario indica que la mayoría de imágenes proceden de ocasiones distintas. Esto favorece la viabilidad, pero se necesita adjudicación por fotografía. La inferencia visual puede apoyar un análisis de sensibilidad por contexto; no acredita separación temporal. Cero sesiones temporales están confirmadas por esta auditoría. No se modificó PetTrace v6 ni el test público.

Resultados privados completados: E1 Rank-1 65,34%, Rank-5 92,43%, mAP 38,85%; E2 Rank-1 69,92%, Rank-5 91,63%, mAP 40,27%. Se incorporaron a la tabla IV del manuscrito; los controles públicos reprodujeron sus métricas anteriores. La entrega del 19 de noviembre conservó cuatro páginas y tres columnas de autores. La actualización del 20 de noviembre, con distractores externos, tiene cinco páginas y preserva exactamente la tabla de tres autores.

## Aclaración del autor sobre procedencia y permisos · 20 de noviembre de 2026

El autor aclara que sí dispone de autorizaciones por mensaje para fotografías aportadas por personas conocidas. El conjunto también incluye fotografías de animales callejeros. Según su declaración, no se incluyen datos identificativos de los propietarios ni se conserva un registro nominal que los vincule con las imágenes; el dataset permanece privado. Esta aclaración sustituye la interpretación anterior de que no existían autorizaciones.

No se afirma que se hayan obtenido formularios firmados, aprobación ética o permisos de publicación de fotografías. Los mensajes no han sido revisados en esta auditoría: su alcance y cobertura por fotografía permanecen sin verificar. Tampoco se ha comprobado exhaustivamente la ausencia de elementos identificativos dentro de las imágenes. Para las fotos de animales callejeros debe documentarse quién las tomó o aportó y cómo se adjudicó la identidad del animal; su condición de callejero no determina la procedencia de la fotografía.

C06 (acceso del asesor) queda a cargo del autor. C11 permanece parcial por la documentación de procedencia, adjudicación de identidad, sesiones y alcance de las autorizaciones, no por una supuesta ausencia total de permiso. No se exige crear un registro nominal de propietarios. Privacidad, autorización de uso y acceso para revisión se describen por separado. Las fotografías privadas se mantienen fuera del material compartido y los otros trabajos de evaluación y documentación pueden continuar.

Actualización del 20 de noviembre: C12 tiene reproducción exacta documentada. C34 incorpora sensibilidad interna y expansión con fotografías externas de Oxford; sus resultados y límites se documentan por separado. Esto no cierra la evaluación entre sesiones.

La versión del 20 de noviembre incorpora Oxford-IIIT Pet como referencia [13], tabla VI, selección del piloto público, procedencia declarada y exposición/tiempos registrados. El índice para el asesor se redujo a dos páginas con las 13 referencias actuales y nueve PDFs verificados en su carpeta OneDrive. Esto no verifica que una cuenta externa pueda abrirlos.
