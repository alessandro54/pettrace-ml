# Análisis de las observaciones de Adrián y plan de mejora

Análisis inicial: 17 de noviembre de 2026. Actualización de ejecución: 18 de noviembre de 2026.

En el manuscrito revisado, el resultado más sólido era la adaptación visual E3–E4. La revisión respalda su resultado observado y reproduce E4 público; pide reforzar la evaluación antes de sostener una contribución metodológica para Q1. La comparación posterior C1 favorece priorizar DINOv2 como base de la siguiente etapa. E5 está implementado y evaluado, pero su beneficio es pequeño, no significativo en las pruebas reportadas y contrario al control E4 en validación. Debe presentarse como exploratorio mientras se resuelven las observaciones.

Este documento analiza los 38 comentarios del manuscrito y el informe científico. Distingue el diagnóstico inicial, las acciones propuestas y la evidencia nueva. Solo los puntos marcados como ejecutados tienen resultados publicados.

## Estado de la actualización del 18 de noviembre

| Punto | Estado verificable |
|---|---|
| Fuentes solicitadas por Adrián | 37 referencias verificadas; 32 PDF académicos, 5 fichas web y sus README; índice y copia del manuscrito en [ZIP](../../pettrace/share/P202620066_Fuentes_Adrian.zip). Correcciones en la auditoría del paquete. |
| Competidor independiente (C36) | C1 DINOv2-L ejecutado y publicado: 92,06 % / 0,8383 público, 94,82 % / 0,8650 privado (Rank-1 / mAP). [Informe y límites](c1_dinov2_results_20261118.md). Falta integrar al manuscrito y completar la discusión de costes. |
| Entrenamiento efectivo (C05/C16) | Confirmado en código y checkpoint: E4 entrena LoRA más proyección visual. Se corrigió la descripción del notebook histórico conservando todo su código y salidas. Se corrigió la inicialización de semilla para nuevas ejecuciones. |
| Variabilidad E4 (C15) | [Seis corridas y doce evaluaciones publicadas](e4_sampling_results_20261118.md). Seleccionado por validación: P16×K2, semilla 46. Obtiene 238/277 público y 425/502 privado: +1 y −8 frente al E4 histórico; no es una mejora general. Versión 24 candidata, sin promover aliases. |
| Ablación de componentes (C05) | Notebook preparado y probado; no ha entrenado. Queda en segundo plano ante la prioridad DINO; mantiene pendiente el acceso a secretos del nuevo cuaderno de Colab. |
| Adaptación DINO | [Piloto D1 ejecutado y publicado](d1_dinov2_pilot_results_20261118.md): 1.040 train, 273 val, tres semillas; mAP macro medio 0,8387→0,8692 y Rank-1 macro 89,47→92,06 %. Solo validación. [Pipeline DINO implementado y comprobado técnicamente](dino_pipeline.md); estudio completo pendiente. |
| Incertidumbre (C31) | C1 y la auditoría E5 incluyen diferencias pareadas y bootstrap descriptivo por animal/especie con galería fija. Los intervalos E5 incluyen cero. E4 también incluye el análisis pareado del seleccionado y dispersión entre semillas; estos análisis no confirman generalización entre sesiones. |
| Luna y cambios reales E5 (C29/C30/C32/C33) | [Auditoría de registros verificada](e5_evidence_audit_20261118.md): E5 cambia 8/277 y 13/502 líderes; Luna 2/10→4/10, consultas 55/56 verificadas. Sunset es el mejor rival incorrecto 9/10 veces, pero supera a Luna 7/10. Estudio de caso existente en MLflow contrastado. Siguen pendientes el alias cat-01, cambios ungated y atribución por campos. C1 posterior obtiene 8/10 de Luna. |
| Evaluación entre sesiones (C11/C34) | Pendiente identificar sesiones reales: el CSV v6 no contiene sesión y 0/502 fotos tienen fecha de captura EXIF. Los 502 hashes coinciden con el manifiesto; no se inventaron etiquetas ni particiones. |

**Cambio de prioridad:** el encoder independiente congelado supera al E4 histórico y al nuevo seleccionado en Rank-1 y mAP globales en ambos dominios. Se prioriza [adaptación sobre DINOv2](dinov2_adaptation_protocol.md) antes de añadir E6: la ganancia pública de mAP es pequeña y gatos públicos presentan una regresión. La mejora E3→E4 sigue siendo un resultado propio dentro de CLIP, pero ya no basta para sostener superioridad frente al comparador independiente. Los tests observados motivaron este giro; sigue haciendo falta confirmación independiente.

## Fuentes y alcance

- [Informe científico de Adrián](</Users/alessandro/Develop/personal/pettrace/P202620066 - Scientific Review.docx>).
- [Manuscrito con 38 comentarios](</Users/alessandro/Develop/personal/pettrace/P202620066 - Short Paper ENG v1.0 - Annotated.docx>).
- Código auditado al inicio: revisión `8dfc120`, coincidente con la auditada en el informe; las modificaciones locales posteriores se describen en la actualización.
- [Ranking y calibración de E5](../src/pettrace/evaluation/reranking.py), [extracción de atributos](../src/pettrace/evaluation/attributes.py) y [orquestación y exportación de E5](../src/pettrace/experiments/e5.py).
- [Protocolo E4](e4_protocol.md), [plan E5](e5_optimization_plan.md) y [organización de notebooks](../notebooks/README.md).

Los identificadores C01–C38 corresponden al orden de los comentarios en el documento anotado, no a identificadores internos de Word. Los resultados reproducidos por Adrián se atribuyen a su revisión. La fase inicial no reentrenó ni repitió inferencia; C1 y el estudio E4 posteriores se distinguen explícitamente de esa evidencia histórica.

## Evidencia histórica revisada por Adrián

| Evidencia | Resultado | Alcance |
|---|---|---|
| E2 público | 188/277 aciertos Rank-1; 67,87 % | CLIP congelado con recorte. |
| E3 público seleccionado | 226/277; 81,59 % | Añade 38 aciertos respecto de E2. |
| E4 público | 237/277; 85,56 %; mAP 0,8344670270 | Añade 11 respecto de E3. Reproducido por Adrián con pesos e imágenes publicados. |
| E5 con compuertas público | 238/277; 85,92 % | Cuatro rescates y tres daños: mejora neta de uno. |
| E4 privado v6 | 433/502; 86,25 % | Contrastado con registros publicados; no reproducción privada independiente. |
| E5 con compuertas privado v6 | 439/502; 87,45 % | Ocho rescates y dos daños: mejora neta de seis. |
| E3 con cinco semillas | Rank-1 público medio 80,00 %, DE muestral 2,23 puntos | Variabilidad de E3; no demuestra estabilidad de LoRA. |
| Luna | E4 2/10 y E5 4/10 según el manuscrito | Caso incluido en la versión revisada; falta evidencia ejecutada por consulta. |

Los resultados globales no aíslan por sí solos el efecto de cada decisión arquitectónica. La mejora E3→E4 también es heterogénea: en público suma diez aciertos en gatos y uno en perros, según el informe.

## Prioridades

- **P0:** corregir afirmaciones, cifras y trazabilidad antes de circular otra versión.
- **P1:** reforzar evidencia para decidir el envío como trabajo de métodos a Q1.
- **P2:** completar presentación, documentación y material editorial antes del envío.

Una prioridad baja no significa que la observación pueda omitirse. Algunos puntos incluyen una corrección inmediata y un experimento posterior.

## Matriz de los 38 comentarios

### Aporte, posicionamiento y diseño experimental

| ID | Prioridad | Observación y situación actual | Acción y criterio de cierre |
|---|---|---|---|
| C01 | P0 | El título da protagonismo comparable a morfología y adaptación visual. E5 aporta +1 público y +6 privado. | Centrar título y argumento en E3–E4; identificar E5 como exploratorio. Cierre: título, resumen y conclusión reflejan la fuerza de la evidencia. |
| C02 | P0 | Una oración del resumen tiene un verbo sin objeto y confunde extracción con activación del reranking. | Explicar que los atributos se extraen offline y la regla consulta esa evidencia ante ambigüedad visual. Cierre: oración gramatical y consistente con el algoritmo. |
| C03 | P0 | El resumen no presenta suficientemente el resultado principal. | Incluir 67,87→81,59→85,56 % público y describir el pequeño incremento E5 con denominadores. Cierre: cifras generadas desde evidencia completa, sin sugerir significación no demostrada. |
| C04 | P0/P1 | Hubo exposición previa a los benchmarks; E4 v6 quedó registrado antes de introducir E5. | Documentar cronología y selección por etapa; llamar a las mediciones posteriores evaluaciones sobre benchmarks ya observados. Reservar una cohorte nueva para confirmación. Cierre: fechas y decisiones verificables, sin presentar retrospectivamente un protocolo como prospectivo. |
| C05 | P1 | Las etapas anidadas no separan arquitectura del head, supervisión y entrenamiento adicional; tampoco prueban que LoRA necesite ese head. | Definir controles con presupuesto comparable, incluyendo los necesarios para la afirmación arquitectónica que se mantenga. Cierre: comparación controlada o reducción explícita de la afirmación. |
| C08 | P0/P1 | La referencia [11] ya estudia descripciones VLM y fusión con compuertas. | Precisar qué aporta el vocabulario de siete campos y la promoción dentro del Top-5 fijo. Contrastarlo experimentalmente si se mantiene como contribución. Cierre: diferencia verificable frente al antecedente, sin atribuir novedad a toda la idea de fusión. |
| C09 | P2 | El estado del arte enumera muchas especies y sensores, con poca comparación directa. | Reducir catálogo y comparar [9], [11] y [32] por población, galería, supervisión, fusión y partición. Cierre: relación clara con nuestro problema; no comparar porcentajes entre protocolos incompatibles. |
| C36 | P1 | Falta un competidor fuerte independiente; mencionarlo como trabajo futuro no basta para la propuesta Q1. | Congelar un encoder o método animal Re-ID competitivo y evaluarlo con iguales splits, política de detector, agregación y presupuesto de ajuste. Reportar parámetros entrenables y coste de extracción VLM. Cierre: resultados y protocolo publicados, incluyendo diferencias inevitables de recursos. |

### Datos, muestreo, protocolo y entrenamiento

| ID | Prioridad | Observación y situación actual | Acción y criterio de cierre |
|---|---|---|---|
| C10 | P1 | Los totales de curación están respaldados, pero no muestran qué se excluyó ni en qué orden. | Crear flujo con conteos por motivo y exclusiones de identidades completas por tener menos de cuatro fotos. Cierre: inventarios iniciales, exclusiones y corpus final reconciliados sin doble conteo. |
| C11 | P1 | Falta explicar reclutamiento y etiquetado de PetTrace v6, sesiones, permisos y diferencias con v5. | Documentar procedencia, adjudicación de identidad, distribución de fotos por animal, sesiones y base de permisos; declarar exposición previa a v6. Cierre: ficha de datos revisable sin exponer información privada. |
| C12 | P1 | Los splits públicos son consistentes y disjuntos por identidad; falta justificar cómo se eligieron las 60 identidades de test entre 42.113 retenidas. | Recuperar la regla y semilla de selección si existen y discutir representatividad. Cierre: selección reproducible o limitación explícita si falta trazabilidad. |
| C13 | P0 | La tabla de datos no distingue suficientemente acceso público y privado. | Añadir estado de acceso por fuente y qué se distribuye. Cierre: público descargable y privado restringido claramente diferenciados. |
| C14 | P0 | Rank-k y AP usan unidades de ranking diferentes. | Explicar que Rank-k ordena identidades con máximo de similitud por animal y AP ordena fotografías; el agregado global pondera cada consulta por igual. Añadir tamaños de galerías por especie. Cierre: métricas reproducibles sin ambigüedad de unidad. |
| C15 | P1 | E3 tiene cinco semillas; E4 solamente semilla 46. | Reportar la distribución E3 y repetir E4 con protocolo de semillas fijado antes de correr. Definir si se varía únicamente LoRA o toda la cadena E3→E4 y mantener controles emparejados. Cierre: media, dispersión y cambios por corrida; no confundir variabilidad de entrenamiento con incertidumbre entre animales. |
| C16 | P0 | Faltan detalles efectivos del entrenamiento. | Documentar MLP 512→256→512, lotes de ocho identidades por cuatro fotos, alternancia de especies, 100 pasos por época, selección y presupuesto. Cierre: distinguir tamaño del corpus disponible de exposición real registrada. |
| C25 | P0 | El estudio de sensibilidad pertenece a un E4 anterior y usa una métrica concreta; no hay recibo de repetición sobre el ganador final v3. | Identificar checkpoint, métrica y prioridad de selección. Cierre: limitar afirmaciones a ejecuciones existentes; si se compara el modelo final, producir una evaluación nueva y separada. |
| C34 | P1 | Todas las consultas tienen compañero conocido; no hay separación garantizada por sesión. Galerías: público 34 gatos/26 perros; privado 17/33. | Diseñar evaluación entre sesiones o tiempos y, cuando corresponda, propietarios; estudiar distractores adicionales. Cierre: protocolo congelado y resultados por especie con galerías definidas. No afirmar búsqueda abierta o recuperación en campo a partir del protocolo actual. |

### Morfología, selección y evidencia por consulta

| ID | Prioridad | Observación y situación actual | Acción y criterio de cierre |
|---|---|---|---|
| C17 | P1 | Hay 974 extracciones exitosas, nueve reparadas y 69 fallback de 1.052; JSON válido no implica atributo correcto. Los collares pueden cambiar. | Reportar fallos por dominio y especie; evaluar una muestra ciega con anotación de campos y discutir atributos transitorios. Cierre: precisión de campos y tratamiento de fallos documentados. |
| C18 | P0/P1 | Los escalares desconocidos se omiten, pero listas fallback usan “none” y pueden penalizar marcas reales. | Preservar diferencia entre ausencia y desconocimiento; revisar propagación del estado y repetir E5 con manejo neutral. Cierre: pruebas del scorer y comparación emparejada que mida el efecto. No asumir mejora antes de medir. |
| C19 | P0/P1 | Las 96 configuraciones dañan al menos un acierto. Seleccionada: cero rescates, un daño y 247/273 frente a 248/273 de E4. | Mostrar el control sin intervención e incluirlo en una futura selección con criterio previamente fijado. Cierre inmediato: reconocer la pérdida en validación. Cierre experimental: permitir abstención total y publicar la selección, aunque gane E4. |
| C28 | P0 | Falta la fila pública E5 sin compuertas; el Rank-5 igual es consecuencia del Top-5 fijo. | Incluir ambas variantes en ambos dominios: público sin compuertas Rank-1 85,92 %, Rank-2 94,22 %, mAP 0,8363. Cierre: no presentar conservación de Rank-5 como evidencia independiente de calidad. |
| C29 | P0 | “Ungated changes every query” confunde evaluar fusión con cambiar al líder. | Contar intentos, cambios reales, rescates, daños y cambios entre dos identidades incorrectas. Cierre: conteos obtenidos de registros por consulta; retirar la afirmación actual. |
| C30 | P0 | 13,72 % y 12,75 % son elegibilidad por margen visual, no cambios de líder. La exportación omite `changed_queries`. | Incorporar cambios reales a los resultados exportados y recuperar sus valores desde registros. Cierre: diferenciar claramente 38/277 y 64/502 elegibles del número de cambios. No deducir cambios totales sumando solamente rescates y daños. |
| C32 | P0 | Luna sí aparece en la versión revisada. El notebook tiene aserciones sin salidas y nombres no reconciliados. | Producir auditoría anonimizada de diez consultas; mapear Luna, Sunset/Cat-8 y cat-01; verificar el 9/10. Cierre: separar “rival con mayor puntuación” de “rival que supera a Luna” y respaldar 2/10→4/10 con salidas. |
| C33 | P0/P2 | La figura usa fotos representativas y no muestra decisiones medidas; seis consultas de Luna siguen fallando. | Presentar un rescate y un daño reales con margen visual, ventaja textual, campos decisivos e imágenes autorizadas. Cierre: separar esquema explicativo de evidencia y mostrar también limitaciones. |
| C35 | P0 | La conclusión repite “una de cada ocho consultas cambia”. | Sustituir por elegibilidad, rescates, daños y mejora neta; indicar E4 como referencia. Cierre: ninguna sección equipara primera compuerta con intervención efectiva. |

### Cifras, interpretación y estadística

| ID | Prioridad | Observación y situación actual | Acción y criterio de cierre |
|---|---|---|---|
| C22 | P0 | E3 público Rank-5 figura como 98,90 %, incompatible con los conteos a esa precisión. | Corregir a 274/277 = 98,92 %. Cierre: redondeo final desde conteos o métricas completas. |
| C23 | Evidencia respaldada | Adrián reprodujo E4: Rank-1 237/277, Rank-2 266/277, Rank-5 275/277, mAP 0,8344670270. | Conservar resultados y vincular revisiones de datos, pesos y evaluador. Cierre: trazabilidad estable; no requiere repetir solo para atender este comentario. |
| C24 | P0 | La caída tras recortar se atribuye al contexto, aunque también cambian escala, encuadre y animal visible. | Presentarlo como hipótesis. Si se mantiene una afirmación causal, añadir control de máscara o recorte comparable. Cierre: redacción limitada a lo demostrado o evidencia adicional. |
| C26 | P0 | E3 privado Rank-1 se aproxima como 80,10 %; el control emparejado corresponde a 402/502. | Corregir a 80,08 % y mejora E4 a 6,18 puntos. Cierre: tabla y narrativa coherentes. |
| C27 | P0 | E3 privado mAP aparece como 0,6840. | Corregir a 0,6843190361, mostrado como 0,6843. Cierre: generar tabla y diferencias desde la misma fuente de precisión completa. |
| C31 | P1 | McNemar e intervalos por fotografía están reproducidos, pero fotos comparten animal y galería. | Mantener resultados existentes y añadir análisis por identidad y especie, con intervalo emparejado remuestreando animales dentro de especie para galería fija. Cierre: estimando y unidad explícitos; no extrapolar ese intervalo a galerías nuevas. |

### Reproducibilidad, referencias y presentación

| ID | Prioridad | Observación y situación actual | Acción y criterio de cierre |
|---|---|---|---|
| C06 | P0 | MLflow anónimo devuelve 401; hay notebooks y agregados públicos, pero el índice está atrasado. | Publicar enlaces inmutables y procedimiento de acceso para revisión privada; actualizar índice. Cierre: comprobar la ruta de acceso pública y la ruta autorizada privada. No abrir imágenes privadas automáticamente. |
| C07 | P0 | Una cita de pérdida cuádruple puede confundirse con la pérdida triplet real; [8] aparece antes de [7]. | Explicar qué motiva la referencia y citar el método implementado; revisar numeración. Cierre: cada cita respalda la afirmación concreta y orden consistente. |
| C20 | P0/P2 | La figura llama entrenable al MLP de E4, aunque está congelado; simplifica el orden de la decisión. | Marcar parámetros congelados/entrenables por etapa; mostrar propuesta por fusión antes de evaluar ventaja textual; aclarar coseno y agregación por identidad. Cierre: figura fiel al código y legible a tamaño final. |
| C21 | P2 | Unidades, ceros iniciales y tamaño de notas son inconsistentes; algunas notas rondan seis puntos. | Usar encabezados porcentuales, mAP con cero inicial y notas legibles. Cierre: revisión visual del PDF final sin reducir tipografía para encajar. |
| C37 | P0 | La disponibilidad mezcla código, modelos, datos y registros restringidos; descripciones públicas conservan estados antiguos y alusiones a open-set. | Separar activos y accesos, fijar revisiones y actualizar estados. Cierre: README, model card y manuscrito describen la evaluación closed-set ejecutada. |
| C38 | P0 | La referencia histórica de YOLO no identifica la implementación usada. | Documentar YOLOv8n de Ultralytics y versión/artefacto: confianza 0,25, margen 10 % y fallback a imagen completa. Cierre: implementación reproducible y referencia adecuada junto al antecedente histórico. |

## Acciones adicionales del informe y del correo

Estas acciones complementan los comentarios; no deben perderse al resolver solo las anotaciones de Word.

| Acción | Motivo | Evidencia de cierre |
|---|---|---|
| Control de atributos permutados dentro de especie | Determinar si la información morfológica aporta señal y no solo perturba el orden. | Protocolo fijado y comparación contra atributos reales con el mismo sistema visual. |
| Ablación de cada compuerta | Distinguir el efecto del margen visual y de la ventaja textual. | Condiciones comparables, con rescates, daños y cambios reales; selección sin usar test. |
| Resultados macro por identidad y por especie | Evitar ocultar concentración de ganancia en ciertos animales o especies. | Métricas con denominadores y definición del promedio. |
| Condiciones de captura difíciles | Los filtros excluyen escenas múltiples o baja visibilidad; eso limita el alcance. | Descripción del sesgo y evaluación separada de condiciones omitidas cuando haya datos adecuados. |
| Nueva cohorte y construcción de galerías | Una galería fija no estima generalización a galerías nuevas. | Cohorte no usada para desarrollo o protocolo repetido de galerías, fijado previamente. |
| PDF de todas las referencias | Adrián solicita validar cada cita y el estado del arte. | Inventario por referencia con título, DOI, PDF disponible y afirmaciones respaldadas; registrar cualquier faltante. No se ha completado esta auditoría. |
| Texto alternativo de ambas figuras | Mejorar accesibilidad del documento. | Alt text presente y coherente con el contenido. |
| Revista y tipo de envío | Q1 no determina un formato único. El encargo previo pide IEEE y máximo seis páginas. | Destino y reglas confirmados; hasta entonces conservar la restricción de seis páginas y mover evidencia extensa a suplemento cuando proceda. |

## Correcciones numéricas consolidadas

| Ubicación | Actual | Correcto |
|---|---:|---:|
| E3 público Rank-5 | 98,90 % | 98,92 % |
| E3 privado v6 Rank-1 | 80,10 % | 80,08 % |
| E3 privado v6 mAP | 0,6840 | 0,6843 |
| Ganancia privada E3→E4 Rank-1 | 6,15 puntos | 6,18 puntos |

E5 sin compuertas supera al E5 con compuertas en Rank-1 privado (87,65 % frente a 87,45 %) y mAP (0,7482 frente a 0,7346), pero daña tres aciertos en lugar de dos. En público empatan en Rank-1, mientras la versión con compuertas conserva mejor Rank-2. Esto respalda describir un intercambio entre beneficios y daños, no una superioridad general de las compuertas.

Los valores p ya reportados son 1,000 en público y 0,109375 en privado. Los intervalos por fotografía son [−1,44; 2,17] y [0,00; 2,39] puntos porcentuales; el extremo privado redondeado a cero no demuestra una mejora estrictamente positiva. Ausencia de significación tampoco demuestra equivalencia.

## Orden de ejecución propuesto

1. **Congelar la evidencia existente.** Guardar revisiones, salidas completas, fuentes de tablas y cronología. Mantener intactos los resultados y notebooks ejecutados originales.
2. **Corregir el manuscrito con evidencia disponible.** Resolver cifras, métricas, lenguaje de compuertas, carácter exploratorio de E5, figuras y disponibilidad. No esperar a nuevos entrenamientos para corregir errores conocidos.
3. **Completar auditoría sin reentrenar.** Recuperar registros por consulta, auditar Luna, contar cambios reales, identificar fallbacks y calcular desgloses e incertidumbre adecuada. Si requiere nueva evaluación, publicarla como tal.
4. **Definir las nuevas preguntas experimentales.** Fijar control E4 sin cambios, manejo de desconocidos, controles morfológicos, semillas y competidor. Registrar hipótesis y criterios de selección antes de ejecutar.
5. **Ejecutar y publicar.** Evaluar público y PetTrace por separado. Repetir E4 y realizar la confirmación entre sesiones según el protocolo. Conservar resultados negativos.
6. **Preparar paquete de revisión.** Manuscrito corregido, suplemento de evidencia, inventario de citas y matriz de respuesta a C01–C38. Decidir el envío después de revisar los nuevos resultados.

## Revisión adicional del código E1–E4 y propuesta de E6

Esta sección responde a la revisión del código solicitada después del análisis de Adrián. Los hallazgos I01 e I02 son problemas comprobados de implementación/protocolo. Las demás ideas son hipótesis pendientes de evaluación; no son mejoras de rendimiento demostradas.

### Dos hallazgos que deben preceder a las nuevas corridas

**I01. E4 entrena la proyección visual además de LoRA.** `build_encoder()` incorpora `clip.visual_projection` y congela únicamente `encoder.vision_model` mediante `inject_lora()`. La cabeza E3 sí llega congelada. La proyección visual conserva `requires_grad=True` y el optimizador incluye todos los parámetros entrenables. Esto contradice la descripción de proyección congelada en el protocolo.

- Evidencia: `src/pettrace/training/lora.py`, funciones `build_encoder`, `inject_lora` y `fit`.
- Comprobación: un CLIP pequeño creado localmente, sin descarga ni datos reales, pasó por el constructor del repositorio; `visual_projection.weight` apareció entre parámetros entrenables y recibió gradiente no nulo.
- Trazabilidad histórica: el archivo es idéntico en la revisión `64c79c9`, que declara versión 0.2.39, y en HEAD. El notebook ejecutado fija 0.2.39. No se inspeccionó el checkpoint histórico de entrenamiento en esta comprobación.
- Consecuencia: el resultado reproducido de E4 sigue siendo evidencia de adaptación visual, pero no permite atribuir toda la ganancia exclusivamente a LoRA.
- Acción propuesta: conservar el E4 publicado e identificarlo correctamente. Comparar E3, ajuste de proyección solamente, LoRA con proyección congelada y LoRA más proyección. Fijar presupuesto, semillas y selección para separar las contribuciones. No reemplazar silenciosamente pesos ni resultados históricos.

**I02. La semilla de E4 se fija después de inicializar los adaptadores.** `LoRALinear` inicializa aleatoriamente su matriz A durante `build_encoder()`. `seed_all(config.seed)` se ejecuta después, dentro de `fit()`. El notebook no fija la semilla antes de construir el encoder.

- Comprobación: dos construcciones con `Config(seed=46)` y distinto estado aleatorio previo produjeron matrices A diferentes, incluso tras ejecutar el reinicio de semilla que hace `fit()`.
- Consecuencia: repetir “semilla 46” no garantiza reproducir la inicialización. La equivalencia inicial con E3 puede seguir pasando porque la matriz B empieza en cero.
- Acción propuesta: fijar la semilla antes de crear adaptadores, verificar igualdad de inicializaciones y registrar configuración y parámetros entrenables. Esto mejora control experimental, no garantiza una subida de métricas.

### Oportunidades concretas de mejora

| Orden | Etapa y evidencia actual | Hipótesis y prueba propuesta | Coste relativo y precaución |
|---|---|---|---|
| 1 | E3/E4: el sampler elige ocho identidades aleatorias y cuatro fotos por identidad. Cada ancla tiene siete identidades negativas posibles; batch-hard solo busca dentro del lote. | Comparar 8×4 con 16×2 manteniendo 32 imágenes por paso. Después, probar muestreo de identidades visualmente próximas dentro de train. | E3 permite pilotos económicos con embeddings cacheados. 16×2 aumenta diversidad de negativos, pero reduce positivos por ancla: no es mejora automática. No usar test para elegir negativos. |
| 2 | E3/E4: 30 épocas × 100 pasos × 32 imágenes = 96.000 presentaciones como máximo, frente a 193.276 fotos de train. El checkpoint E4 de época 27 corresponde a 86.400 presentaciones. | Medir cobertura real por especie e identidad y comparar presupuestos mayores o muestreo con cobertura controlada. | El límite superior de cobertura única en una corrida completa es 49,67 %; las repeticiones lo reducen. No equivale a demostrar subentrenamiento. Comparar por pasos/exposición y tiempo, no por la palabra “época”; mantener frecuencia de validación comparable. |
| 3 | E3: MLP 512→256→512 y pérdida batch-hard triplet con margen fijo. | Probar una cabeza residual que conserve parte de la representación base, o una pérdida que aproveche más pares del lote, primero por separado. | Se pueden reutilizar embeddings congelados. Cambiar arquitectura y pérdida juntas impediría atribuir la ganancia. [Circle Loss](https://openaccess.thecvf.com/content_CVPR_2020/html/Sun_Circle_Loss_A_Unified_Perspective_of_Pair_Similarity_Optimization_CVPR_2020_paper.html) es un método externo candidato, no una contribución propia ni una mejora ya verificada aquí. |
| 4 | E2/E4: tras recortar con YOLO se redimensiona por lado corto y se aplica recorte central. | Comparar conservación del recorte completo mediante padding cuadrado con la política actual; auditar visualmente qué partes se pierden. | En un recorte 2:1 llevado a un cuadrado mediante recorte central se descarta la mitad del eje largo. No se midió cuántas fotos reales sufren este caso. Cambiar píxeles de entrada requiere una variante explícita y coherencia entre entrenamiento e inferencia. |
| 5 | E4: entrada determinista, sin augmentations; AdamW con tasa constante. | Probar transformaciones suaves que conserven identidad —pequeñas variaciones de encuadre y exposición— y, en otra comparación, warmup/decaimiento de tasa. | No introducir cambios fuertes de color ni recortes que eliminen señales de identidad. La utilidad debe verse en validación y luego confirmarse; el flip en inferencia ya tuvo resultados históricos desfavorables para Rank-1. |
| 6 | E4: MLP E3 congelada mientras cambia la representación visual que recibe. | Comparar ajuste conjunto del head con tasa menor frente al head congelado, con idéntico presupuesto. | Es una variante adicional de adaptación; distinguirla del problema I01. Puede sobreajustar y no debe combinarse de entrada con todas las otras modificaciones. |

Además, `pk_batch()` reconstruye los grupos de todas las fotografías y ordena las identidades en cada paso. `PreparedPreprocessor.pixels()` decodifica y procesa imágenes de forma síncrona. Precalcular el índice del sampler y medir el tiempo de carga permitiría aprovechar mejor Colab. Son mejoras de eficiencia: por sí solas no elevan Rank-1. El entrenamiento actual tampoco usa precisión mixta; se debe medir estabilidad y rendimiento en la GPU asignada antes de adoptarla.

No se recomienda cambiar E1 y E2 históricos para elevar sus cifras: son controles. Cada nueva política de recorte o backbone debe tener una variante y procedencia identificables. Tampoco ampliar indefinidamente la búsqueda sobre las 60 identidades de validación; congelar un conjunto pequeño de hipótesis y reservar confirmación independiente.

### E6 propuesto como verificación visual del Top-5

**Actualización:** esta propuesta sigue sin implementar. Los resultados posteriores de C1 cambian la prioridad hacia adaptar el encoder independiente y validar entre sesiones. Una capa adicional tendría que justificar su aporte frente al control fuerte, no solo frente al E4 histórico.

**Pregunta:** ¿una comparación visual local entre la consulta y sus candidatos reduce las confusiones que quedan tras la recuperación global de E4?

La motivación está en el resultado existente: E4 público obtiene 237 aciertos Rank-1, 266 Rank-2 y 275 Rank-5 sobre 277 consultas. De sus 40 errores de primer puesto, 29 tienen la identidad correcta segunda y 38 la tienen dentro del Top-5. El 95 % de esos errores son, por tanto, potencialmente abordables dentro de la lista actual. **Esto es un diagnóstico retrospectivo y un techo ideal; no una predicción de lo que E6 logrará.** Un reranker también puede dañar consultas que E4 resolvía bien.

Propuesta de rama experimental:

```text
E4 congelado → cinco identidades candidatas → comparación visual por pares → ranking final
             ↘ E5 morfológico permanece como comparación exploratoria separada
```

El piloto E6 compararía la consulta con una fotografía de referencia por identidad, elegida por la puntuación E4, excluyendo siempre la propia consulta. Usaría rasgos locales de las imágenes para aprender una puntuación adicional. Los pares positivos y negativos difíciles procederían solo de identidades de entrenamiento; la selección y la combinación con el score global se fijarían en validación. Incluir la opción de conservar E4 sin intervención.

Este enfoque tiene antecedentes: [ETR](https://openaccess.thecvf.com/content/WACV2023/papers/Zhang_ETR_An_Efficient_Transformer_for_Re-Ranking_in_Visual_Place_Recognition_WACV_2023_paper.pdf) combina descriptores globales y locales para reranking en reconocimiento de lugares; [TransReID](https://openaccess.thecvf.com/content/ICCV2021/html/He_TransReID_Transformer-Based_Object_Re-Identification_ICCV_2021_paper.html) investiga representaciones globales/locales en Re-ID de personas y vehículos. Son motivaciones externas de diseño, no evidencia de rendimiento en mascotas ni garantía de novedad.

**Dependencia técnica:** el encoder exportado actual devuelve un vector global de 512 dimensiones. No entrega los tokens locales que necesitaría esta propuesta. E6 requiere exponer y cachear esos rasgos, entrenar el comparador y definir su exportación, agregación por identidad y latencia; no es simplemente añadir otra MLP al vector final.

Controles mínimos antes de considerar E6 sólido:

1. E4 congelado sin reranker.
2. Comparador entrenado usando solo rasgos globales, con el mismo protocolo de pares.
3. Comparador con evidencia local, manteniendo fijo el E4 de origen.
4. Comparación del beneficio frente a invertir un presupuesto adicional comparable en E4, si se afirma eficiencia o superioridad de la nueva etapa.

Reportar varias semillas, rescates, daños, cambios reales, resultados por especie/identidad, incertidumbre, tiempo por consulta y memoria. Fijar cómo se calculan los scores por fotografía para mAP; no mezclar un ranking por identidad con AP por fotografía sin definición explícita. Si se permutan solo cinco identidades, conservar Rank-5 es estructural y no una segunda mejora.

E6 no necesita depender de E5 ni incorporar sus atributos. Su aporte sería empírico si demuestra que la evidencia visual local resuelve confusiones en identidades no usadas para entrenar y en sesiones separadas. Sin esa evidencia, sigue siendo una hipótesis razonable. No sustituye al competidor independiente solicitado por Adrián.

### Decisión recomendada para el presupuesto disponible

La recomendación inicial era resolver I01 e I02 y estudiar muestreo sobre CLIP. Ambos hallazgos ya se documentaron y el estudio de seis corridas terminó sin mejora general. La prioridad actual es el [protocolo DINO](dinov2_adaptation_protocol.md): medir primero su coste real, comparar proyección y LoRA por separado contra C1 y reservar recursos para semillas y confirmación entre sesiones. E6 y la ablación CLIP quedan en segundo plano.

En la fase inicial, el saldo mostrado era 295,58 unidades de cómputo, no horas. Todavía no se había estimado un número garantizado de corridas ni lanzado Colab. Las comprobaciones descritas fueron pruebas locales de implementación con un modelo pequeño y datos sintéticos, no entrenamientos ni evaluaciones del benchmark.

## Reglas para las nuevas ejecuciones

- Publicar todos los experimentos en `https://ml.pettrace.app`; una ejecución no está completa hasta que la publicación sea exitosa.
- Ante errores de publicación, preservar evidencia local y permitir reintento sin reentrenar. No introducir opciones para desactivar el seguimiento.
- Mantener registro de versión y promoción de alias separados de publicación de resultados, respetando la configuración elegida por el usuario.
- Preservar salidas de notebooks y no incluir credenciales. Usar notebooks únicamente para experimentos y preparación de datos, dentro de las tres fases documentadas.
- Conservar el test público inmutable y las galerías separadas; no seleccionar checkpoints ni nuevas reglas usando resultados de test.
- Identificar el nuevo análisis como retrospectivo cuando reutilice benchmarks observados. Cambiar el método ahora no convierte los datos anteriores en un holdout nuevo.
- Cada experimento debe declarar objetivo, pregunta, hipótesis, intervención, controles, muestra, protocolo de entrenamiento/selección, métricas, resultados y limitaciones.

## Criterio para cerrar la revisión

Cada comentario debe quedar acompañado de una ubicación del cambio o de una justificación explícita de alcance, más la evidencia correspondiente. Un experimento propuesto, una aserción sin salida o una promesa de publicación no resuelven una observación empírica.

Si E4 mantiene su ganancia con semillas, competidor y evaluación más representativa, el artículo puede centrarse en esa adaptación. Si E5 continúa inconcluso o pierde contra el control sin intervención, debe conservarse como resultado exploratorio o negativo. El caso de Luna ayuda a explicar decisiones concretas, pero no reemplaza la evaluación global.

En la fase inicial del análisis no se modificaron el manuscrito, el código ni los resultados, ni se reprodujo inferencia. La verificación de los PDF y la ejecución posterior se registran en la actualización siguiente. El cierre de estos puntos fortalece la evidencia, pero no garantiza aceptación en una revista Q1.

## Ejecución autorizada y fuentes verificadas · 2026-11-18

Esta actualización sucede al análisis prospectivo anterior. Se completó la comprobación de las
37 referencias del manuscrito: 32 PDF académicos y cinco fichas para fuentes web, con índice,
versiones identificadas, observaciones de sustento y copia del manuscrito. El paquete está en
`../pettrace/share/P202620066_Fuentes_Adrian.zip` y su auditoría en
`../pettrace/share/P202620066_Fuentes_Adrian/AUDITORIA.md`. Se verificaron integridad del ZIP,
hashes de archivos, lectura de los 39 PDF (incluidos índice y manuscrito) y enlaces relativos.
No se enviaron archivos al profesor ni se modificó el paper consultado.

Se corrigió la inicialización de la semilla en `build_encoder` y se conservó explícitamente el
conjunto histórico de parámetros: LoRA q/v más proyección visual. La comparación nueva está en
`notebooks/e4_sampling_multiseed.ipynb`: P8×K4 y P16×K2, semillas 45–47, igual presupuesto de
imágenes/pasos y selección por validación. Esta ejecución directa de E4 permite probar una
intervención acotada y estimar dispersión de optimización; no constituye aún una ablación
LoRA-solo/proyección-sola ni una evaluación entre sesiones.

[Notebook en Colab](https://colab.research.google.com/drive/1_mtyN57j9BJHrwjVWusUWh3K7qxZt3L-).
Inicio de sesión: saldo 295,58 unidades, ninguna sesión previa. Se solicitó A100, pero Colab
asignó NVIDIA L4 por falta de disponibilidad. Entorno 0.2.46 y acceso a credenciales verificados;
se inició la descarga de snapshots. Reporte de esta sesión:
`/content/reports/e4_sampling_study/20261118T053018Z-9d7ddbb5`.

Estado al registrar esta entrada: **primera corrida en entrenamiento (P8×K4, semilla 45);
193.826 fotos públicas verificadas y galerías cargadas; sin métricas de test nuevas ni
recibos de publicación**. No atribuir mejora de rendimiento a las pruebas de código. Se preservaron las
salidas del notebook E4 histórico. Los checks de E4 y protocolo nuevo pasaron; la suite detectó
seis fallos existentes en los checks del notebook Luna (guard de versión Python ausente y
expectativa de commit v5 frente al v6 utilizado), separados de esta intervención.

### Evidencia adicional del checkpoint histórico

Se descargó de MLflow `report/adapter.pt` del run E4 v3
`fbf9ca83a0df4c48b1063ea9c609295d`, cuyo estado es FINISHED. Su SHA-256 es
`1a579dfcd8bcd49b737ea6ae65c952ab19ebf1f7b1fa442eef8cb20953a14ac2`.
Contiene 49 tensores y 688.128 parámetros: las matrices A/B de los 24 adaptadores
más `visual_projection.weight` de forma 512×768. El checkpoint corresponde a época 27,
con mAP macro de validación 0,8531207848327025. Esto confirma que la proyección está
incluida en la evidencia de entrenamiento publicada, no solo en una inspección del
código actual. Metadatos locales: `reports/e4-implementation-audit/checkpoint-metadata.json`.
Esta lectura de un artefacto ya publicado no es otra corrida experimental.

Interpretación prevista del contraste P×K: ambos brazos procesan 32 imágenes por paso, pero
P16×K2 también reduce los positivos por ancla y evita el relleno con fotografías repetidas
cuando una identidad tiene solo dos o tres imágenes. Por tanto, se medirá la cobertura única
registrada por `fit`; no se atribuirá cualquier diferencia exclusivamente a negativos más
numerosos. El control E3 publicado registra mejor mAP macro de validación
0,7993994602552579 (época 27); el E4 histórico, 0,8531207848327025. Estos valores de
validación no son los mAP de test y no deben intercambiarse en tablas.

### Ablación prospectiva de componentes

Se preparó `notebooks/e4_component_ablation.ipynb` antes de observar tests nuevos: solo
LoRA q/v y solo proyección visual, semillas 45–47, P8×K4. El control combinado son las
tres corridas P8×K4 del estudio `20261118T053018Z-9d7ddbb5`. Se mantienen cabeza, datos,
recortes, margen, tasa, pasos máximos y paciencia. La elección dentro de los dos brazos
nuevos utiliza la media de validación y después la mejor semilla de validación; no implica
que el seleccionado supere al control combinado. Las seis condiciones se publicarán en
ambos dominios. Los conjuntos exactos de parámetros se verifican antes de entrenar.
Estado de esta entrada: preparado y pendiente de ejecución, sin resultados nuevos.

### Referencia independiente C1

Se fijó `notebooks/c1_dinov2_reference.ipynb`: DINOv2-L/14 congelado, token CLS
normalizado y procesador oficial (256→224), con el detector/recorte E2. Revisión HF
`47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`. Se verifican contra el recibo E4 los
hashes e identidades de las 277 consultas públicas y el snapshot privado v6 de 502
fotos. No hay ajuste ni selección de hiperparámetros. Inferencia local MPS con paridad
CPU/MPS sobre imágenes sintéticas antes de test. Publicación obligatoria en MLflow,
sin registro ni promoción de aliases para este control externo.

Es un comparador de arquitectura y preentrenamiento distintos, no una ablación ni
un aporte propio. No resuelve por sí solo la exigencia de evaluación entre sesiones
y no se puede descartar todo solapamiento del preentrenamiento externo con fotos públicas.
Estado de esta entrada: ejecución iniciada; resultados y recibos pendientes.

### C1 ejecutado y verificado

DINOv2-L congelado obtuvo 255/277 (92,06 %) y mAP 0,8383 en público;
476/502 (94,82 %) y mAP 0,8650 en PetTrace v6. Frente al E4 histórico,
las diferencias netas de Rank-1 son +18 y +43 consultas. Runs FINISHED, commits,
fingerprints y métricas comprobados por API: público `3e604ebd78254956a2bc3f9a9f9a199c`,
privado `39e0a0fc04904406893ca2adbac98e6b`. Notebook ejecutado y suplemento
subidos a ambos runs, con hash del notebook verificado tras descargarlo.

[Informe C1](c1_dinov2_results_20261118.md). Este es un comparador externo más grande,
no una mejora atribuible a nuestro LoRA. La ganancia pública de mAP es pequeña y su
intervalo descriptivo incluye cero. El resultado refuerza la prioridad de adaptar
el backbone competitivo y validar entre sesiones antes de prometer una capa E6.
Las corridas nuevas de E4 posteriormente finalizaron: [seis entrenamientos y doce evaluaciones publicados](e4_sampling_results_20261118.md), sin mejora general del candidato seleccionado.


### Exposición efectiva medida en las seis nuevas corridas E4

El respaldo local completo confirma 14–30 épocas, 1.400–3.000 pasos,
44.800–96.000 presentaciones y 39.373–73.301 fotografías distintas por corrida.
Las seis fases de entrenamiento suman 8.685,02 segundos (2,41 horas) en L4;
este total no incluye preparación, exportación, evaluación ni publicación.
La disponibilidad del 100 % del corpus no significa haber recorrido todas sus fotos.
El presupuesto máximo y la regla de parada son comunes, pero la exposición efectiva
varía por la parada temprana.

Una posible siguiente intervención es aumentar el presupuesto de pasos con la misma
regla de selección por validación, conservando el control de presupuesto actual.
Debe declararse como intervención de presupuesto, no atribuir su efecto a una capa
nueva. Todavía no se ha ejecutado ni se afirma que mejore. La comparación sobre
DINOv2 y la evaluación entre sesiones siguen siendo prioridades independientes.
