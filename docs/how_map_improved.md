# Cómo PetTrace elevó el mAP: lectura técnica de E1 a E5

## 1. Qué intenta explicar este documento

Este documento no describe cómo ejecutar los notebooks. Explica **por qué** cada etapa del sistema cambió la calidad del ranking y cómo se llegó desde un descriptor CLIP sin entrenamiento hasta el modelo visual E4 y el reordenamiento semántico E5.

La conclusión central es sencilla:

> El mayor aumento de mAP no vino de añadir más componentes, sino de transformar un espacio visual genérico en un espacio métrico especializado en identidad. E3 produjo el salto principal; E4 refinó el extractor visual; E5 actuó únicamente sobre ambigüedades residuales.

Los resultados se interpretan bajo el protocolo real del proyecto: recuperación *closed-set*, comparación dentro de la misma especie y evaluación *leave-one-out* a nivel de fotografía. Los conjuntos Public y PetTrace se evalúan por separado y nunca comparten galería.

## 2. Qué mide realmente el mAP

Para una fotografía consulta \(q\), la galería contiene fotografías de otros animales de la misma especie y las demás fotografías del animal correcto. Si \(R_q\) es el conjunto de fotografías relevantes, la precisión promedio es:

\[
AP(q)=\frac{1}{|R_q|}\sum_{k=1}^{N}P_q(k)\,\operatorname{rel}_q(k),
\]

donde \(P_q(k)\) es la precisión acumulada hasta la posición \(k\) y \(\operatorname{rel}_q(k)=1\) cuando la fotografía situada en esa posición pertenece al mismo animal que la consulta. El resultado global es:

\[
mAP=\frac{1}{|Q|}\sum_{q\in Q}AP(q).
\]

Por tanto, mAP no pregunta solamente si el animal correcto quedó primero. Premia que **todas sus fotografías aparezcan temprano** y penaliza cada fotografía impostora intercalada antes de ellas.

Ejemplo pedagógico, no resultado experimental:

- Relevancia `[1, 0, 1, 0]`: \(AP=(1+2/3)/2=0.833\).
- Relevancia `[0, 1, 0, 1]`: \(AP=(1/2+2/4)/2=0.500\).

Ambas listas recuperan dos positivos, pero la primera organiza mucho mejor la evidencia.

### Rank-k y mAP no usan exactamente la misma agregación

En PetTrace, Rank-k es una métrica por identidad. Para cada identidad se toma el máximo coseno entre la consulta y sus fotografías de galería; luego se ordenan las identidades. En cambio, el mAP base conserva el ranking de fotografías individuales.

Esto explica dos observaciones importantes:

1. Cambiar `max` por un prototipo de identidad puede alterar Rank-1 sin cambiar el mAP base.
2. Una mejora de mAP indica una reorganización más amplia de las fotografías relevantes, no solo un acierto adicional en la primera posición.

## 3. La geometría común a todos los experimentos

Cada modelo produce un vector \(z\), normalizado a longitud unitaria:

\[
\hat z=\frac{z}{\lVert z\rVert_2}.
\]

La similitud visual es el coseno:

\[
s(q,g)=\hat z_q^\top \hat z_g.
\]

Sobre la esfera unitaria, distancia euclidiana y coseno están ligados:

\[
\lVert \hat z_q-\hat z_g\rVert_2^2=2-2s(q,g).
\]

Por eso E3 puede entrenarse con distancia euclidiana y evaluarse con coseno sin introducir objetivos geométricos contradictorios: acercar dos vectores normalizados también aumenta su coseno.

## 4. Evolución cuantitativa en Public v2

Public v2 permite seguir E1–E5 sobre una población común. Los valores siguientes provienen de los notebooks ejecutados y sus reportes.

| Etapa | Intervención principal | Rank-1 | mAP | Δ mAP frente a la etapa anterior |
|---|---|---:|---:|---:|
| E1 | CLIP congelado sobre imagen completa | 68.59% | 0.5501 | — |
| E2 | Recorte YOLO + CLIP congelado | 67.87% | 0.5262 | -0.0239 |
| E3 | CLIP congelado + cabeza MLP con triplet loss | 81.59% | 0.7672 | +0.2410 |
| E4 | LoRA en CLIP + cabeza E3 congelada | 85.56% | 0.8345 | +0.0673 |
| E5 gated | Re-ranking por atributos sobre E4 | 85.92% | 0.8346 | +0.0001 |

La tabla separa tres tipos de aporte:

- **E2 cambia la observación**: qué píxeles ve el modelo.
- **E3 y E4 cambian la representación**: cómo se organiza el espacio de embeddings.
- **E5 cambia la decisión**: cómo se resuelven empates visuales después de obtener los embeddings.

## 5. E1: CLIP como descriptor visual genérico

E1 utiliza CLIP ViT-B/32 sin entrenamiento para Pet Re-ID. CLIP ya contiene una representación semántica potente porque fue preentrenado contrastivamente con imágenes y texto, pero su objetivo original no exige distinguir dos gatos visualmente parecidos ni agrupar fotografías del mismo animal bajo pose, iluminación y fondo diferentes.

El resultado de E1 es importante como control: demuestra cuánto puede recuperarse usando únicamente conocimiento visual transferido. Su limitación geométrica es que atributos semánticos gruesos —especie, color, contexto o tipo de escena— pueden dominar sobre señales finas de identidad.

## 6. E2: localizar al animal no equivale a aprender su identidad

E2 aplica YOLO antes de CLIP para retirar buena parte del fondo. En Public v2, el mAP baja de 0.5501 a 0.5262.

Este resultado negativo es técnicamente coherente. El CLIP congelado de E1 podía aprovechar correlaciones contextuales: suelo, habitación, manos, cámara o composición recurrente. El recorte elimina parte de esa información sin enseñar todavía al encoder qué detalles del animal deben reemplazarla.

No se concluye que YOLO sea inútil. E2 aporta:

- una entrada centrada en el animal;
- mayor consistencia entre entrenamiento y despliegue;
- menor dependencia de fondos potencialmente espurios;
- una base más limpia para que E3 y E4 aprendan identidad.

La lectura correcta es: **el recorte, por sí solo, no mejora un descriptor congelado**. Su valor aparece al combinarlo con aprendizaje métrico posterior.

## 7. E3: el salto principal mediante aprendizaje métrico

E3 deja CLIP congelado y aprende una proyección residual sencilla:

\[
h(z)=\operatorname{norm}\bigl(W_2\,\operatorname{ReLU}(W_1z)\bigr),
\]

con dimensiones \(512\rightarrow256\rightarrow512\). La cabeza no necesita reconstruir la imagen: solo reorganiza las características existentes para que sean útiles para identidad.

### 7.1 Triplet Margin Loss

Para un ancla \(a\), un positivo \(p\) del mismo animal y un negativo \(n\) de otro animal:

\[
L=\max\left(0, d(a,p)-d(a,n)+m\right),
\]

con margen \(m=0.2\). La pérdida es cero cuando el negativo ya está suficientemente más lejos que el positivo.

E3 usa *batch-hard mining*. Dentro de cada lote selecciona:

- el positivo más lejano del ancla;
- el negativo más cercano al ancla.

En consecuencia, el gradiente se concentra en variaciones difíciles del mismo animal y en los impostores visuales más peligrosos disponibles en ese lote.

### 7.2 Muestreo P×K

Cada lote contiene \(P=8\) identidades y \(K=4\) fotografías por identidad. Esto garantiza positivos y negativos útiles dentro del lote. El entrenamiento alterna especies, evitando que el problema se reduzca a separar perros de gatos.

### 7.3 Por qué E3 elevó tanto el mAP

E3 produjo el mayor salto: **+0.2410 de mAP** en Public v2. La explicación geométrica es global:

1. compactó las distintas vistas de cada animal;
2. separó de forma explícita los rivales visuales difíciles;
3. corrigió dimensiones de CLIP útiles para semántica general pero poco discriminativas para identidad;
4. mejoró simultáneamente la posición de varias fotografías positivas, justo lo que recompensa AP.

El checkpoint no es el de la última época. Se selecciona por mAP macro por especie en validación, se guarda solo cuando mejora y al final se vuelve a cargar el mejor estado. Los conjuntos de test permanecen sin leer durante esta selección.

Las semillas no crean información nueva. Permiten medir la variabilidad introducida por inicialización, muestreo P×K y optimización. La semilla 46 fue seleccionada por validación, no por sus resultados de test.

## 8. E4: adaptar la atención de CLIP con LoRA

E4 parte del espacio E3 ya aprendido. La cabeza MLP queda congelada y se insertan adaptadores LoRA únicamente en las proyecciones `q_proj` y `v_proj` de CLIP.

Para una capa lineal congelada \(W\), LoRA aprende una actualización de bajo rango:

\[
y=Wx+\frac{\alpha}{r}BAx,
\]

con rango \(r=8\) y \(\alpha=16\). En el código, la matriz \(B\) comienza en cero; por tanto, antes de entrenar:

\[
BA=0 \quad\Rightarrow\quad y=Wx.
\]

Una prueba de equivalencia comprueba que E4 empieza reproduciendo E3. Esto hace que el experimento mida la adaptación aprendida y no una inicialización accidentalmente distinta.

### 8.1 Por qué LoRA aporta después del MLP

La cabeza E3 solo puede recombinar características que CLIP ya extrajo. LoRA modifica ligeramente **qué relaciones visuales atiende el encoder**: manchas, contornos faciales, distribución del color o morfología local pueden adquirir mayor influencia antes de llegar a la cabeza.

El mAP de Public v2 sube de 0.7672 a 0.8345, un aumento de 0.0673. Es menor que E3 porque el espacio ya era competitivo, pero es una mejora amplia: no corrige únicamente el primer vecino, sino el orden de múltiples fotografías relevantes.

El entrenamiento conserva el mismo principio de higiene: validación para seleccionar el mejor checkpoint, test leído una sola vez después de congelar la decisión.

## 9. E5: una corrección semántica deliberadamente limitada

E5 no vuelve a entrenar el embedding visual. Extrae atributos observables de los recortes YOLO —color, patrón, longitud del pelaje, orejas y marcas— y reordena únicamente el Top-5 de E4.

La compatibilidad textual pondera más el color primario y el patrón, ignora campos `unknown` o no visibles y penaliza contradicciones directas. Para una consulta ambigua, el score fusionado es:

\[
S_{total}(q,c)=z_v(q,c)+\lambda z_t(q,c).
\]

Sin embargo, la fusión solo puede actuar cuando se cumplen dos condiciones:

1. la diferencia visual entre las dos primeras identidades es menor que \(\tau_v\);
2. la candidata alternativa posee una ventaja textual mínima \(\tau_t\).

Los umbrales se congelaron usando únicamente validación. En la ejecución registrada fueron \(\tau_v=0.01\), \(\lambda=0.02\) y \(\tau_t=0.05\).

### 9.1 Por qué E5 apenas mueve el mAP público

La versión gated elevó Rank-1 público en 0.36 puntos porcentuales, pero el mAP solo pasó de 0.8345 a 0.8346. Esto es esperable:

- la compuerta se activa en una fracción pequeña de consultas;
- E4 ya deja Rank-5 por encima de 99%;
- E5 intenta intercambiar candidatos cercanos, no reconstruir todo el espacio;
- proteger aciertos confiables limita deliberadamente la magnitud posible del cambio.

La variante sin compuerta alcanzó mAP 0.8363, pero dañó tres consultas correctas y redujo Rank-2. La variante gated mantuvo Rank-2 y restringió la intervención. E5 debe entenderse como una capa de seguridad interpretable para casos ambiguos, no como otro aprendizaje métrico global.

## 10. Generalización en PetTrace v6

Los resultados históricos de E1–E3 sobre PetTrace v5 no deben concatenarse directamente con E4–E5 sobre v6: cambió la población del benchmark. Para la comparación válida se usan controles reevaluados sobre v6.

| Etapa | Rank-1 | Rank-2 | Rank-5 | mAP |
|---|---:|---:|---:|---:|
| E3, control reevaluado en v6 | 80.10% | — | — | 0.6840 |
| E4, baseline visual | 86.25% | 94.22% | 99.00% | 0.7333 |
| E5, double-gated | 87.45% | 94.62% | 99.00% | 0.7346 |
| E5, sin compuerta | 87.65% | 94.82% | 99.00% | 0.7482 |

E3→E4 añade 6.15 puntos de Rank-1 y 0.0493 de mAP sobre la misma versión. E5 gated añade 1.20 puntos de Rank-1 y 0.0013 de mAP. La variante sin compuerta aumenta más el mAP, pero sacrifica la política conservadora elegida para despliegue.

Que la mejora E4 también aparezca en PetTrace v6 respalda que LoRA no se limitó a memorizar la validación pública. No demuestra, por sí solo, generalización a cualquier población futura.

## 11. Experimentos que evitaron conclusiones equivocadas

### 11.1 TTA y agregación por identidad

La ablación de validación comparó vista única, *horizontal-flip TTA*, `max`, media Top-2 y prototipo esférico. Ninguna alternativa superó el Rank-1 de `single + max`:

| Representación | Agregación | Δ Rank-1 | Δ mAP |
|---|---|---:|---:|
| horizontal flip TTA | max | -0.007613 | +0.001724 |
| single | top-2 mean | -0.019771 | 0.000000 |
| horizontal flip TTA | prototype | -0.021472 | +0.001724 |
| single | prototype | -0.023062 | 0.000000 |
| horizontal flip TTA | top-2 mean | -0.028974 | +0.001724 |

Estas variantes se descartaron antes de leer los tests. La diferencia nula de mAP entre agregaciones de identidad es consistente con su definición fotográfica; el pequeño cambio con TTA proviene de modificar los embeddings.

### 11.2 Curación de datos

Eliminar duplicados, conflictos de identidad e imágenes inservibles no debe presentarse como una ganancia cuantificada de mAP si no existe una ablación pareada. Su contribución demostrable es metodológica: evita que copias de la misma imagen o etiquetas contradictorias inflen o corrompan las métricas.

## 12. Qué empujó el mAP y cuánto

En orden de impacto observado:

1. **Aprendizaje métrico E3 (+0.2410)**: cambió globalmente la geometría y ordenó mejor todas las vistas positivas.
2. **Adaptación LoRA E4 (+0.0673)**: permitió que el encoder extrajera señales finas de identidad que la cabeza sola no podía crear.
3. **Re-ranking E5 gated (+0.0001 público; +0.0013 PetTrace)**: resolvió una parte de los empates visuales, con una política conservadora.
4. **Recorte YOLO E2 (-0.0239 por sí solo)**: no elevó el descriptor congelado, pero redujo dependencia contextual y estandarizó la entrada de las etapas aprendidas.

La historia técnica no es “cada bloque aumenta la métrica”. Es:

> Localizar al animal hizo el problema más limpio; aprender una métrica produjo el gran salto; adaptar la atención refinó las señales de identidad; el lenguaje corrigió una fracción pequeña de errores residuales sin sustituir al modelo visual.

## 13. Límites que deben acompañar la interpretación

- El protocolo es *closed-set*: la identidad correcta está en la galería. No mide rechazo de animales desconocidos.
- La recuperación es dentro de la misma especie; no mide clasificación perro/gato.
- Cada fotografía actúa como consulta *leave-one-out*, por lo que las consultas del mismo animal no son observaciones poblacionales completamente independientes.
- El EER es descriptivo sobre los mismos pares, no una calibración independiente de producción.
- El *batch-hard mining* encuentra los casos más difíciles del lote de 32 imágenes, no necesariamente el negativo más difícil de todo el dataset.
- Tres o cinco semillas describen variabilidad de entrenamiento, no incertidumbre sobre toda la población de mascotas.
- Comparar versiones distintas de PetTrace puede confundir una mejora del modelo con un cambio de composición del benchmark.

## 14. Mapa de evidencia en el repositorio

- Definición de Rank-k, mAP y EER: [`src/pettrace/evaluation/metrics.py`](../src/pettrace/evaluation/metrics.py)
- Evaluación y extracción de embeddings: [`src/pettrace/evaluation/core.py`](../src/pettrace/evaluation/core.py)
- Cabeza MLP, muestreo P×K y triplet loss: [`src/pettrace/training/triplet.py`](../src/pettrace/training/triplet.py)
- Inyección y entrenamiento LoRA: [`src/pettrace/training/lora.py`](../src/pettrace/training/lora.py)
- Compatibilidad de atributos y doble compuerta: [`src/pettrace/evaluation/reranking.py`](../src/pettrace/evaluation/reranking.py)
- Protocolo E5 y controles congelados: [`src/pettrace/experiments/e5.py`](../src/pettrace/experiments/e5.py)
- Resultados ejecutados: [`notebooks/e1_clip_zero_shot.ipynb`](../notebooks/e1_clip_zero_shot.ipynb), [`notebooks/e2_yolo_crop_zero_shot.ipynb`](../notebooks/e2_yolo_crop_zero_shot.ipynb), [`notebooks/e3_mlp_head_triplet.ipynb`](../notebooks/e3_mlp_head_triplet.ipynb), [`notebooks/e4_clip_lora.ipynb`](../notebooks/e4_clip_lora.ipynb) y [`notebooks/e5_language_reranking.ipynb`](../notebooks/e5_language_reranking.ipynb)
- Decisiones y ablaciones de E4/E5: [`docs/e5_optimization_plan.md`](e5_optimization_plan.md)

## 15. Referencias metodológicas principales

- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html), 2021.
- Hermans, Beyer y Leibe, [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737), 2017.
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/forum?id=nZeVKeeFYf9), 2022.
- Zheng et al., [Scalable Person Re-identification: A Benchmark](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_A_ICCV_2015_paper.html), 2015.
- Jocher et al., [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), consultado como implementación del detector usado en el preprocesamiento.

## 16. Síntesis para defensa

Si hubiera que resumir el aporte en una sola respuesta técnica:

> El mAP aumentó principalmente porque E3 aprendió una geometría de identidad sobre embeddings CLIP normalizados, usando triplet loss con minería difícil y selección exclusivamente por validación. E4 añadió capacidad al encoder mediante actualizaciones LoRA de bajo rango, preservando como punto inicial el mejor espacio E3. E5 no buscó otra transformación global: aplicó atributos observables únicamente en empates visuales, por lo que su ganancia fue menor pero interpretable y controlada. Las ablaciones negativas —YOLO sin aprendizaje posterior, TTA y pooling alternativo— delimitan qué partes no explican la mejora.
