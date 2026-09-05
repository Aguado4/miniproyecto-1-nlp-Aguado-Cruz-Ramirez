# SPEC — Especificación del entregable

> **Estado:** aprobado · **Versión:** 1.0 · **Última actualización:** 2026-09-05
>
> Este documento es el contrato del entregable. Define qué debe contener el notebook,
> con qué criterios se acepta cada sección y cómo se mide el éxito.
> Si la implementación se desvía, se actualiza primero este archivo.

---

## 1. Problema

### 1.1 Enunciado

Dada una reseña turística escrita en español por un visitante de un destino mexicano,
predecir la **polaridad** que el autor asignó (1 a 5 estrellas) usando únicamente el texto.

Como tarea de contraste, sobre la misma entrada y con la misma arquitectura, predecir el
**tipo de establecimiento** reseñado (Hotel / Restaurant / Attractive).

### 1.2 Por qué este problema es interesante y no trivial

Cuatro propiedades del corpus hacen que este caso exija decisiones de modelado reales, y
no la aplicación mecánica de una receta:

1. **La variable objetivo es ordinal, no categórica.** Confundir 4★ con 5★ es un error
   leve; confundir 1★ con 5★ es un error grave. La cross-entropy trata ambos errores como
   idénticos. Esto justifica comparar formulaciones de pérdida y reportar MAE y QWK, no
   solo F1.
2. **El desbalance es extremo** (65.6% de la masa en 5★, 2.6% en 1★). El baseline de clase
   mayoritaria alcanza 65.6% de accuracy sin leer una sola palabra, lo que invalida el
   accuracy como métrica de progreso y obliga a macro-F1.
3. **El corpus está geográficamente sesgado** (41% Quintana Roo; Tulum e Isla Mujeres solos
   son el 36%). Un split aleatorio mide memorización del destino tanto como comprensión del
   sentimiento. Esto habilita un experimento de generalización que el split aleatorio oculta.
4. **El vocabulario es de dominio y regional** (*sargazo*, *alberca*, *mesero*, *palapa*,
   *camastro*, *all inclusive*), con anglicismos y errores ortográficos. Los embeddings
   preentrenados en español general se enfrentan a un problema de cobertura (OOV) medible.

### 1.3 Por qué la propuesta es adecuada

La tarea es clasificación de secuencias de longitud media (mediana ~350 caracteres) donde
el orden y la negación importan (*«no estuvo mal»* vs. *«estuvo mal»*). Una escalera de
cuatro representaciones —bolsa de palabras dispersa, embeddings aprendidos con memoria
recurrente, embeddings preentrenados, y atención contextual completa— permite aislar
**qué aporta cada nivel de modelado sobre esta tarea concreta**, que es exactamente la
transición conceptual que los notebooks 3 y 4 del curso plantean.

## 2. Datos

`vg055/Rest-Mex2025` en HuggingFace Hub. Detalle completo en [`DATASET.md`](DATASET.md).

Resumen: 208,051 reseñas, campos `Title`, `Review`, `Polarity` (1–5), `Town` (40),
`Region` (19), `Type` (3). Licencia CC-BY-4.0.

- **EDA:** sobre el corpus completo (208k). Es solo pandas, es barato y da estadísticas honestas.
- **Modelado:** submuestra **estratificada por polaridad de 40,000 reseñas**, semilla 42.
  Justificación en `DECISIONS.md` §D-004.

## 3. Protocolo experimental

### 3.1 Particiones

Dos esquemas, ambos sobre la misma submuestra de 40k:

- **Split A — aleatorio estratificado**, 80/10/10, estratificado por `Polarity`.
  Es el protocolo estándar y el que se usa para todas las comparaciones entre modelos.
- **Split B — geográfico**, regiones held-out. Las reseñas de un conjunto de regiones
  nunca vistas en entrenamiento forman el test. Se usa **solo** en la Extensión B, con el
  mejor modelo, para medir generalización a destinos nuevos.

Ambos splits se construyen con `SEED = 42` y se verifican imprimiendo la distribución de
clases resultante.

### 3.2 Métricas

| Métrica | Rol | Por qué |
|---|---|---|
| **macro-F1** | **Principal** | Da el mismo peso a las 5 clases; es la única que penaliza abandonar 1★ y 2★ |
| Accuracy | Contexto | Solo se reporta junto al baseline mayoritario (65.6%) para exponer su inutilidad |
| **MAE** | Ordinal | Mide la magnitud del error en estrellas; distingue equivocarse por 1 de equivocarse por 4 |
| **QWK** | Ordinal | Quadratic Weighted Kappa: acuerdo corregido por azar y ponderado por distancia |
| F1 por clase | Diagnóstico | Muestra exactamente dónde muere cada modelo (esperado: clases 2★ y 3★) |
| Tiempo de entrenamiento | Costo | Habilita el análisis costo/beneficio del cierre |

Todos los modelos reportan **las seis** sobre el mismo test de Split A.

### 3.3 Baselines de referencia obligatorios

Antes del primer modelo se calculan y se dejan fijos como línea de comparación:

- **Clase mayoritaria** (predecir siempre 5★): accuracy ≈ 0.656, macro-F1 ≈ 0.158.
- **Azar estratificado**: muestrea según la distribución empírica.

Ningún modelo se declara «bueno» sin superar ambos en macro-F1.

## 4. Estructura del notebook

Archivo único: `notebooks/miniproyecto1-restmex.ipynb`

### Sección 0 — Portada
Título, autores, contexto del curso, enunciado del problema en un párrafo, mapa de
navegación del notebook, y declaración explícita de qué se hereda de los notebooks guía
y qué es aporte propio.

### Sección 1 — Entorno y reproducibilidad
Detección Colab/local, instalación de dependencias, fijación de `SEED = 42` en todas las
librerías, detección de dispositivo, y una **tabla de configuración** que ajusta tamaños y
épocas según haya GPU o no. Imprime versiones de librerías para trazabilidad.

**Aceptación:** corre idéntico en Colab-GPU, Colab-CPU y local; imprime la configuración activa.

### Sección 2 — Carga del corpus
Descarga desde HuggingFace, conversión a `pandas`, inspección inicial (`shape`, `dtypes`,
`head`), y presentación de 3 reseñas completas de distinta polaridad para que el lector
vea el material crudo.

### Sección 3 — Análisis exploratorio (EDA)

Sobre el corpus completo. Cada subsección produce al menos una gráfica y su lectura.

| # | Contenido | Salida esperada |
|---|---|---|
| 3.1 | Calidad: nulos, duplicados exactos, reseñas vacías o degeneradas | Tabla de calidad + decisión de limpieza |
| 3.2 | Distribución de `Polarity` | Barras + baseline mayoritario calculado explícitamente |
| 3.3 | Distribución de `Type`, `Region`, `Town` | Barras; se hace visible la concentración en Quintana Roo |
| 3.4 | Longitudes en caracteres **y en tokens** | Histograma + percentiles; **de aquí sale `MAX_LEN` como P95, no un número arbitrario** |
| 3.5 | Léxico distintivo por polaridad | Log-odds ratio con prior de Dirichlet (no frecuencia cruda) + nube o barras |
| 3.6 | Interacciones: polaridad × tipo, polaridad × región, longitud × polaridad | Heatmap + boxplot |
| 3.7 | Cobertura de vocabulario | % de tokens del corpus presentes en los embeddings preentrenados; lista de OOV frecuentes |
| 3.8 | **Síntesis** | Lista numerada: cada hallazgo del EDA → la decisión de modelado que induce |

**Aceptación:** §3.8 conecta explícitamente cada hallazgo con una decisión posterior.
El EDA no es decorativo: es el que fija `MAX_LEN`, la métrica, y el diseño del Split B.

### Sección 4 — Preprocesamiento y tokenización
Tokenizador propio con normalización que **preserva la ñ y las tildes** (el del notebook
guía las conserva en la clase de caracteres pero conviene verificarlo y documentarlo),
manejo de URLs, números y puntuación. Construcción del vocabulario **por frecuencia con
`most_common`** (ver `DECISIONS.md` §D-002). Padding a `MAX_LEN` derivado del EDA.
Se muestra el efecto del tokenizador sobre una reseña real, ida y vuelta.

**Aceptación:** se reporta tamaño de vocabulario, tasa de `[UNK]` y tasa de truncamiento.

### Sección 5 — Modelo 1: TF-IDF + Regresión Logística
Baseline clásico y fuerte. n-gramas 1–2, `class_weight='balanced'`. Sirve de piso: si una
red neuronal no lo supera, la red no está aportando.

### Sección 6 — Modelo 2: LSTM desde cero
Réplica fiel de la arquitectura del notebook 3 (`Embedding → LSTM → Linear`) sobre nuestros
datos. Es el punto de comparación con el material del curso. Loop de entrenamiento explícito
con curvas de pérdida y macro-F1 por época.

### Sección 7 — Modelo 3: BiLSTM + embeddings preentrenados en español
Matriz de embeddings inicializada desde `es_core_news_lg` de spaCy (300d). Dos variantes:
**congelados** vs. **afinados**. Bidireccional, con `pack_padded_sequence` para que el
padding no contamine el estado oculto (mejora real sobre el notebook guía, que lee
`hidden[-1]` sobre secuencias con padding).

**Aceptación:** se reporta el % de vocabulario cubierto por los vectores y se comparan las
dos variantes.

### Sección 8 — Modelo 4: Transformer en español (BETO)
Fine-tuning de `dccuchile/bert-base-spanish-wwm-cased`. Se documenta el costo
computacional frente a la ganancia. Si no hay GPU, se degrada a extracción de features
congeladas + clasificador lineal, y se dice explícitamente en el notebook.

### Sección 9 — Extensión A: tratamiento ordinal de la polaridad
Compara sobre la mejor arquitectura recurrente:
- (a) cross-entropy estándar,
- (b) regresión + umbrales optimizados,
- (c) codificación ordinal acumulativa (*ordinal / coral*).

Se evalúa con MAE y QWK además de macro-F1, y se muestra cómo cambia la estructura de la
matriz de confusión (los errores deberían concentrarse cerca de la diagonal).

### Sección 10 — Extensión B: generalización geográfica
Entrena el mejor modelo con Split A y con Split B y compara la caída. Se analiza qué
regiones son más difíciles y se inspeccionan errores para ver si el modelo se apoyaba en
señales de destino (nombres de hoteles, topónimos) en lugar de señales de sentimiento.

**Hipótesis a contrastar:** el macro-F1 cae de forma apreciable bajo Split B; si no cae,
también es un hallazgo y se reporta como tal.

### Sección 11 — Extensión C: BiLSTM con atención e interpretabilidad
Capa de atención aditiva sobre las salidas de la BiLSTM. Se visualizan los pesos sobre
reseñas reales de cada polaridad (texto coloreado por peso). Se examina si el modelo atiende
a los términos que el EDA §3.5 identificó como distintivos — es decir, se cierra el círculo
entre exploración y modelo.

### Sección 12 — Extensión D: el espacio de embeddings
Retoma la parte de similitud del notebook 3 pero con vocabulario del dominio:
- vecinos más cercanos de términos propios (*sargazo*, *alberca*, *mesero*, *amabilidad*),
  comparando embeddings preentrenados vs. los aprendidos por la red en la tarea;
- proyección 2D (t-SNE o UMAP) coloreada por polaridad media del término;
- una aritmética vectorial propia del dominio, análoga al `king - man + woman` del guía.

**Aceptación:** debe mostrar que los embeddings *aprendidos en la tarea* organizan el
espacio por sentimiento, mientras los preentrenados lo organizan por tema.

### Sección 13 — Tarea de contraste: predecir `Type`
La mejor arquitectura, sin cambios, sobre `Type` (3 clases, balanceadas). Sirve para
demostrar que el macro-F1 modesto en polaridad **no es culpa del modelo ni del pipeline**,
sino de la naturaleza ordinal y del desbalance de la tarea.

### Sección 14 — Comparación global y análisis de errores
- Tabla única: modelo × (macro-F1, accuracy, MAE, QWK, tiempo, nº de parámetros).
- Gráfica de costo (tiempo/parámetros) contra beneficio (macro-F1).
- Matrices de confusión lado a lado.
- **Análisis cualitativo de errores:** ejemplos concretos mal clasificados, agrupados por
  causa (ironía, reseña mixta, texto muy corto, polaridad incoherente con el texto).
- Análisis de los casos donde *todos* los modelos fallan.

### Sección 15 — Conclusiones y limitaciones
Hallazgos numerados, respuesta explícita a la pregunta del problema, limitaciones honestas
(submuestra, un solo corpus, ruido de etiquetas del propio TripAdvisor) y trabajo futuro.

## 5. Criterios de aceptación del entregable

- [ ] «Restart & Run All» completo sin errores, en ≤ 30 min con GPU T4.
- [ ] Las cuatro técnicas principales entrenadas y comparadas en una tabla única.
- [ ] Las cuatro extensiones implementadas con su análisis.
- [ ] Ninguna celda de código sin markdown explicativo previo.
- [ ] Ninguna gráfica ni tabla sin su lectura escrita.
- [ ] Todo en español, incluidas las etiquetas de los ejes.
- [ ] Cero uso de los datasets de los notebooks guía.
- [ ] Conclusiones que respondan a la pregunta de §1.1 con evidencia del propio notebook.
- [ ] `EXPERIMENTS.md` con los resultados reales de la corrida final.
- [ ] Salidas de celdas guardadas en el `.ipynb` versionado.

## 6. Fuera de alcance

- Despliegue, API o demo interactiva.
- Entrenamiento sobre las 208k reseñas completas.
- Búsqueda exhaustiva de hiperparámetros (se hace exploración acotada y se documenta).
- Comparación formal contra los resultados publicados del shared task Rest-Mex 2025
  (protocolo y métrica oficiales difieren; se menciona el contexto, no se reclama paridad).
