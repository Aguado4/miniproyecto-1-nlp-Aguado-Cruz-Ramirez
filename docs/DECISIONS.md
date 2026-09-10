# DECISIONS — Registro de decisiones de diseño

Formato ADR abreviado. Cada entrada: contexto, decisión, alternativas descartadas y
consecuencias. Se añade una entrada cada vez que se toma una decisión no obvia.

Estados: `aceptada` · `propuesta` · `revisada` · `revertida`

---

## D-001 · Elección del corpus: Rest-Mex 2025

**Estado:** aceptada · 2026-09-05

**Contexto.** La consigna prohíbe explícitamente reutilizar los datos de los notebooks
guía y exige un EDA sustancial. Se evaluaron tres corpus en español verificando que
cargaran sin autenticación: `vg055/Rest-Mex2025` (turismo), `SINAI/EmoEvent` (emoción en
tweets) y `mteb/MassiveIntentClassification` (intenciones de asistente de voz).

**Decisión.** Rest-Mex 2025.

**Alternativas descartadas.**
- *EmoEvent ES*: buen ángulo (evaluación cross-event, texto ruidoso), pero solo ~8.4k
  ejemplos y metadatos pobres, lo que limita el EDA que la rúbrica premia.
- *MASSIVE es*: limpio y rápido, pero utterances de ~7 palabras. Con secuencias tan cortas
  una LSTM no tiene estructura temporal que explotar y la comparación entre las cuatro
  técnicas pierde sentido.

**Consecuencias.** Ganamos metadatos ricos (geografía, tipo), una variable objetivo ordinal
y desbalance real, que son la materia prima de tres de las cuatro extensiones. A cambio,
el corpus es grande y hay que submuestrear (ver D-004).

---

## D-002 · No replicar los defectos de los notebooks guía

**Estado:** aceptada · 2026-09-05

**Contexto.** El trabajo se basa en los notebooks 3 y 4 de la Sesión 1. Al revisarlos se
identificaron tres puntos que conviene no heredar tal cual.

**Decisión.** Corregir y **documentar la corrección en el notebook**, porque explicar por
qué algo estaba mal es evidencia de comprensión y suma en el criterio de Innovación.

1. **Vocabulario por orden de aparición.** El notebook 4 hace
   `list(token_counts.keys())[:50000-2]`, que sobre un `Counter` toma los tokens en orden
   de primera aparición, no por frecuencia. El notebook 3 sí usa `most_common`.
   → Usamos `most_common` y explicamos la diferencia.
2. **Padding contaminando el estado oculto.** Ambos notebooks leen `hidden[-1]` de una LSTM
   alimentada con secuencias rellenas con `[PAD]`, de modo que el último estado corresponde
   a los pasos de padding y no al final real del texto.
   → Usamos `pack_padded_sequence` en la BiLSTM y mostramos el efecto.
3. **Accuracy como única métrica.** Aceptable en un corpus balanceado; inservible aquí,
   donde predecir siempre 5★ da 0.656.
   → macro-F1 como métrica principal, con el baseline mayoritario siempre visible al lado.

**Consecuencias.** El notebook no es una copia con datos nuevos: cada componente heredado
está justificado o mejorado.

---

## D-003 · Notebook autocontenido, sin módulo `src/` importable

**Estado:** aceptada · 2026-09-05

**Contexto.** La estructura habitual de proyecto pondría las funciones en `src/` y las
importaría. Pero el entregable se ejecuta principalmente en Colab, donde importar del repo
obliga a clonarlo primero.

**Decisión.** Todo el código vive en el notebook. `results/` guarda salidas; no hay paquete
importable.

**Alternativas descartadas.** `src/` + `git clone` en la primera celda: añade un punto de
fallo (repo privado, red, rama) a un criterio que vale 2 puntos.

**Consecuencias.** El notebook es más largo, pero se abre en Colab y corre. La legibilidad
se sostiene con secciones bien delimitadas y funciones definidas donde se usan.

---

## D-004 · Submuestra estratificada de 40,000 para modelado

**Estado:** aceptada · 2026-09-05

**Contexto.** 208k reseñas y cuatro modelos más cuatro extensiones no caben en un notebook
reproducible en tiempo razonable. La reproducibilidad vale 2 de 7 puntos.

**Decisión.** EDA sobre el corpus completo; entrenamiento sobre 40,000 registros
estratificados por `Polarity` con `SEED = 42`.

**Alternativas descartadas.**
- *20k*: dejaría ~520 ejemplos en cada clase minoritaria; el macro-F1 se vuelve ruidoso y
  las conclusiones sobre 1★ y 2★ dejan de ser defendibles.
- *80k / completo*: mejores métricas, pero el notebook se va a una hora o más y arriesga el
  límite de sesión de Colab.

**Consecuencias.** Con 40k, las clases 1★ y 2★ quedan con ~1,040 ejemplos cada una: poco,
pero suficiente para que macro-F1 sea informativo. **Se preserva la distribución original**
en lugar de balancear por submuestreo, porque el desbalance es una propiedad del dominio
que el trabajo quiere estudiar, no un artefacto a eliminar.

---

## D-005 · `MAX_LEN` derivado del percentil 95, no fijado por costumbre

**Estado:** aceptada · 2026-09-05

**Contexto.** Los notebooks guía fijan 256 y 512 tokens sin justificación empírica
(el notebook 4 incluso dice «2048» en el texto y usa 512 en el código).

**Decisión.** El EDA §3.4 calcula la distribución de longitudes en tokens y `MAX_LEN` se
fija en el percentil 95 redondeado. Se reporta la tasa de truncamiento resultante.

**Consecuencias.** Menos cómputo desperdiciado en padding y una decisión defendible con
datos. Es un ejemplo concreto de EDA que alimenta el modelado, que es lo que pide el SPEC.

---

## D-006 · Métrica principal macro-F1, con MAE y QWK para lo ordinal

**Estado:** aceptada · 2026-09-05

**Contexto.** Objetivo ordinal y fuertemente desbalanceado.

**Decisión.** macro-F1 como métrica de selección y de early stopping. MAE y QWK se reportan
siempre al lado para capturar la magnitud del error en la escala de estrellas. El accuracy
solo aparece acompañado del baseline mayoritario.

**Consecuencias.** Los modelos se seleccionan por su desempeño en clases minoritarias, que
es lo que hace la tarea interesante. Es probable que esto baje el accuracy reportado frente
a optimizar cross-entropy pura; se explica en el notebook en lugar de esconderse.

---

## D-007 · Embeddings preentrenados desde spaCy `es_core_news_lg`

**Estado:** propuesta · 2026-09-05

**Contexto.** Se necesitan vectores de palabra en español para el Modelo 3. Opciones:
fastText `cc.es.300.vec` (~7 GB, inviable en Colab) o spaCy `es_core_news_lg` (300d,
~568 MB, instalable con pip).

**Decisión.** spaCy `es_core_news_lg`. Además conecta directamente con la sección de
similitud del notebook 3 del curso, que usa spaCy.

**Riesgo abierto.** Hay que verificar en el notebook que `es_core_news_lg` efectivamente
trae vectores densos y medir la cobertura del vocabulario del dominio. Si la cobertura
resulta muy baja, se documenta como hallazgo (el vocabulario turístico regional no está bien
representado en corpus de español general) — que también es un resultado válido y publicable
en el notebook.

---

## D-008 · Estructura del repositorio y nombre

**Estado:** aceptada · 2026-09-05

**Decisión.** Repositorio `miniproyecto-1-nlp-Aguado-Cruz-Ramirez`, privado por defecto.
`consigna.txt` y `rubrica.txt` se versionan para que el entregable sea autoexplicativo.
El notebook con salidas ejecutadas también se versiona, para que el profesor pueda leer
resultados sin ejecutar nada.

---

## D-009 · Exclusión del Modelo 4 (BETO) de esta entrega

**Estado:** aceptada · 2026-09-09

**Contexto.** El SPEC original (§2, Sección 8) contemplaba cuatro modelos, el último un
fine-tuning de `dccuchile/bert-base-spanish-wwm-cased`. El profesor indicó en clase que el
fine-tuning de un transformer no aplica para este miniproyecto — se trabajará en una entrega
posterior del curso, donde tendrá el tiempo y el contexto teórico que merece.

**Decisión.** No implementar la Sección 8. Se deja una nota explícita en el notebook
(en vez de una celda vacía o un `TODO`) explicando la exclusión y por qué no afecta la
rúbrica: el criterio "Implementación de técnicas" exige *al menos tres* técnicas comparadas,
y el notebook ya presenta tres representaciones genuinamente distintas — TF-IDF (bolsa de
palabras dispersa), LSTM desde cero (embeddings aprendidos con memoria recurrente) y BiLSTM
con embeddings de spaCy en sus variantes congelada y afinada (embeddings preentrenados con
lectura bidireccional). Las Extensiones A-D y la tarea de contraste `Type` (Secciones 9-13)
aportan además el diferencial de Innovación que el SPEC había repartido entre el Modelo 4 y
las extensiones.

**Alternativas descartadas.** Implementar una versión reducida de BETO (features congeladas
sin fine-tuning) solo para tener una cuarta fila en la tabla comparativa: se descarta porque
contradice la indicación explícita del profesor, no porque falte tiempo o cómputo.

**Consecuencias.** `requirements.txt` ya no necesita `transformers` ni `accelerate`. La
"escalera de representaciones" del notebook queda en tres escalones en lugar de cuatro; se
compensa con la profundidad de las extensiones, que es donde el SPEC concentraba el valor
de Innovación de todos modos.

---

## Plantilla para nuevas entradas

```markdown
## D-0NN · Título breve

**Estado:** propuesta · AAAA-MM-DD

**Contexto.** Qué situación obliga a decidir.

**Decisión.** Qué se hace.

**Alternativas descartadas.** Qué más se consideró y por qué no.

**Consecuencias.** Qué se gana, qué se paga, qué queda abierto.
```
