# EXPERIMENTS — Bitácora de resultados

> **Regla:** aquí solo se escriben números que efectivamente se ejecutaron.
> Nada de estimaciones ni de valores esperados. Si un experimento no se corrió,
> la fila se queda vacía.

Entorno de la corrida de referencia (Restart & Run All completo, sin errores):

| Campo | Valor |
|---|---|
| Fecha | 2026-09-09 |
| Plataforma | Local (no Colab) |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| Python / torch / sklearn | 3.14.4 / 2.14.0+cu130 / 1.9.0 |
| Semilla | 42 |
| Tamaño de submuestra | 40,000 (Split A: 32,000 / 4,000 / 4,000) |
| `MAX_LEN` | 150 tokens (percentil 95, EDA §3.4) |
| Tamaño de vocabulario | 30,000 (1.5% OOV y 4.2% truncamiento en test) |

---

## 1. Baselines — Split A, test

| Baseline | Accuracy | macro-F1 | MAE | QWK |
|---|---:|---:|---:|---:|
| Clase mayoritaria (siempre 5★) | 0.6565 | 0.1585 | 0.549 | 0.0000 |
| Azar estratificado | 0.4798 | 0.1931 | 0.834 | 0.0045 |

Valores teóricos esperados sobre la distribución completa: accuracy 0.656, macro-F1 ≈ 0.158.
Confirmado: la submuestra estratificada reproduce esas cifras casi exactas.

## 2. Modelos principales — tarea polaridad, Split A, test

Modelo 4 (BETO) excluido de esta entrega — ver `docs/DECISIONS.md` §D-009. En su lugar se
añade el Modelo 3c (BiLSTM + atención, Extensión C), que resultó ser el mejor modelo del
notebook.

| # | Modelo | macro-F1 | Accuracy | MAE | QWK | Params | Tiempo entren. (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | TF-IDF + LogReg | 0.5235 | 0.6785 | 0.376 | 0.7261 | 60,000 (features) | 27.0 |
| 2 | LSTM desde cero | 0.2951 | 0.5755 | 0.651 | 0.3552 | 3,972,741 | 50.1 |
| 3a | BiLSTM + spaCy (congelados) | 0.4530 | 0.6152 | 0.485 | 0.6525 | 9,441,605 | 39.0 |
| 3b | BiLSTM + spaCy (afinados) | 0.4863 | 0.6358 | 0.431 | 0.6942 | 9,441,605 | 38.9 |
| 3c | BiLSTM + atención (Ext. C, afinados) | **0.5274** | 0.6758 | 0.362 | 0.7546 | 9,441,862 | 88.4 |
| ~~4~~ | ~~BETO fine-tuned~~ | — | — | — | — | — | excluido (§D-009) |

### F1 por clase

| Modelo | 1★ | 2★ | 3★ | 4★ | 5★ |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LogReg | 0.596 | 0.301 | 0.433 | 0.475 | 0.813 |
| LSTM desde cero | 0.314 | 0.173 | 0.178 | 0.053 | 0.758 |
| BiLSTM + spaCy (congelados) | 0.517 | 0.221 | 0.352 | 0.400 | 0.775 |
| BiLSTM + spaCy (afinados) | 0.524 | 0.311 | 0.381 | 0.433 | 0.784 |

Hipótesis contrastada: las clases 2★ y 3★ son, en efecto, las peores en todos los modelos
(F1 tan bajo como 0.17-0.31), muy por debajo de 5★ (0.76-0.81) — confirmado.

Nota: el LSTM desde cero rindió notablemente peor en esta corrida local (macro-F1 0.295) que
en una corrida previa en Colab/T4 (0.44 con la misma configuración y semilla); ambas corridas
usan `SEED=42` pero difieren en versión de PyTorch/CUDA y hardware, lo que puede introducir
variación en la optimización de una red que ya mostraba señales de sobreajuste temprano
(pérdida de entrenamiento bajando mientras el macro-F1 de validación se estanca). Se reporta
el número de la corrida final versionada en el `.ipynb`, que es la que debe reproducirse.

## 3. Extensión A — tratamiento ordinal

Arquitectura fija: 3b · BiLSTM + spaCy (afinada), la mejor arquitectura recurrente por
macro-F1 antes de añadir atención.

| Formulación | macro-F1 | MAE | QWK | % errores a distancia ≥ 2 |
|---|---:|---:|---:|---:|
| (a) Cross-entropy (categórica) | 0.4863 | 0.431 | 0.6942 | 4.8% |
| (b) Regresión + umbrales optimizados | **0.4976** | 0.364 | 0.7385 | 3.1% |
| (c) Ordinal acumulativa (CORAL simplificado) | 0.4854 | **0.336** | **0.7534** | 3.2% |

Respuesta: sí mejora el MAE y el QWK de forma clara, sin sacrificar macro-F1 (de hecho (b) lo
mejora ligeramente). Los errores se concentran más cerca de la diagonal: el % de errores
graves (≥2 estrellas) cae casi a la mitad.

## 4. Extensión B — generalización geográfica

Regiones held-out del Split B (`docs/DECISIONS.md`, ya fijadas en el notebook §4.2):
Chiapas, Baja California Sur y Querétaro (7,456 reseñas de test, 18.5% de la submuestra).

| Modelo | macro-F1 Split A | macro-F1 Split B | Δ |
|---|---:|---:|---:|
| Mejor modelo (BiLSTM + spaCy afinada) | 0.4863 | 0.5291 | **+0.0429** |

| Región held-out | macro-F1 | n test |
|---|---:|---:|
| Chiapas | 0.5159 | 4,560 |
| Baja California Sur | 0.4947 | 1,937 |
| Querétaro | 0.5955 | 959 |

Respuesta: el rendimiento **no cae** ante destinos nunca vistos; sube ligeramente. La
inspección cualitativa de errores no mostró un patrón asociado a topónimos. Ver la lectura
completa en el notebook, Sección 10 (contradice la hipótesis inicial, y se reporta como tal).

## 5. Extensión C — atención

| Variante | macro-F1 | Δ vs. BiLSTM sin atención (3b) |
|---|---:|---:|
| BiLSTM + atención aditiva | **0.5274** | **+0.0411** |

Es el único modelo recurrente que supera a TF-IDF (0.5235) en este notebook. De las 20
palabras con mayor peso de atención (top-1 por reseña, muestra de 2,000 reseñas de test),
8 coinciden exactamente con el léxico distintivo de 5★ del EDA §3.5 (excelente, gran,
hermoso, mejor, buen, buena, agradable, increíble). Ninguna coincide con el léxico de 1★ en
este agregado global (artefacto del desbalance: 65% de las reseñas son 5★), pero la
visualización por polaridad individual sí muestra atención concentrada en negaciones y
adjetivos negativos para reseñas de 1★-2★.

## 6. Extensión D — espacio de embeddings

Cobertura del vocabulario del corpus en `es_core_news_lg`: 74.4% por tipos, 99.1% por tokens
(EDA §3.7).

Términos de dominio OOV más frecuentes: bacalar, sayulita, taxco, chichen, ajijic (todos
topónimos o nombres propios).

Vecinos más cercanos, preentrenados vs. aprendidos en la tarea:

| Término | Vecinos (preentrenados) | Vecinos (aprendidos) |
|---|---|---|
| sargazo | sargazos, alga, oleaje, manglar | sargazos, manglar, alga, oleaje |
| alberca | albercas, piscina, pileta, chapoteadero | albercas, piscina, pileta, jacuzzi |
| mesero | mesera, camarero, camarera, mozo | mesera, camarero, mozo, camarera |
| amabilidad | cordialidad, generosidad, hospitalidad | cordialidad, generosidad, hospitalidad |

Prácticamente idénticos antes y después de afinar: son sustantivos temáticos, no adjetivos de
sentimiento, y el afinado no tuvo motivo para moverlos.

Aritmética vectorial de dominio: `sucio - malo + excelente ≈ impecable` en **ambos** espacios
(similitud coseno 0.617 preentrenado, 0.623 aprendido).

## 7. Tarea de contraste — `Type`

| Modelo | macro-F1 | Accuracy |
|---|---:|---:|
| Mejor arquitectura, sin cambios (BiLSTM + spaCy afinada) | **0.9489** | 0.9510 |

F1 por clase: Attractive 0.959, Hotel 0.933, Restaurant 0.955 (n=4,000 test).

Comparado con el macro-F1 de 0.4863 de la misma arquitectura en polaridad: la diferencia
(+0.46) es el argumento central de que la dificultad está en la tarea (desbalance +
ordinalidad + ruido de etiqueta), no en el modelo ni el preprocesamiento.

## 8. Análisis de errores

Reseñas donde fallan los 5 modelos principales a la vez: **342 de 4,000 (8.6%)** del test de
Split A.

Causas observadas en los ejemplos inspeccionados (Sección 14 del notebook):

| Causa | Ejemplo observado |
|---|---|
| Reseña mixta (elogio + queja) | "el sazón malo" en una reseña de 3★ con más elogios que quejas |
| Texto ambiguo / tono no decantado | reseña sobre un museo "no está cerrado" mal clasificada en el Split B |
| Etiqueta incoherente con el texto | reseña de "viaje de graduación" con lenguaje muy entusiasta calificada 4★, no 5★ |

## 9. Notas de la corrida

- Corrida completa ("Restart & Run All") sin errores, notebook `.ipynb` versionado con todas
  las salidas.
- El Modelo 4 (BETO) no se ejecutó por indicación del profesor (§D-009), no por límite de
  cómputo ni de tiempo.
- El LSTM desde cero (Modelo 2) mostró variación notable frente a una corrida previa en
  Colab/T4 (ver nota en §2); no afecta las conclusiones del notebook porque en ambas corridas
  TF-IDF y la BiLSTM afinada superan claramente al LSTM desde cero.
