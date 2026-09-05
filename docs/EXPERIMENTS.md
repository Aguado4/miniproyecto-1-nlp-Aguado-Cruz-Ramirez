# EXPERIMENTS — Bitácora de resultados

> **Regla:** aquí solo se escriben números que efectivamente se ejecutaron.
> Nada de estimaciones ni de valores esperados. Si un experimento no se corrió,
> la fila se queda vacía.

Entorno de la corrida de referencia: _(completar tras la primera corrida completa)_

| Campo | Valor |
|---|---|
| Fecha | |
| Plataforma | |
| GPU | |
| Semilla | 42 |
| Tamaño de submuestra | 40,000 |
| `MAX_LEN` | _(derivado del EDA §3.4)_ |
| Tamaño de vocabulario | |

---

## 1. Baselines — Split A, test

| Baseline | Accuracy | macro-F1 | MAE | QWK |
|---|---:|---:|---:|---:|
| Clase mayoritaria (siempre 5★) | | | | |
| Azar estratificado | | | | |

Valores teóricos esperados sobre la distribución completa: accuracy 0.656, macro-F1 ≈ 0.158.
Confirmar contra lo que salga en la submuestra.

## 2. Modelos principales — tarea polaridad, Split A, test

| # | Modelo | macro-F1 | Accuracy | MAE | QWK | Params | Tiempo entren. |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | TF-IDF + LogReg | | | | | — | |
| 2 | LSTM desde cero | | | | | | |
| 3a | BiLSTM + spaCy (congelados) | | | | | | |
| 3b | BiLSTM + spaCy (afinados) | | | | | | |
| 4 | BETO fine-tuned | | | | | | |

### F1 por clase

| Modelo | 1★ | 2★ | 3★ | 4★ | 5★ |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LogReg | | | | | |
| LSTM desde cero | | | | | |
| BiLSTM + spaCy (afinados) | | | | | |
| BETO | | | | | |

Hipótesis a contrastar: las clases 2★ y 3★ serán las peores en todos los modelos, por ser
minoritarias y por estar rodeadas de clases contiguas en una escala ordinal.

## 3. Extensión A — tratamiento ordinal

Arquitectura fija (la mejor recurrente), solo cambia la formulación de la salida.

| Formulación | macro-F1 | MAE | QWK | % errores a distancia ≥ 2 |
|---|---:|---:|---:|---:|
| Cross-entropy (categórica) | | | | |
| Regresión + umbrales optimizados | | | | |
| Ordinal acumulativa | | | | |

Pregunta: ¿mejora el MAE sin sacrificar macro-F1? ¿Se concentran los errores cerca de la diagonal?

## 4. Extensión B — generalización geográfica

Regiones held-out del Split B: _(documentar en `DECISIONS.md` al decidirlo)_

| Modelo | macro-F1 Split A | macro-F1 Split B | Δ |
|---|---:|---:|---:|
| Mejor modelo | | | |

| Región held-out | macro-F1 | n test |
|---|---:|---:|

Pregunta: ¿cae el rendimiento ante destinos nunca vistos? Si no cae, ¿qué dice eso sobre
qué está aprendiendo el modelo?

## 5. Extensión C — atención

| Variante | macro-F1 | Δ vs. BiLSTM sin atención |
|---|---:|---:|
| BiLSTM + atención aditiva | | |

Observaciones cualitativas sobre a qué atiende el modelo, y coincidencia con el léxico
distintivo identificado en el EDA §3.5:

## 6. Extensión D — espacio de embeddings

Cobertura del vocabulario del corpus en `es_core_news_lg`: _%_

Términos de dominio OOV más frecuentes:

Vecinos más cercanos, preentrenados vs. aprendidos en la tarea:

| Término | Vecinos (preentrenados) | Vecinos (aprendidos) |
|---|---|---|
| sargazo | | |
| alberca | | |
| mesero | | |
| amabilidad | | |

## 7. Tarea de contraste — `Type`

| Modelo | macro-F1 | Accuracy |
|---|---:|---:|
| Mejor arquitectura, sin cambios | | |

Comparar contra el macro-F1 de la misma arquitectura en polaridad. La diferencia es el
argumento de que la dificultad está en la tarea, no en el modelo.

## 8. Análisis de errores

Casos donde todos los modelos fallan, agrupados por causa:

| Causa | Frecuencia aprox. | Ejemplo |
|---|---:|---|
| Ironía / sarcasmo | | |
| Reseña mixta (elogio + queja) | | |
| Texto muy corto o poco informativo | | |
| Etiqueta incoherente con el texto | | |

## 9. Notas de la corrida

_(Incidencias, tiempos reales, desviaciones respecto al plan.)_
