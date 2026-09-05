# PLAN — Ejecución por fases

> Estado global: **Fase 0 completada.** Siguiente: Fase 1.
>
> Marca cada tarea al completarla. No avances de fase sin cerrar la anterior:
> el orden está pensado para que cada fase produzca los insumos de la siguiente.

Leyenda: `[ ]` pendiente · `[~]` en curso · `[x]` hecho

---

## Fase 0 — Scaffolding y especificación

Dejar el contrato escrito antes de escribir código.

- [x] Revisar `consigna.txt` y `rubrica.txt`, extraer restricciones duras
- [x] Revisar notebooks guía 3 y 4, identificar qué heredar y qué corregir
- [x] Seleccionar corpus y verificar que carga sin autenticación
- [x] Verificar estadísticas del corpus contra la API de HuggingFace
- [x] `CLAUDE.md`, `SPEC.md`, `DATASET.md`, `DECISIONS.md`, `PLAN.md`, `EXPERIMENTS.md`
- [x] `README.md`, `requirements.txt`, `.gitignore`
- [x] Estructura de carpetas
- [x] Esqueleto del notebook con todas las secciones del SPEC (114 celdas)
- [x] `git init`, primer commit, repositorio remoto
      (`Aguado4/miniproyecto-1-nlp-Aguado-Cruz-Ramirez`, privado)

**Salida:** especificación completa y notebook vacío pero estructurado.

---

## Fase 1 — Entorno y carga (SPEC §1–2)

- [ ] Celda de detección Colab/local + GPU/CPU con tabla de configuración
- [ ] Instalación de dependencias, una sola celda
- [ ] `set_seed(42)` sobre `random`, `numpy`, `torch`, `torch.cuda`
- [ ] Carga del corpus desde HuggingFace y conversión a pandas
- [ ] Inspección inicial y muestra de 3 reseñas de distinta polaridad

**Criterio de salida:** las cifras impresas coinciden con `DATASET.md` §3.
Si no coinciden, el dataset cambió: actualizar `DATASET.md` antes de seguir.

**Riesgo:** el corpus podría cambiar o desaparecer del Hub. Mitigación: fijar la revisión
del dataset si `datasets` lo permite, y dejar constancia de la fecha de descarga.

---

## Fase 2 — EDA (SPEC §3)

La fase más importante para la rúbrica y la que alimenta todas las demás.

- [ ] 3.1 Calidad: nulos, duplicados, reseñas degeneradas
- [ ] 3.2 Distribución de polaridad + cálculo explícito del baseline mayoritario
- [ ] 3.3 Distribución de `Type`, `Region`, `Town`
- [ ] 3.4 Longitudes en caracteres y tokens → **fija `MAX_LEN` (P95)**
- [ ] 3.5 Léxico distintivo por polaridad (log-odds con prior de Dirichlet)
- [ ] 3.6 Interacciones polaridad × tipo × región × longitud
- [ ] 3.7 Cobertura de vocabulario frente a los embeddings preentrenados
- [ ] 3.8 Síntesis: cada hallazgo → la decisión de modelado que induce

**Criterio de salida:** `MAX_LEN` fijado con evidencia; §3.8 escrita; toda gráfica leída.

---

## Fase 3 — Preprocesamiento y protocolo (SPEC §4, §3 protocolo)

- [ ] Tokenizador propio + verificación ida y vuelta sobre una reseña real
- [ ] Vocabulario por frecuencia (`most_common`), con reporte de tasa de `[UNK]`
- [ ] Submuestra estratificada de 40k, `SEED = 42`, con verificación de proporciones
- [ ] Split A (aleatorio estratificado 80/10/10)
- [ ] Split B (geográfico, regiones held-out) — **decidir y documentar qué regiones**
- [ ] Funciones de métricas (macro-F1, accuracy, MAE, QWK, F1 por clase)
- [ ] Baselines: clase mayoritaria y azar estratificado

**Criterio de salida:** una función `evaluar(y_true, y_pred)` que devuelve las seis métricas
y que todos los modelos usarán sin excepción. Registrar en `DECISIONS.md` el Split B.

---

## Fase 4 — Los cuatro modelos (SPEC §5–8)

Cada uno se cierra con sus métricas registradas en `EXPERIMENTS.md` antes de pasar al siguiente.

- [ ] Modelo 1 — TF-IDF + Regresión Logística
- [ ] Modelo 2 — LSTM desde cero (réplica del notebook 3, sobre nuestros datos)
- [ ] Modelo 3 — BiLSTM + embeddings de spaCy, variantes congelada y afinada
- [ ] Modelo 4 — BETO fine-tuned (con degradación a features congeladas si no hay GPU)

**Criterio de salida:** cuatro filas en la tabla de resultados, mismo test, mismas métricas.

**Riesgo:** BETO puede exceder el presupuesto de tiempo. Mitigación: `MAX_LEN` recortado
para el transformer, 2 épocas, y si aun así no cabe, degradar a features congeladas y
documentarlo como decisión, no como fallo.

---

## Fase 5 — Extensiones (SPEC §9–12)

- [ ] Extensión A — tratamiento ordinal (CE vs. regresión+umbrales vs. ordinal acumulativa)
- [ ] Extensión B — generalización geográfica (Split A vs. Split B)
- [ ] Extensión C — BiLSTM con atención + visualización de pesos
- [ ] Extensión D — espacio de embeddings (vecinos, proyección 2D, aritmética de dominio)

**Criterio de salida:** cada extensión responde una pregunta planteada explícitamente antes
de ejecutarla, y reporta el resultado aunque contradiga la hipótesis.

---

## Fase 6 — Contraste, comparación y cierre (SPEC §13–15)

- [ ] Tarea `Type` con la mejor arquitectura
- [ ] Tabla comparativa global
- [ ] Gráfica costo vs. beneficio
- [ ] Matrices de confusión comparadas
- [ ] Análisis cualitativo de errores por causa
- [ ] Casos donde todos los modelos fallan
- [ ] Conclusiones, limitaciones y trabajo futuro

---

## Fase 7 — Verificación y entrega

- [ ] «Restart & Run All» limpio, cronometrado
- [ ] Repasar checklist de aceptación de `SPEC.md` §5, punto por punto
- [ ] Repasar `rubrica.txt` criterio por criterio y anotar dónde se satisface cada uno
- [ ] `EXPERIMENTS.md` con los números de la corrida final
- [ ] `README.md` actualizado con los resultados principales
- [ ] Quitar `TODO`, celdas vacías y código muerto
- [ ] Notebook con salidas guardadas, commit y push

---

## Presupuesto de tiempo (GPU T4)

| Fase | Ejecución estimada |
|---|---|
| Carga + EDA | 3–4 min |
| Preprocesamiento | 1 min |
| Modelo 1 (TF-IDF) | < 1 min |
| Modelo 2 (LSTM) | 3 min |
| Modelo 3 (BiLSTM ×2 variantes) | 6 min |
| Modelo 4 (BETO) | 8–10 min |
| Extensiones A–D | 8–10 min |
| Tarea `Type` | 2 min |
| **Total** | **~30–35 min** |

Si en la Fase 7 el total supera 40 min, recortar: primero la variante congelada del Modelo 3,
después las épocas de la Extensión A. Nunca recortar el EDA ni el análisis de errores.
