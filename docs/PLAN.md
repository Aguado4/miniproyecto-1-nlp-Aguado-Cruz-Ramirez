# PLAN — Ejecución por fases

> Estado global: **Fases 0-7 completadas.** Notebook con las 15 secciones del SPEC
> implementadas, ejecutadas de punta a punta sin errores y con resultados reales en
> `docs/EXPERIMENTS.md`. El Modelo 4 (BETO, Fase 4) se excluye por indicación del profesor
> — ver `docs/DECISIONS.md` §D-009.
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

- [x] Celda de detección Colab/local + GPU/CPU con tabla de configuración
- [x] Instalación de dependencias, una sola celda
- [x] `set_seed(42)` sobre `random`, `numpy`, `torch`, `torch.cuda`
- [x] Carga del corpus desde HuggingFace y conversión a pandas
- [x] Inspección inicial y muestra de 3 reseñas de distinta polaridad

**Criterio de salida:** las cifras impresas coinciden con `DATASET.md` §3.
Si no coinciden, el dataset cambió: actualizar `DATASET.md` antes de seguir.

**Riesgo:** el corpus podría cambiar o desaparecer del Hub. Mitigación: fijar la revisión
del dataset si `datasets` lo permite, y dejar constancia de la fecha de descarga.

---

## Fase 2 — EDA (SPEC §3)

La fase más importante para la rúbrica y la que alimenta todas las demás.

- [x] 3.1 Calidad: nulos, duplicados, reseñas degeneradas
- [x] 3.2 Distribución de polaridad + cálculo explícito del baseline mayoritario
- [x] 3.3 Distribución de `Type`, `Region`, `Town`
- [x] 3.4 Longitudes en caracteres y tokens → **fija `MAX_LEN` (P95)**
- [x] 3.5 Léxico distintivo por polaridad (log-odds con prior de Dirichlet)
- [x] 3.6 Interacciones polaridad × tipo × región × longitud
- [x] 3.7 Cobertura de vocabulario frente a los embeddings preentrenados
- [x] 3.8 Síntesis: cada hallazgo → la decisión de modelado que induce

**Criterio de salida:** `MAX_LEN` fijado con evidencia; §3.8 escrita; toda gráfica leída.

---

## Fase 3 — Preprocesamiento y protocolo (SPEC §4, §3 protocolo)

- [x] Tokenizador propio + verificación ida y vuelta sobre una reseña real
- [x] Vocabulario por frecuencia (`most_common`), con reporte de tasa de `[UNK]`
- [x] Submuestra estratificada de 40k, `SEED = 42`, con verificación de proporciones
- [x] Split A (aleatorio estratificado 80/10/10)
- [x] Split B (geográfico, regiones held-out) — Chiapas, Baja California Sur y Querétaro
- [x] Funciones de métricas (macro-F1, accuracy, MAE, QWK, F1 por clase)
- [x] Baselines: clase mayoritaria y azar estratificado

**Criterio de salida:** una función `evaluar(y_true, y_pred)` que devuelve las seis métricas
y que todos los modelos usarán sin excepción. Registrar en `DECISIONS.md` el Split B.

---

## Fase 4 — Los cuatro modelos (SPEC §5–8)

Cada uno se cierra con sus métricas registradas en `EXPERIMENTS.md` antes de pasar al siguiente.

- [x] Modelo 1 — TF-IDF + Regresión Logística
- [x] Modelo 2 — LSTM desde cero (réplica del notebook 3, sobre nuestros datos)
- [x] Modelo 3 — BiLSTM + embeddings de spaCy, variantes congelada y afinada
- [x] ~~Modelo 4 — BETO fine-tuned~~ **excluido por indicación del profesor** (`DECISIONS.md` §D-009)

**Criterio de salida:** tres filas en la tabla de resultados, mismo test, mismas métricas
(ya no cuatro: ver §D-009).

---

## Fase 5 — Extensiones (SPEC §9–12)

- [x] Extensión A — tratamiento ordinal (CE vs. regresión+umbrales vs. ordinal acumulativa)
- [x] Extensión B — generalización geográfica (Split A vs. Split B)
- [x] Extensión C — BiLSTM con atención + visualización de pesos
- [x] Extensión D — espacio de embeddings (vecinos, proyección 2D, aritmética de dominio)

**Criterio de salida:** cada extensión responde una pregunta planteada explícitamente antes
de ejecutarla, y reporta el resultado aunque contradiga la hipótesis.

---

## Fase 6 — Contraste, comparación y cierre (SPEC §13–15)

- [x] Tarea `Type` con la mejor arquitectura
- [x] Tabla comparativa global
- [x] Gráfica costo vs. beneficio
- [x] Matrices de confusión comparadas
- [x] Análisis cualitativo de errores por causa
- [x] Casos donde todos los modelos fallan
- [x] Conclusiones, limitaciones y trabajo futuro

---

## Fase 7 — Verificación y entrega

- [x] «Restart & Run All» limpio, cronometrado (local, GPU RTX 3050, sin errores)
- [x] Repasar checklist de aceptación de `SPEC.md` §5, punto por punto
- [x] Repasar `rubrica.txt` criterio por criterio y anotar dónde se satisface cada uno
- [x] `EXPERIMENTS.md` con los números de la corrida final
- [x] `README.md` actualizado con los resultados principales
- [x] Quitar `TODO`, celdas vacías y código muerto
- [ ] Notebook con salidas guardadas, commit y push — **commit hecho, push pendiente de
      confirmación del usuario**

---

## Presupuesto de tiempo (GPU T4)

| Fase | Ejecución estimada |
|---|---|
| Carga + EDA | 3–4 min |
| Preprocesamiento | 1 min |
| Modelo 1 (TF-IDF) | < 1 min |
| Modelo 2 (LSTM) | 3 min |
| Modelo 3 (BiLSTM ×2 variantes) | 6 min |
| ~~Modelo 4 (BETO)~~ | excluido — ver `DECISIONS.md` §D-009 |
| Extensiones A–D | 8–10 min |
| Tarea `Type` | 2 min |
| **Total** | **~30–35 min** |

Si en la Fase 7 el total supera 40 min, recortar: primero la variante congelada del Modelo 3,
después las épocas de la Extensión A. Nunca recortar el EDA ni el análisis de errores.
