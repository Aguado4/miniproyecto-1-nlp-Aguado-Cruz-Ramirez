# CLAUDE.md — Instrucciones de trabajo para este repositorio

Este archivo es la fuente de verdad operativa para cualquier sesión de agente que trabaje
en este proyecto. Léelo completo antes de tocar cualquier archivo.

---

## 1. Qué es este repositorio

Entregable del **Miniproyecto 1** del curso de NLP (Maestría, Universidad Icesi).
El producto final es **un único Jupyter Notebook** ejecutable de principio a fin que
implementa y compara técnicas de clasificación de texto en español sobre un corpus
propio (reseñas turísticas mexicanas, Rest-Mex 2025).

Los dos archivos que definen el éxito del trabajo están en la raíz y **no se modifican**:

- `consigna.txt` — qué pide el profesor.
- `rubrica.txt` — cómo se califica (7 puntos en 4 criterios).

Toda decisión de diseño debe poder justificarse contra uno de esos dos archivos.

## 2. Documentos de especificación (leer en este orden)

| Archivo | Contiene |
|---|---|
| `docs/SPEC.md` | Especificación completa del notebook: secciones, modelos, métricas, criterios de aceptación. **Es el contrato.** |
| `docs/PLAN.md` | Plan de ejecución por fases, con estado y orden de implementación. |
| `docs/DATASET.md` | Ficha del corpus: esquema, estadísticas verificadas, sesgos conocidos, licencia. |
| `docs/DECISIONS.md` | Registro de decisiones de diseño (ADR). Toda decisión no trivial se documenta aquí. |
| `docs/EXPERIMENTS.md` | Bitácora de resultados. Se llena a medida que se ejecutan los experimentos. |

**Flujo de trabajo:** `SPEC.md` define *qué*, `PLAN.md` define *en qué orden*,
`DECISIONS.md` registra *por qué*, `EXPERIMENTS.md` registra *qué salió*.
Si implementas algo que no está en el SPEC, primero actualiza el SPEC.

## 3. Reglas duras (violarlas cuesta puntos de rúbrica)

### 3.1 Reproducibilidad — vale 2 de 7 puntos

- **Toda celda debe correr desde cero, en orden, sin intervención manual.**
  El criterio de aceptación es «Restart & Run All» limpio.
- **Semilla global `SEED = 42`** aplicada a `random`, `numpy`, `torch` y a todo split.
  Cualquier submuestreo o partición usa esa semilla explícitamente.
- **Nada de rutas locales.** El corpus se descarga desde HuggingFace en la celda de carga.
  No se asume ningún archivo preexistente en disco.
- **El notebook debe detectar Colab vs. local y GPU vs. CPU**, y degradarse sin romperse
  (menos épocas / submuestra menor en CPU), nunca fallar.
- **Presupuesto de tiempo: ~30 min end-to-end en GPU T4.** Si una sección se pasa, se
  recorta la sección, no el rigor.
- Las salidas de las celdas se guardan en el `.ipynb` versionado. El profesor debe poder
  leer los resultados sin ejecutar nada.

### 3.2 Narrativa — vale 1 de 7 puntos

- **Todo en español.** Código, comentarios, markdown, gráficas, ejes, títulos.
- **Ninguna celda de código sin una celda markdown antes que explique el porqué**, no el qué.
  Mal: «Ahora entrenamos el modelo». Bien: «Entrenamos con macro-F1 como criterio de
  early stopping porque el desbalance 65/22/8/3/3 hace que la pérdida de validación baje
  aun cuando el modelo abandona las clases minoritarias».
- **Toda gráfica y toda tabla va seguida de su lectura.** Un número sin interpretación
  no cuenta como análisis.
- Cada sección de modelo cierra con un párrafo de hallazgos; el notebook cierra con
  conclusiones globales y limitaciones honestas.

### 3.3 Originalidad — vale 2 de 7 puntos

- **Prohibido reutilizar los datos de los notebooks guía** (`mteb/spanish_news`,
  `moviereviews.tsv`, `owlcreek.txt`, `reaganomics.txt`). La consigna lo prohíbe explícitamente.
- Se puede reutilizar la *estructura* pedagógica de los notebooks 3 y 4, pero cada
  componente heredado debe estar mejorado o cuestionado, no copiado.
- Los notebooks guía tienen defectos conocidos que **no se deben replicar**; ver
  `docs/DECISIONS.md` §D-002.

### 3.4 Comparación de técnicas — vale 2 de 7 puntos

La rúbrica exige **al menos tres técnicas** con comparación argumentada. El SPEC define
cuatro niveles más cuatro extensiones. Todas las comparaciones deben reportarse sobre
**el mismo split y las mismas métricas**, en una tabla única y consolidada.

## 4. Convenciones técnicas

- **El notebook es autocontenido.** No se importa nada de un `src/` del repo: en Colab
  eso obligaría a clonar y es un punto de fallo de reproducibilidad. Ver `DECISIONS.md` §D-003.
- Funciones auxiliares se definen en el propio notebook, en la sección donde se usan por
  primera vez.
- Métrica principal: **macro-F1**. Nunca reportar accuracy sin ponerlo al lado del
  baseline de clase mayoritaria (65.6%).
- Gráficas: `matplotlib` + `seaborn`, paleta consistente en todo el notebook, ejes
  siempre etiquetados en español, sin títulos redundantes con el markdown.
- Nombres de variables y funciones en español o inglés, pero **consistentes**; no mezclar
  `datos_entrenamiento` con `test_loader` en el mismo bloque.

## 5. Qué NO hacer

- No añadir dependencias fuera de `requirements.txt` sin actualizarlo y sin justificar
  el peso de la descarga (Colab reinstala todo en cada sesión).
- No dejar celdas con `!pip install` dispersas; toda la instalación va en la sección 1.
- No dejar `TODO`, celdas vacías, ni código comentado muerto en el entregable final.
- No inventar resultados en `EXPERIMENTS.md`: solo se registra lo que efectivamente se ejecutó.
- No hacer `git push` ni crear releases sin pedirlo explícitamente al usuario.

## 6. Comandos útiles

```bash
# Entorno local (Windows, Git Bash)
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt

# Verificar que el notebook corre limpio de principio a fin
jupyter nbconvert --to notebook --execute notebooks/miniproyecto1-restmex.ipynb \
  --output ejecutado.ipynb --ExecutePreprocessor.timeout=3600
```
