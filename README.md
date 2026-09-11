# Miniproyecto 1 - NLP

**Clasificación de polaridad en reseñas turísticas en español: de la bolsa de palabras a los transformers**

Maestría · Universidad Icesi · Curso de Procesamiento de Lenguaje Natural

Autores: Juan José Aguado · Juan David Cruz · Juan Diego Ramírez

---

## El problema

Dada una reseña turística escrita en español sobre un destino mexicano, predecir cuántas
estrellas (1 a 5) le puso su autor, usando solo el texto.

Parece un problema resuelto de análisis de sentimiento. No lo es, y por cuatro razones que
salen de los propios datos:

1. **La etiqueta es ordinal, no categórica.** Confundir 4★ con 5★ es un desliz; confundir
   1★ con 5★ es un desastre. La cross-entropy, que es lo que usan por defecto los notebooks
   del curso, considera ambos errores exactamente iguales.
2. **El desbalance es brutal.** El 65.6% de las reseñas son 5★. Un modelo que no lea nada y
   responda siempre «5» acierta el 65.6% de las veces. Eso convierte el *accuracy* en una
   métrica que mide sobre todo la pereza.
3. **El corpus está concentrado geográficamente.** Quintana Roo aporta el 41% de las reseñas;
   solo Tulum e Isla Mujeres son el 36%. Un modelo puede aprender el destino en lugar del
   sentimiento, y una partición aleatoria jamás lo delataría.
4. **El vocabulario es regional y de dominio.** *Sargazo*, *alberca*, *mesero*, *palapa*,
   *camastro*. Los embeddings preentrenados en español general tienen un problema de
   cobertura aquí, y se puede medir.

## La propuesta

Una escalera de cuatro representaciones del texto, evaluadas sobre exactamente el mismo
split y las mismas métricas, para aislar qué aporta cada salto:

| # | Técnica | Qué representa |
|---|---|---|
| 1 | TF-IDF + Regresión Logística | Bolsa de palabras dispersa, sin orden |
| 2 | LSTM desde cero | Embeddings aprendidos + memoria secuencial |
| 3 | BiLSTM + embeddings de spaCy | Conocimiento léxico preentrenado, contexto bidireccional |
| 4 | BETO (BERT en español) | Representaciones contextuales completas |

Sobre esa base, cuatro extensiones propias que van más allá del ejemplo del curso:

- **Tratamiento ordinal** - cross-entropy vs. regresión con umbrales vs. codificación
  ordinal acumulativa, evaluadas con MAE y QWK además de F1.
- **Generalización geográfica** - entrenar sin ver ciertas regiones y medir cuánto cae el
  modelo ante destinos nuevos.
- **Interpretabilidad con atención** - visualizar qué palabras pesan en cada predicción y
  contrastarlas con el léxico que el EDA identificó como distintivo.
- **Anatomía del espacio de embeddings** - vecinos de términos del dominio y proyección 2D,
  comparando vectores preentrenados contra los aprendidos en la tarea.

Y una **tarea de control**: la misma arquitectura, sin cambio alguno, prediciendo el tipo de
establecimiento (Hotel / Restaurant / Attractive), que sí está balanceado. Sirve para
demostrar que un macro-F1 modesto en polaridad no es culpa del pipeline sino de la naturaleza
de la tarea.

## El corpus

[`vg055/Rest-Mex2025`](https://huggingface.co/datasets/vg055/Rest-Mex2025) - 208,051 reseñas
turísticas en español, del shared task Rest-Mex 2025 (IberLEF). CC-BY-4.0.
Campos: título, reseña, polaridad (1–5), pueblo (40), región (19) y tipo (3).

El EDA se hace sobre el corpus completo; el modelado sobre una submuestra estratificada de
40,000 reseñas, para que el notebook sea reproducible de principio a fin en tiempo razonable.

Ficha completa, estadísticas verificadas y sesgos conocidos: [`docs/DATASET.md`](docs/DATASET.md).

## Resultados

Corrida de referencia completa (`Restart & Run All`, sin errores). Detalle completo con
todas las métricas, extensiones y análisis de errores en
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

| Modelo | macro-F1 | Accuracy | MAE | QWK |
|---|---:|---:|---:|---:|
| Baseline (clase mayoritaria) | 0.159 | 0.657 | 0.549 | 0.000 |
| TF-IDF + Regresión Logística | 0.524 | 0.679 | 0.376 | 0.726 |
| LSTM desde cero | 0.295 | 0.576 | 0.651 | 0.355 |
| BiLSTM + spaCy (afinada) | 0.486 | 0.636 | 0.431 | 0.694 |
| **BiLSTM + atención (Extensión C)** | **0.527** | 0.676 | 0.362 | 0.755 |

El Modelo 4 (BETO) se excluyó de esta entrega por indicación del profesor
([`docs/DECISIONS.md`](docs/DECISIONS.md) §D-009); en su lugar, la Extensión C (atención)
es la que termina superando al baseline clásico de TF-IDF.

Tres hallazgos centrales, desarrollados en la Sección 15 del notebook:

1. La misma arquitectura pasa de macro-F1 0.486 en polaridad a **0.949** prediciendo el tipo
   de establecimiento (`Type`, balanceado, no ordinal): la dificultad está en la tarea, no en
   el modelo.
2. Tratar la polaridad como ordinal (regresión con umbrales o codificación acumulativa) no
   sube mucho el macro-F1, pero corta a la mitad los errores graves (a ≥2 estrellas de
   distancia).
3. El modelo generaliza igual de bien a regiones de México nunca vistas en entrenamiento
   (Chiapas, Baja California Sur, Querétaro) - contra la hipótesis inicial de que dependería
   de memorizar destinos.

## Cómo ejecutarlo

### Google Colab (recomendado)

Abrir `notebooks/miniproyecto1-restmex.ipynb` en Colab, seleccionar entorno de ejecución con
GPU T4, y ejecutar todo. El notebook detecta el entorno e instala sus dependencias solo.
Tiempo estimado de principio a fin: ~30 minutos.

### Local

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
python -m spacy download es_core_news_lg

jupyter lab notebooks/miniproyecto1-restmex.ipynb
```

Sin GPU el notebook funciona igual: reduce automáticamente el número de épocas y sustituye
el fine-tuning del transformer por extracción de features congeladas. Lo indica de forma
explícita en su salida.

## Estructura del repositorio

```
.
├── CLAUDE.md              Instrucciones de trabajo para agentes
├── README.md              Este archivo
├── consigna.txt           Enunciado del profesor (no modificar)
├── rubrica.txt            Criterios de calificación (no modificar)
├── requirements.txt
├── docs/
│   ├── SPEC.md            Especificación del notebook: el contrato
│   ├── PLAN.md            Plan de ejecución por fases, con estado
│   ├── DATASET.md         Ficha del corpus y estadísticas verificadas
│   ├── DECISIONS.md       Registro de decisiones de diseño (ADR)
│   └── EXPERIMENTS.md     Bitácora de resultados
├── notebooks/
│   └── miniproyecto1-restmex.ipynb
└── results/
    ├── figures/
    └── metrics/
```

## Cómo está organizado el trabajo

El proyecto sigue un flujo *spec-driven*: la especificación se escribió y se acordó **antes**
de la primera línea de código, y las estadísticas del corpus se verificaron contra la API de
HuggingFace antes de diseñar los experimentos. `docs/SPEC.md` define qué se construye,
`docs/PLAN.md` en qué orden, `docs/DECISIONS.md` por qué se decidió así, y
`docs/EXPERIMENTS.md` qué resultó.

## Relación con los notebooks guía

Este trabajo parte de los notebooks 3 y 4 de la Sesión 1 del curso, pero con datos propios,
como exige la consigna, y corrigiendo tres puntos de los originales que se documentan y se
explican dentro del notebook: la construcción del vocabulario por orden de aparición en lugar
de por frecuencia, la lectura del último estado oculto de la LSTM sobre secuencias con padding,
y el uso del accuracy como única métrica en un problema fuertemente desbalanceado.
Detalle en [`docs/DECISIONS.md`](docs/DECISIONS.md) §D-002.
