# Documentación Parcial del Proyecto

## Proyecto Integrador — Segundo Parcial

| Campo | Detalle |
|-------|---------|
| **Universidad** | Universidad Tecnológica de Panamá (UTP) |
| **Facultad** | Ingeniería de Sistemas Computacionales |
| **Curso** | Gestión de la Información — I Semestre 2026 |
| **Tema (Grupo 4)** | Análisis del Mercado Laboral IT en Panamá |
| **Estudiante** | Angel Martínez |
| **Cédula** | 8-893-602 |
| **Fecha** | Junio 2026 |

---

## 1. Introducción

El sector de Tecnologías de la Información en Panamá experimenta un crecimiento
sostenido, impulsado por la posición del país como hub regional de servicios y
por la transformación digital de la banca, la logística y el comercio. Sin
embargo, tanto los profesionales como las instituciones educativas carecen de
herramientas analíticas que les permitan responder, con base en datos, preguntas
estratégicas como: ¿cuáles son las tecnologías más demandadas?, ¿qué habilidades
están emergiendo?, ¿cuánto debería ganar un perfil determinado según su nivel,
modalidad y stack tecnológico?

Este proyecto, correspondiente al **Segundo Parcial** del curso, integra un
**pipeline de datos**, **técnicas de Machine Learning** y un **dashboard
interactivo en Streamlit** para responder esas preguntas y constituir la base del
proyecto final.

## 2. Definición de la problemática

> *¿Cómo se comporta el mercado laboral IT panameño en términos de demanda de
> habilidades, salarios y tendencias tecnológicas, y cómo podemos predecir el
> salario de un perfil y anticipar las habilidades emergentes?*

Objetivos específicos:

1. Construir un pipeline reproducible que integre al menos dos fuentes de datos.
2. Limpiar y transformar los datos para dejarlos listos para análisis.
3. Aplicar técnicas de ML para extraer conocimiento (predicción y segmentación).
4. Presentar los hallazgos en un dashboard interactivo para la toma de decisiones.

## 3. Fuentes de datos

Se integran **dos fuentes de datos diferentes**:

### Fuente 1 — Ofertas de empleo IT (CSV)
Representa el resultado del *web scraping* de portales de empleo panameños
(Konzerta, encuentra24, LinkedIn, Computrabajo). Campos principales:
`id_oferta, titulo, empresa, fuente, fecha_publicacion, provincia, modalidad,
nivel, tipo_contrato, salario_min, salario_max, habilidades, descripcion`.

### Fuente 2 — Catálogo de tecnologías (JSON)
Catálogo de referencia que clasifica cada tecnología por `categoria`
(Lenguaje, Frontend, Backend, Cloud, DevOps, Datos, IA, Base de Datos), su
`indice_demanda` (0–100) y un indicador booleano `emergente`. Se utiliza para
**enriquecer** las ofertas, normalizar los nombres de habilidades y calcular
métricas derivadas.

> **Sobre la reproducibilidad:** dado que el scraping en vivo depende de la
> disponibilidad de los portales y de sus términos de uso, las fuentes crudas se
> generan de manera **sintética pero calibrada** con proporciones realistas del
> mercado panameño y una semilla fija (`42`). La etapa de ingesta del pipeline
> está desacoplada: sustituir el generador por un scraper real no requiere
> cambios en el resto del flujo.

## 4. Pipeline de datos

El pipeline (`src/pipeline.py`) sigue cuatro etapas claramente documentadas:

1. **Ingesta:** lectura del CSV de ofertas y del JSON de tecnologías.
2. **Limpieza:**
   - eliminación de duplicados (avisos re-publicados),
   - conversión de tipos (fechas con `to_datetime`),
   - manejo de valores faltantes (empresa → "Confidencial"),
   - normalización de categóricas (`title case`, *strip*).
3. **Transformación / Feature engineering:**
   - parseo e imputación de salarios (`salario_min`, `salario_max`,
     `salario_promedio`; imputación por mediana del nivel),
   - explosión de la lista de habilidades y cruce con el catálogo,
   - variables derivadas: `num_habilidades`, `num_habilidades_emergentes`,
     `indice_demanda_prom`, `categoria_principal`, `tiene_emergente`,
   - variables temporales (`anio`, `mes`, `anio_mes`),
   - filtrado de *outliers* salariales (percentiles 0.5 % y 99.5 %).
4. **Carga:** persistencia de `ofertas_procesadas.csv` y `habilidades_largo.csv`
   en `data/processed/`.

El resultado de referencia procesa **1,286 ofertas** y genera una tabla larga de
**~6,000 pares oferta–habilidad**.

## 5. Análisis con Machine Learning

Aunque el parcial exige **al menos una** técnica, se implementaron **tres** más
un análisis de tendencias:

### A. Regresión — Predicción de salario
- **Modelo:** `RandomForestRegressor` (250 árboles, `max_depth=14`).
- **Baseline:** `LinearRegression`.
- **Variables:** nivel, modalidad, provincia, categoría, tipo de contrato,
  número de habilidades, habilidades emergentes e índice de demanda.
- **Resultados:** R² = **0.838**, MAE ≈ **$279**. El baseline lineal alcanza
  R² = 0.868, lo que confirma una fuerte relación lineal entre las variables y
  el salario; el Random Forest captura además interacciones no lineales.

### B. Clasificación — Nivel del puesto
- **Modelo:** `RandomForestClassifier` (300 árboles, `max_depth=16`).
- **Clases:** Junior, Semi Senior, Senior, Lead.
- **Resultados:** Accuracy = **0.733**, F1-macro = **0.643**. La confusión se
  concentra entre niveles contiguos (p. ej. Semi Senior vs. Senior), lo que es
  esperable porque sus rangos salariales y de habilidades se solapan.

### C. Clustering — Segmentación de perfiles (KMeans, k = 4)
Sobre salario, número de habilidades, habilidades emergentes e índice de
demanda. Segmentos interpretados:

| Segmento | Salario medio | Habilidades | Emergentes | N° ofertas |
|----------|---------------|-------------|------------|------------|
| Entrada / Bajo costo | menor | pocas | bajas | — |
| Generalista intermedio | medio | medias | bajas | — |
| Especialista demandado | medio-alto | muchas | altas | — |
| Alta especialización / Premium | mayor | medias-altas | medias | — |

*(Los valores exactos por segmento se generan en `models/metricas.json` y se
visualizan en el dashboard.)*

### D. Predicción de habilidades emergentes
Mediante **regresión lineal** sobre la **cuota mensual** de cada tecnología
emergente (porcentaje de ofertas del mes que la solicitan), excluyendo el último
mes parcial. Las de mayor crecimiento proyectado fueron **TypeScript**
(+0.94 pp/mes), **Vue** (+0.66) y **Go** (+0.46), señalando hacia dónde conviene
orientar la capacitación.

## 6. Visualización — Dashboard (Streamlit)

El dashboard (`app/dashboard.py`) ofrece:

- **KPIs** dinámicos (ofertas, salario promedio, % remoto/híbrido, tecnologías).
- **Filtros** por provincia, modalidad, nivel, tecnología y rango salarial.
- **Pestaña Tendencias:** evolución de ofertas y salarios, modalidad, salario por nivel.
- **Pestaña Habilidades:** top tecnologías, demanda por categoría, salario por
  tecnología y predicción de habilidades emergentes.
- **Pestaña Modelos ML:** métricas de los tres modelos y mapa de clusters.
- **Pestaña Predictor de salario:** formulario interactivo que invoca el modelo
  de regresión entrenado.
- **Pestaña Datos:** tabla filtrable con exportación a CSV.

## 7. Conclusiones parciales

1. Es factible construir un pipeline **reproducible y desacoplado** que integre
   múltiples fuentes y deje los datos listos para análisis.
2. El salario es **altamente predecible** (R² ≈ 0.84) a partir del nivel, el
   stack tecnológico y la modalidad, siendo la presencia de habilidades de
   **Cloud, DevOps e IA** el principal motor del salario.
3. El mercado se segmenta en perfiles claramente diferenciados, útiles para
   orientar tanto a candidatos como a empresas.
4. Las tecnologías emergentes (**TypeScript, Vue, Go, Kubernetes, IA/ML**)
   muestran una tendencia de adopción creciente, información valiosa para la
   planificación curricular y profesional.

## 8. Trabajo futuro (hacia el proyecto final)

- Conectar la ingesta a *scraping* real con actualización periódica.
- Incorporar extracción de habilidades y salarios mediante **LLM** sobre el
  texto libre de las descripciones.
- Análisis de series temporales más robusto (ARIMA / Prophet) para la
  proyección de demanda.

## 9. Referencias

- Instituto Nacional de Estadística y Censo (INEC). (2025). *Estadísticas del
  mercado laboral de Panamá.* Contraloría General de la República.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
  *Journal of Machine Learning Research, 12*, 2825–2830.
- McKinney, W. (2010). Data Structures for Statistical Computing in Python.
  *Proceedings of the 9th Python in Science Conference.*
- Streamlit Inc. (2026). *Streamlit Documentation.* https://docs.streamlit.io

---
*Documento elaborado por Angel Martínez (cédula 8-893-602) para el Segundo
Parcial del curso Gestión de la Información, UTP, I Semestre 2026.*
