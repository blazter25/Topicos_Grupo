# Análisis del Mercado Laboral IT en Panamá

**Proyecto Integrador — Segundo Parcial**
**Curso:** Gestión de la Información · Facultad de Ingeniería de Sistemas Computacionales · UTP · I Semestre 2026
**Tema:** Grupo 4 — Análisis del Mercado Laboral IT en Panamá

| | |
|---|---|
| **Estudiante** | Angel Martínez Anllelina Varcacia  David Ortega  Carlos Jaen  Gabriel Rodriguez |
| **Cédula** 

---

## 1. Problemática

El mercado laboral de tecnologías de información (IT) en Panamá evoluciona muy
rápido: aparecen nuevas tecnologías, los salarios varían por nivel, modalidad y
provincia, y resulta difícil para estudiantes y profesionales saber **qué
habilidades aprender** y **cuánto vale su perfil**. Este proyecto integra un
pipeline de datos, técnicas de Machine Learning y un dashboard interactivo para
responder esas preguntas con datos.

## 2. Fuentes de datos (mínimo 2 requeridas)

| # | Fuente | Formato | Descripción |
|---|--------|---------|-------------|
| 1 | `ofertas_empleo_it.csv` | CSV | Ofertas de empleo IT (título, empresa, salario, modalidad, nivel, habilidades, fecha). Simula el resultado de *web scraping* de portales (Konzerta, encuentra24, LinkedIn, Computrabajo). |
| 2 | `catalogo_tecnologias.json` | JSON | Catálogo de tecnologías con categoría, índice de demanda y marca de "emergente". Se usa para **enriquecer** y normalizar las habilidades de las ofertas. |

> **Nota de reproducibilidad:** para que el proyecto funcione sin conexión y de
> forma reproducible (semilla fija), las fuentes crudas se generan con
> `src/generar_datos.py` usando proporciones realistas del mercado panameño. La
> arquitectura del pipeline es idéntica a la que se usaría con datos scrapeados
> reales: solo cambiaría la etapa de ingesta.

## 3. Arquitectura del pipeline

```
generar_datos.py            pipeline.py                         modelos_ml.py
─────────────────   ──────────────────────────────   ────────────────────────
Fuente 1 (CSV) ─┐   1. INGESTA   (CSV + JSON)         A. Regresión (salario)
                ├─> 2. LIMPIEZA  (dups, tipos, NaN)   B. Clasificación (nivel)
Fuente 2 (JSON)─┘   3. TRANSFORM. (salarios, skills,  C. Clustering (perfiles)
                       feature engineering)           + Proyección emergentes
                    4. CARGA  -> data/processed/      -> models/*.pkl + métricas
                                                              │
                                                              ▼
                                              app/dashboard.py (Streamlit)
```

## 4. Técnicas de Machine Learning aplicadas

El parcial exige **al menos 1** técnica; este proyecto implementa **tres**:

- **A. Regresión** (`RandomForestRegressor`, baseline `LinearRegression`):
  predice el **salario promedio** de una oferta. Métrica: R² y MAE.
- **B. Clasificación** (`RandomForestClassifier`): predice el **nivel** del
  puesto (Junior / Semi Senior / Senior / Lead). Métrica: Accuracy y F1-macro.
- **C. Clustering** (`KMeans`, k=4): segmenta los perfiles del mercado en
  grupos interpretables (entrada, generalista, especialista, premium).
- **Análisis de tendencias:** proyección lineal de la frecuencia mensual de las
  habilidades **emergentes** para anticipar demanda futura.

## 5. Instalación y ejecución

```bash
# 1. (Opcional) crear entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el pipeline completo (genera datos + ETL + entrena modelos)
python main.py

# 4. Levantar el dashboard interactivo
streamlit run app/dashboard.py
```

El dashboard se abre en `http://localhost:8501`.

## 6. Estructura del repositorio

```
parcial2-mercado-it/
├── main.py                     # Orquestador end-to-end
├── requirements.txt
├── README.md
├── src/
│   ├── generar_datos.py        # Genera las 2 fuentes crudas
│   ├── pipeline.py             # ETL: ingesta, limpieza, transformación, carga
│   └── modelos_ml.py           # Regresión, clasificación, clustering, tendencias
├── app/
│   └── dashboard.py            # Dashboard Streamlit (5 pestañas)
├── data/
│   ├── raw/                    # Fuentes crudas (CSV + JSON)
│   └── processed/              # Datasets procesados (se regeneran)
├── models/                     # Modelos .pkl + metricas.json
└── docs/
    └── documentacion_parcial.md
```

## 7. Funcionalidades del dashboard

- **KPIs:** total de ofertas, salario promedio, % remoto/híbrido, tecnologías.
- **Filtros** (sidebar): provincia, modalidad, nivel, tecnología y rango salarial.
- **Tendencias:** evolución temporal de ofertas y salarios; modalidad; salario por nivel.
- **Habilidades:** top tecnologías, demanda por categoría, salario por tecnología y **predicción de habilidades emergentes**.
- **Modelos ML:** métricas de los 3 modelos y visualización de clusters.
- **Predictor de salario:** formulario interactivo que usa el modelo de regresión.
- **Datos:** tabla filtrable con descarga a CSV.

## 8. Resultados obtenidos (ejecución de referencia)

| Modelo | Métrica | Valor |
|--------|---------|-------|
| Regresión salario (RF) | R² | 0.838 |
| Regresión salario (RF) | MAE | $279 |
| Clasificación de nivel | Accuracy | 0.733 |
| Clasificación de nivel | F1-macro | 0.643 |
| Clustering | k | 4 segmentos |

> Las cifras corresponden a la ejecución de referencia con la semilla fija (42)
> y pueden variar levemente si se modifican los parámetros de generación.

---
*Proyecto desarrollado para el Segundo Parcial del curso Gestión de la Información, UTP.*
