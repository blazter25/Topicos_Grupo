"""
Modelos de Machine Learning.
====================================================================
Proyecto: Análisis del Mercado Laboral IT en Panamá
Curso:    Gestión de la Información - UTP, I Semestre 2026
Autor:    Angel Martínez  -  Cédula: 8-893-602

Se aplican TRES técnicas de ML (el parcial exige al menos 1):
  A) REGRESIÓN     : predecir el salario promedio de una oferta.
  B) CLASIFICACIÓN : predecir el nivel del puesto (Junior..Lead).
  C) CLUSTERING    : segmentar los perfiles del mercado (KMeans).

Adicionalmente, un análisis de tendencias proyecta la demanda futura de
las habilidades emergentes (regresión lineal temporal).

Los modelos entrenados se serializan en /models para que el dashboard
los consuma sin reentrenar.
====================================================================
"""

from __future__ import annotations

import json               # Para guardar las métricas finales en JSON.
import pickle             # Para serializar (guardar) los modelos entrenados.
from pathlib import Path  # Rutas portables.

import numpy as np        # Operaciones numéricas.
import pandas as pd       # Manejo de los datos de entrenamiento.
# Componentes de scikit-learn usados por los tres modelos:
from sklearn.cluster import KMeans                       # Clustering.
from sklearn.compose import ColumnTransformer           # Preprocesado por columna.
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression       # Baseline y tendencias.
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_absolute_error, r2_score)
from sklearn.pipeline import Pipeline                   # Encadenar pre + modelo.
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Rutas de datos procesados (entrada) y de modelos (salida).
RAIZ = Path(__file__).resolve().parents[1]
DIR_PROC = RAIZ / "data" / "processed"
DIR_MODELOS = RAIZ / "models"
DIR_MODELOS.mkdir(parents=True, exist_ok=True)

SEMILLA = 42   # Semilla fija para reproducibilidad de los entrenamientos.

# Definición de las columnas de entrada (features) categóricas y numéricas.
CAT_FEATURES = ["modalidad", "provincia", "categoria_principal", "tipo_contrato"]
NUM_FEATURES = ["num_habilidades", "num_habilidades_emergentes",
                "indice_demanda_prom", "tiene_emergente"]


def cargar_datos() -> pd.DataFrame:
    """Carga el dataset procesado que produjo el pipeline."""
    ruta = DIR_PROC / "ofertas_procesadas.csv"
    if not ruta.exists():
        raise FileNotFoundError("Ejecute primero el pipeline (src/pipeline.py).")
    return pd.read_csv(ruta, parse_dates=["fecha_publicacion", "anio_mes"])


# ---------------------------------------------------------------------------
# A) REGRESIÓN: salario
# ---------------------------------------------------------------------------
def entrenar_regresion(df: pd.DataFrame) -> dict:
    """Entrena un Random Forest para predecir el salario promedio."""
    from sklearn.model_selection import train_test_split

    # X = variables predictoras; y = variable objetivo (salario).
    X = df[CAT_FEATURES + NUM_FEATURES + ["nivel"]]
    y = df["salario_promedio"]

    # Preprocesado: one-hot a las categóricas y escalado a las numéricas.
    cat = CAT_FEATURES + ["nivel"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), NUM_FEATURES),
    ])

    # Pipeline = preprocesado + modelo Random Forest.
    modelo = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=250, max_depth=14,
                                     random_state=SEMILLA, n_jobs=-1)),
    ])

    # División 80/20 entre entrenamiento y prueba.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)
    modelo.fit(Xtr, ytr)        # Entrenamiento.
    pred = modelo.predict(Xte)  # Predicción sobre el conjunto de prueba.

    # Métricas: R² (bondad de ajuste) y MAE (error absoluto medio).
    r2 = r2_score(yte, pred)
    mae = mean_absolute_error(yte, pred)

    # Baseline lineal para comparar contra el Random Forest.
    base = Pipeline([("pre", pre), ("lr", LinearRegression())])
    base.fit(Xtr, ytr)
    r2_base = r2_score(yte, base.predict(Xte))

    # Se guarda el modelo entrenado para usarlo en el dashboard.
    with open(DIR_MODELOS / "modelo_regresion_salario.pkl", "wb") as f:
        pickle.dump(modelo, f)

    print(f"   [Regresión] R2(RandomForest)={r2:.3f} | "
          f"R2(LinealBase)={r2_base:.3f} | MAE=${mae:,.0f}")
    return {"r2": round(r2, 3), "r2_base": round(r2_base, 3),
            "mae": round(mae, 1), "n_test": len(yte)}


# ---------------------------------------------------------------------------
# B) CLASIFICACIÓN: nivel del puesto
# ---------------------------------------------------------------------------
def entrenar_clasificacion(df: pd.DataFrame) -> dict:
    """Entrena un Random Forest para clasificar el nivel del puesto."""
    from sklearn.model_selection import train_test_split

    # Aquí el salario SÍ es una feature (ayuda a inferir el nivel).
    feats_num = NUM_FEATURES + ["salario_promedio"]
    X = df[CAT_FEATURES + feats_num]
    y = df["nivel"]   # Variable objetivo: Junior/Semi Senior/Senior/Lead.

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", StandardScaler(), feats_num),
    ])
    modelo = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=16,
                                      random_state=SEMILLA, n_jobs=-1)),
    ])

    # stratify=y mantiene la proporción de cada nivel en train y test.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA, stratify=y)
    modelo.fit(Xtr, ytr)
    pred = modelo.predict(Xte)

    # Métricas: accuracy global y F1 macro (promedio por clase).
    acc = accuracy_score(yte, pred)
    reporte = classification_report(yte, pred, output_dict=True, zero_division=0)

    with open(DIR_MODELOS / "modelo_clasificacion_nivel.pkl", "wb") as f:
        pickle.dump(modelo, f)

    print(f"   [Clasificación] Accuracy={acc:.3f} "
          f"| F1-macro={reporte['macro avg']['f1-score']:.3f}")
    return {"accuracy": round(acc, 3),
            "f1_macro": round(reporte["macro avg"]["f1-score"], 3),
            "n_test": len(yte)}


# ---------------------------------------------------------------------------
# C) CLUSTERING: segmentación de perfiles
# ---------------------------------------------------------------------------
def entrenar_clustering(df: pd.DataFrame, k: int = 4) -> dict:
    """Agrupa las ofertas en k segmentos con KMeans (aprendizaje no supervisado)."""
    # Variables usadas para segmentar (todas numéricas).
    feats = ["salario_promedio", "num_habilidades",
             "num_habilidades_emergentes", "indice_demanda_prom"]
    X = df[feats].fillna(df[feats].median())   # Imputa faltantes con la mediana.

    # KMeans requiere escalar para que ninguna variable domine por su magnitud.
    escalador = StandardScaler()
    Xs = escalador.fit_transform(X)

    # Entrenamiento de KMeans con k clusters.
    km = KMeans(n_clusters=k, random_state=SEMILLA, n_init=10)
    etiquetas = km.fit_predict(Xs)

    # Se añade la etiqueta de cluster a cada oferta.
    df_cl = df.copy()
    df_cl["cluster"] = etiquetas

    # Perfil promedio de cada cluster (para interpretar los segmentos).
    perfil = (df_cl.groupby("cluster")[feats]
              .mean().round(1).reset_index())
    perfil["n_ofertas"] = df_cl.groupby("cluster").size().values

    # Etiqueta interpretativa por salario medio (ordena clusters por sueldo).
    orden = perfil.sort_values("salario_promedio")["cluster"].tolist()
    nombres = {orden[0]: "Entrada / Bajo costo",
               orden[1]: "Generalista intermedio",
               orden[-2]: "Especialista demandado",
               orden[-1]: "Alta especialización / Premium"}
    perfil["segmento"] = perfil["cluster"].map(nombres)

    # Se guarda el modelo (escalador + kmeans + features usadas).
    with open(DIR_MODELOS / "modelo_clustering.pkl", "wb") as f:
        pickle.dump({"escalador": escalador, "kmeans": km, "feats": feats}, f)

    # Se exportan las ofertas etiquetadas y el perfil de los clusters.
    df_cl.to_csv(DIR_PROC / "ofertas_con_cluster.csv", index=False, encoding="utf-8")
    perfil.to_csv(DIR_PROC / "perfil_clusters.csv", index=False, encoding="utf-8")

    print(f"   [Clustering] k={k} | inercia={km.inertia_:.0f}")
    return {"k": k, "perfil": perfil.to_dict(orient="records")}


# ---------------------------------------------------------------------------
# Análisis de tendencia: habilidades emergentes
# ---------------------------------------------------------------------------
def proyectar_habilidades_emergentes() -> dict:
    """Proyecta la demanda futura de cada skill emergente por regresión lineal."""
    ruta = DIR_PROC / "habilidades_largo.csv"
    largo = pd.read_csv(ruta, parse_dates=["anio_mes"])

    # Total de ofertas por mes (denominador para calcular la cuota %).
    ofertas_mes = (largo.groupby("anio_mes")["id_oferta"].nunique()
                   .rename("total_ofertas"))

    # Se excluye el último mes por ser parcial (sesga la tendencia).
    meses = sorted(ofertas_mes.index)
    meses_validos = meses[:-1] if len(meses) > 3 else meses

    # Se filtran solo las tecnologías marcadas como emergentes.
    emergentes = largo[largo["emergente"] == True].copy()  # noqa: E712
    serie = (emergentes.groupby(["anio_mes", "tecnologia"])["id_oferta"]
             .nunique().reset_index(name="conteo"))
    serie = serie[serie["anio_mes"].isin(meses_validos)]
    serie = serie.merge(ofertas_mes, on="anio_mes", how="left")
    # Cuota mensual: % de ofertas del mes que solicitan la tecnología.
    serie["cuota_pct"] = 100 * serie["conteo"] / serie["total_ofertas"]

    proyecciones = []
    # Para cada tecnología se ajusta una recta a su cuota a lo largo del tiempo.
    for tech, g in serie.groupby("tecnologia"):
        g = g.sort_values("anio_mes")
        if len(g) < 4:        # Se necesitan al menos 4 puntos para la tendencia.
            continue
        x = np.arange(len(g)).reshape(-1, 1)   # Eje temporal (0,1,2,...).
        y = g["cuota_pct"].values
        lr = LinearRegression().fit(x, y)
        pendiente = float(lr.coef_[0])              # Crecimiento por mes.
        prox = float(lr.predict([[len(g) + 2]])[0]) # Proyección a 2 meses.
        proyecciones.append({
            "tecnologia": tech,
            # Crecimiento en puntos porcentuales de cuota por mes.
            "crecimiento_pp_mes": round(pendiente, 3),
            "cuota_actual_pct": round(float(y[-1]), 1),
            "proyeccion_cuota_2m_pct": round(max(prox, 0), 1),
        })

    # Se ordenan de mayor a menor crecimiento.
    proyecciones.sort(key=lambda d: d["crecimiento_pp_mes"], reverse=True)
    return {"tendencias": proyecciones}


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def entrenar_todo() -> dict:
    """Entrena los tres modelos + el análisis de tendencias y guarda métricas."""
    df = cargar_datos()
    print(">> Entrenando modelos de ML...")
    res = {
        "regresion": entrenar_regresion(df),
        "clasificacion": entrenar_clasificacion(df),
        "clustering": entrenar_clustering(df),
        "tendencias_emergentes": proyectar_habilidades_emergentes(),
    }
    # Todas las métricas se consolidan en un único JSON para el dashboard.
    with open(DIR_MODELOS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(">> Modelos entrenados y métricas guardadas en models/metricas.json")
    return res


# Punto de entrada: permite reentrenar todos los modelos de forma independiente.
if __name__ == "__main__":
    entrenar_todo()
