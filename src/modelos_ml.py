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

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_absolute_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
DIR_PROC = RAIZ / "data" / "processed"
DIR_MODELOS = RAIZ / "models"
DIR_MODELOS.mkdir(parents=True, exist_ok=True)

SEMILLA = 42

CAT_FEATURES = ["modalidad", "provincia", "categoria_principal", "tipo_contrato"]
NUM_FEATURES = ["num_habilidades", "num_habilidades_emergentes",
                "indice_demanda_prom", "tiene_emergente"]


def cargar_datos() -> pd.DataFrame:
    ruta = DIR_PROC / "ofertas_procesadas.csv"
    if not ruta.exists():
        raise FileNotFoundError("Ejecute primero el pipeline (src/pipeline.py).")
    return pd.read_csv(ruta, parse_dates=["fecha_publicacion", "anio_mes"])


# ---------------------------------------------------------------------------
# A) REGRESIÓN: salario
# ---------------------------------------------------------------------------
def entrenar_regresion(df: pd.DataFrame) -> dict:
    from sklearn.model_selection import train_test_split

    X = df[CAT_FEATURES + NUM_FEATURES + ["nivel"]]
    y = df["salario_promedio"]

    cat = CAT_FEATURES + ["nivel"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), NUM_FEATURES),
    ])

    modelo = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=250, max_depth=14,
                                     random_state=SEMILLA, n_jobs=-1)),
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)
    modelo.fit(Xtr, ytr)
    pred = modelo.predict(Xte)

    r2 = r2_score(yte, pred)
    mae = mean_absolute_error(yte, pred)

    # Baseline lineal para comparar
    base = Pipeline([("pre", pre), ("lr", LinearRegression())])
    base.fit(Xtr, ytr)
    r2_base = r2_score(yte, base.predict(Xte))

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
    from sklearn.model_selection import train_test_split

    feats_num = NUM_FEATURES + ["salario_promedio"]
    X = df[CAT_FEATURES + feats_num]
    y = df["nivel"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", StandardScaler(), feats_num),
    ])
    modelo = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=16,
                                      random_state=SEMILLA, n_jobs=-1)),
    ])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA, stratify=y)
    modelo.fit(Xtr, ytr)
    pred = modelo.predict(Xte)

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
    feats = ["salario_promedio", "num_habilidades",
             "num_habilidades_emergentes", "indice_demanda_prom"]
    X = df[feats].fillna(df[feats].median())

    escalador = StandardScaler()
    Xs = escalador.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=SEMILLA, n_init=10)
    etiquetas = km.fit_predict(Xs)

    df_cl = df.copy()
    df_cl["cluster"] = etiquetas

    perfil = (df_cl.groupby("cluster")[feats]
              .mean().round(1).reset_index())
    perfil["n_ofertas"] = df_cl.groupby("cluster").size().values

    # Etiqueta interpretativa por salario medio
    orden = perfil.sort_values("salario_promedio")["cluster"].tolist()
    nombres = {orden[0]: "Entrada / Bajo costo",
               orden[1]: "Generalista intermedio",
               orden[-2]: "Especialista demandado",
               orden[-1]: "Alta especialización / Premium"}
    perfil["segmento"] = perfil["cluster"].map(nombres)

    with open(DIR_MODELOS / "modelo_clustering.pkl", "wb") as f:
        pickle.dump({"escalador": escalador, "kmeans": km, "feats": feats}, f)

    df_cl.to_csv(DIR_PROC / "ofertas_con_cluster.csv", index=False, encoding="utf-8")
    perfil.to_csv(DIR_PROC / "perfil_clusters.csv", index=False, encoding="utf-8")

    print(f"   [Clustering] k={k} | inercia={km.inertia_:.0f}")
    return {"k": k, "perfil": perfil.to_dict(orient="records")}


# ---------------------------------------------------------------------------
# Análisis de tendencia: habilidades emergentes
# ---------------------------------------------------------------------------
def proyectar_habilidades_emergentes() -> dict:
    ruta = DIR_PROC / "habilidades_largo.csv"
    largo = pd.read_csv(ruta, parse_dates=["anio_mes"])

    # Total de ofertas por mes (denominador para calcular la cuota %)
    ofertas_mes = (largo.groupby("anio_mes")["id_oferta"].nunique()
                   .rename("total_ofertas"))

    # Se excluye el último mes por ser parcial (sesga la tendencia)
    meses = sorted(ofertas_mes.index)
    meses_validos = meses[:-1] if len(meses) > 3 else meses

    emergentes = largo[largo["emergente"] == True].copy()  # noqa: E712
    serie = (emergentes.groupby(["anio_mes", "tecnologia"])["id_oferta"]
             .nunique().reset_index(name="conteo"))
    serie = serie[serie["anio_mes"].isin(meses_validos)]
    serie = serie.merge(ofertas_mes, on="anio_mes", how="left")
    # Cuota mensual: % de ofertas del mes que solicitan la tecnología
    serie["cuota_pct"] = 100 * serie["conteo"] / serie["total_ofertas"]

    proyecciones = []
    for tech, g in serie.groupby("tecnologia"):
        g = g.sort_values("anio_mes")
        if len(g) < 4:
            continue
        x = np.arange(len(g)).reshape(-1, 1)
        y = g["cuota_pct"].values
        lr = LinearRegression().fit(x, y)
        pendiente = float(lr.coef_[0])
        prox = float(lr.predict([[len(g) + 2]])[0])
        proyecciones.append({
            "tecnologia": tech,
            # Crecimiento en puntos porcentuales de cuota por mes
            "crecimiento_pp_mes": round(pendiente, 3),
            "cuota_actual_pct": round(float(y[-1]), 1),
            "proyeccion_cuota_2m_pct": round(max(prox, 0), 1),
        })

    proyecciones.sort(key=lambda d: d["crecimiento_pp_mes"], reverse=True)
    return {"tendencias": proyecciones}


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def entrenar_todo() -> dict:
    df = cargar_datos()
    print(">> Entrenando modelos de ML...")
    res = {
        "regresion": entrenar_regresion(df),
        "clasificacion": entrenar_clasificacion(df),
        "clustering": entrenar_clustering(df),
        "tendencias_emergentes": proyectar_habilidades_emergentes(),
    }
    with open(DIR_MODELOS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(">> Modelos entrenados y métricas guardadas en models/metricas.json")
    return res


if __name__ == "__main__":
    entrenar_todo()
