"""
Pipeline de datos (ETL).
====================================================================
Proyecto: Análisis del Mercado Laboral IT en Panamá
Curso:    Gestión de la Información - UTP, I Semestre 2026
Autor:    Angel Martínez  -  Cédula: 8-893-602

Etapas del pipeline:
  1. INGESTA       : carga las 2 fuentes (CSV de ofertas + JSON de skills)
  2. LIMPIEZA      : duplicados, tipos, faltantes, normalización
  3. TRANSFORMACIÓN: parseo de salarios, explosión y enriquecimiento de
                     habilidades con el catálogo, feature engineering
  4. CARGA         : guarda dataset procesado en data/processed/

El pipeline es idempotente y documentado paso a paso.
====================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROC = RAIZ / "data" / "processed"
DIR_PROC.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. INGESTA
# ---------------------------------------------------------------------------
def ingestar_ofertas() -> pd.DataFrame:
    """Fuente 1: ofertas de empleo (CSV)."""
    ruta = DIR_RAW / "ofertas_empleo_it.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecute primero src/generar_datos.py")
    df = pd.read_csv(ruta, dtype={"salario_min": "object", "salario_max": "object"})
    print(f"   [ingesta] Fuente 1 (CSV) ofertas: {len(df)} filas")
    return df


def ingestar_catalogo() -> pd.DataFrame:
    """Fuente 2: catálogo de tecnologías (JSON)."""
    ruta = DIR_RAW / "catalogo_tecnologias.json"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecute primero src/generar_datos.py")
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"   [ingesta] Fuente 2 (JSON) catálogo: {len(df)} tecnologías")
    return df


# ---------------------------------------------------------------------------
# 2. LIMPIEZA
# ---------------------------------------------------------------------------
def limpiar_ofertas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    filas_ini = len(df)

    # 2.1 Eliminar duplicados exactos por id + fecha
    df = df.drop_duplicates(subset=["id_oferta", "fecha_publicacion"]).copy()

    # 2.2 Tipos: fechas
    df["fecha_publicacion"] = pd.to_datetime(df["fecha_publicacion"], errors="coerce")
    df = df.dropna(subset=["fecha_publicacion"]).copy()

    # 2.3 Faltantes en texto
    df["empresa"] = df["empresa"].fillna("Confidencial").replace("", "Confidencial")

    # 2.4 Normalización de categóricas
    df["modalidad"] = df["modalidad"].str.strip().str.title()
    df["nivel"] = df["nivel"].str.strip().str.title()
    df["provincia"] = df["provincia"].str.strip()

    filas_fin = len(df)
    print(f"   [limpieza] {filas_ini} -> {filas_fin} filas "
          f"({filas_ini - filas_fin} eliminadas)")
    return df


# ---------------------------------------------------------------------------
# 3. TRANSFORMACIÓN
# ---------------------------------------------------------------------------
def _a_numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie.replace("", np.nan), errors="coerce")


def transformar(df: pd.DataFrame, catalogo: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 3.1 Parseo de salarios
    df["salario_min"] = _a_numero(df["salario_min"])
    df["salario_max"] = _a_numero(df["salario_max"])

    # Imputación: si falta uno de los dos, usar el otro; salario_promedio
    df["salario_min"] = df["salario_min"].fillna(df["salario_max"])
    df["salario_max"] = df["salario_max"].fillna(df["salario_min"])
    df["salario_promedio"] = df[["salario_min", "salario_max"]].mean(axis=1)

    # Si ambos faltan, imputar por mediana del nivel
    mediana_nivel = df.groupby("nivel")["salario_promedio"].transform("median")
    df["salario_promedio"] = df["salario_promedio"].fillna(mediana_nivel)

    # 3.2 Diccionarios del catálogo (Fuente 2) para enriquecer
    cat_demanda = dict(zip(catalogo["tecnologia"], catalogo["indice_demanda"]))
    cat_categoria = dict(zip(catalogo["tecnologia"], catalogo["categoria"]))
    cat_emergente = dict(zip(catalogo["tecnologia"], catalogo["emergente"]))

    # 3.3 Procesar lista de habilidades por oferta
    df["lista_habilidades"] = (
        df["habilidades"].fillna("").apply(
            lambda x: [s.strip() for s in x.split(",") if s.strip()])
    )

    df["num_habilidades"] = df["lista_habilidades"].apply(len)
    df["num_habilidades_emergentes"] = df["lista_habilidades"].apply(
        lambda ls: sum(1 for s in ls if cat_emergente.get(s, False)))
    df["indice_demanda_prom"] = df["lista_habilidades"].apply(
        lambda ls: np.mean([cat_demanda.get(s, 50) for s in ls]) if ls else 50.0)

    def categoria_principal(ls):
        cats = [cat_categoria.get(s) for s in ls if cat_categoria.get(s)]
        if not cats:
            return "Otros"
        return pd.Series(cats).mode().iat[0]

    df["categoria_principal"] = df["lista_habilidades"].apply(categoria_principal)
    df["tiene_emergente"] = (df["num_habilidades_emergentes"] > 0).astype(int)

    # 3.4 Variables temporales
    df["anio"] = df["fecha_publicacion"].dt.year
    df["mes"] = df["fecha_publicacion"].dt.to_period("M").astype(str)
    df["anio_mes"] = df["fecha_publicacion"].dt.to_period("M").dt.to_timestamp()

    # 3.5 Filtro de salarios atípicos (outliers extremos)
    q_low, q_high = df["salario_promedio"].quantile([0.005, 0.995])
    df = df[df["salario_promedio"].between(q_low, q_high)].copy()

    print(f"   [transformación] columnas finales: {df.shape[1]} | filas: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Tabla auxiliar: ofertas explotadas por habilidad (formato largo)
# ---------------------------------------------------------------------------
def construir_tabla_habilidades(df: pd.DataFrame, catalogo: pd.DataFrame) -> pd.DataFrame:
    largo = df[["id_oferta", "fecha_publicacion", "anio_mes", "mes",
                "nivel", "salario_promedio", "lista_habilidades"]].explode(
        "lista_habilidades").rename(columns={"lista_habilidades": "tecnologia"})
    largo = largo.dropna(subset=["tecnologia"])
    largo = largo.merge(catalogo, on="tecnologia", how="left")
    return largo


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def ejecutar_pipeline() -> dict:
    print(">> [1/4] INGESTA")
    ofertas = ingestar_ofertas()
    catalogo = ingestar_catalogo()

    print(">> [2/4] LIMPIEZA")
    ofertas = limpiar_ofertas(ofertas)

    print(">> [3/4] TRANSFORMACIÓN")
    df = transformar(ofertas, catalogo)
    tabla_hab = construir_tabla_habilidades(df, catalogo)

    print(">> [4/4] CARGA")
    # Guardamos sin la columna de lista (no serializa bien en CSV plano)
    df_out = df.drop(columns=["lista_habilidades"])
    ruta_proc = DIR_PROC / "ofertas_procesadas.csv"
    ruta_hab = DIR_PROC / "habilidades_largo.csv"
    df_out.to_csv(ruta_proc, index=False, encoding="utf-8")
    tabla_hab.to_csv(ruta_hab, index=False, encoding="utf-8")
    print(f"   guardado -> {ruta_proc} ({len(df_out)} filas)")
    print(f"   guardado -> {ruta_hab} ({len(tabla_hab)} filas)")

    return {"ofertas": df, "habilidades": tabla_hab, "catalogo": catalogo}


if __name__ == "__main__":
    ejecutar_pipeline()
    print(">> Pipeline finalizado correctamente.")
