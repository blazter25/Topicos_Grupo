"""
Dashboard interactivo (Streamlit).
====================================================================
Proyecto: Análisis del Mercado Laboral IT en Panamá
Curso:    Gestión de la Información - UTP, I Semestre 2026
Autor:    Angel Martínez  -  Cédula: 8-893-602

Ejecutar:
    streamlit run app/dashboard.py
====================================================================
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
DIR_PROC = RAIZ / "data" / "processed"
DIR_MODELOS = RAIZ / "models"

st.set_page_config(page_title="Mercado Laboral IT - Panamá",
                   page_icon="💻", layout="wide")


# ---------------------------------------------------------------------------
# Carga de datos y modelos (cacheada)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv(DIR_PROC / "ofertas_procesadas.csv",
                     parse_dates=["fecha_publicacion", "anio_mes"])
    largo = pd.read_csv(DIR_PROC / "habilidades_largo.csv",
                        parse_dates=["anio_mes"])
    cluster = None
    ruta_cl = DIR_PROC / "ofertas_con_cluster.csv"
    if ruta_cl.exists():
        cluster = pd.read_csv(ruta_cl, parse_dates=["fecha_publicacion"])
    return df, largo, cluster


@st.cache_data
def cargar_metricas():
    ruta = DIR_MODELOS / "metricas.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return {}


@st.cache_resource
def cargar_modelo(nombre):
    ruta = DIR_MODELOS / nombre
    if ruta.exists():
        with open(ruta, "rb") as f:
            return pickle.load(f)
    return None


def verificar_datos():
    if not (DIR_PROC / "ofertas_procesadas.csv").exists():
        st.error("No se encontraron datos procesados. "
                 "Ejecute primero:  python main.py")
        st.stop()


# ---------------------------------------------------------------------------
verificar_datos()
df, largo, cluster = cargar_datos()
metricas = cargar_metricas()

st.title("💻 Análisis del Mercado Laboral IT en Panamá")
st.caption("Proyecto Integrador - Gestión de la Información (UTP) · "
           "Angel Martínez · Cédula 8-893-602")

# ---------------------------------------------------------------------------
# SIDEBAR - Filtros
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros")

provincias = sorted(df["provincia"].dropna().unique())
sel_prov = st.sidebar.multiselect("Provincia", provincias, default=provincias)

modalidades = sorted(df["modalidad"].dropna().unique())
sel_mod = st.sidebar.multiselect("Modalidad", modalidades, default=modalidades)

niveles = ["Junior", "Semi Senior", "Senior", "Lead"]
niveles = [n for n in niveles if n in df["nivel"].unique()]
sel_niv = st.sidebar.multiselect("Nivel", niveles, default=niveles)

techs = sorted(largo["tecnologia"].dropna().unique())
sel_tech = st.sidebar.selectbox("Tecnología (filtro de ofertas)",
                                ["(Todas)"] + techs)

sal_min = int(df["salario_promedio"].min())
sal_max = int(df["salario_promedio"].max())
rango_sal = st.sidebar.slider("Rango salarial (USD)", sal_min, sal_max,
                              (sal_min, sal_max), step=50)

# Aplicar filtros
mask = (df["provincia"].isin(sel_prov)
        & df["modalidad"].isin(sel_mod)
        & df["nivel"].isin(sel_niv)
        & df["salario_promedio"].between(*rango_sal))

if sel_tech != "(Todas)":
    ids = largo.loc[largo["tecnologia"] == sel_tech, "id_oferta"].unique()
    mask &= df["id_oferta"].isin(ids)

dff = df[mask].copy()

if dff.empty:
    st.warning("No hay ofertas que cumplan los filtros seleccionados.")
    st.stop()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ofertas", f"{len(dff):,}")
c2.metric("Salario promedio", f"${dff['salario_promedio'].mean():,.0f}")
c3.metric("% Remoto/Híbrido",
          f"{(dff['modalidad'].isin(['Remoto', 'Híbrido']).mean()*100):.0f}%")
c4.metric("Tecnologías distintas", f"{largo['tecnologia'].nunique()}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Tendencias", "🛠️ Habilidades", "🤖 Modelos ML",
     "🎯 Predictor de salario", "🔎 Datos"])

# ---------------------------------------------------------------------------
# TAB 1: Tendencias
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Tendencia temporal de ofertas")
    serie = dff.groupby("anio_mes").size().reset_index(name="ofertas")
    fig = px.area(serie, x="anio_mes", y="ofertas",
                  labels={"anio_mes": "Mes", "ofertas": "N° de ofertas"})
    st.plotly_chart(fig, width="stretch")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Salario promedio por mes**")
        sm = dff.groupby("anio_mes")["salario_promedio"].mean().reset_index()
        st.plotly_chart(px.line(sm, x="anio_mes", y="salario_promedio",
                                markers=True), width="stretch")
    with colB:
        st.markdown("**Ofertas por modalidad**")
        mod = dff["modalidad"].value_counts().reset_index()
        mod.columns = ["modalidad", "conteo"]
        st.plotly_chart(px.pie(mod, names="modalidad", values="conteo",
                               hole=0.4), width="stretch")

    st.markdown("**Distribución salarial por nivel**")
    st.plotly_chart(
        px.box(dff, x="nivel", y="salario_promedio", color="nivel",
               category_orders={"nivel": niveles}),
        width="stretch")

# ---------------------------------------------------------------------------
# TAB 2: Habilidades
# ---------------------------------------------------------------------------
with tab2:
    largo_f = largo[largo["id_oferta"].isin(dff["id_oferta"])]

    st.subheader("Top 15 tecnologías más demandadas")
    top = largo_f["tecnologia"].value_counts().head(15).reset_index()
    top.columns = ["tecnologia", "conteo"]
    st.plotly_chart(px.bar(top, x="conteo", y="tecnologia", orientation="h",
                           color="conteo", color_continuous_scale="Blues")
                    .update_layout(yaxis={"categoryorder": "total ascending"}),
                    width="stretch")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Demanda por categoría**")
        cat = largo_f["categoria"].value_counts().reset_index()
        cat.columns = ["categoria", "conteo"]
        st.plotly_chart(px.bar(cat, x="categoria", y="conteo", color="categoria"),
                        width="stretch")
    with colB:
        st.markdown("**Salario medio por tecnología (top 12)**")
        sal_tech = (largo_f.groupby("tecnologia")["salario_promedio"]
                    .mean().sort_values(ascending=False).head(12).reset_index())
        st.plotly_chart(px.bar(sal_tech, x="salario_promedio", y="tecnologia",
                               orientation="h", color="salario_promedio",
                               color_continuous_scale="Greens")
                        .update_layout(yaxis={"categoryorder": "total ascending"}),
                        width="stretch")

    st.subheader("Predicción de habilidades emergentes")
    st.caption("Tendencia por regresión lineal sobre la cuota mensual "
               "(% de ofertas que solicitan cada tecnología).")
    tend = metricas.get("tendencias_emergentes", {}).get("tendencias", [])
    if tend:
        td = pd.DataFrame(tend)
        st.dataframe(td, width="stretch", hide_index=True)
        st.plotly_chart(
            px.bar(td.head(8), x="crecimiento_pp_mes", y="tecnologia",
                   orientation="h", color="crecimiento_pp_mes",
                   color_continuous_scale="Oranges",
                   labels={"crecimiento_pp_mes":
                           "Crecimiento de cuota (puntos %/mes)"})
            .update_layout(yaxis={"categoryorder": "total ascending"}),
            width="stretch")

# ---------------------------------------------------------------------------
# TAB 3: Modelos ML
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Resultados de los modelos de Machine Learning")
    if metricas:
        m1, m2, m3 = st.columns(3)
        reg = metricas.get("regresion", {})
        clf = metricas.get("clasificacion", {})
        with m1:
            st.markdown("**A) Regresión de salario**")
            st.metric("R² (Random Forest)", reg.get("r2", "-"))
            st.metric("R² (baseline lineal)", reg.get("r2_base", "-"))
            st.metric("Error medio (MAE)", f"${reg.get('mae', 0):,.0f}")
        with m2:
            st.markdown("**B) Clasificación de nivel**")
            st.metric("Accuracy", clf.get("accuracy", "-"))
            st.metric("F1 macro", clf.get("f1_macro", "-"))
            st.metric("Muestras de prueba", clf.get("n_test", "-"))
        with m3:
            st.markdown("**C) Clustering de perfiles**")
            st.metric("N° de clusters", metricas.get("clustering", {}).get("k", "-"))

        perfil = metricas.get("clustering", {}).get("perfil", [])
        if perfil:
            st.markdown("**Perfiles (segmentos) identificados**")
            st.dataframe(pd.DataFrame(perfil), width="stretch",
                         hide_index=True)
        if cluster is not None:
            st.plotly_chart(
                px.scatter(cluster, x="indice_demanda_prom", y="salario_promedio",
                           color=cluster["cluster"].astype(str),
                           size="num_habilidades", opacity=0.6,
                           labels={"color": "Cluster"},
                           title="Segmentación de ofertas (KMeans)"),
                width="stretch")
    else:
        st.info("No se encontraron métricas. Ejecute: python main.py")

# ---------------------------------------------------------------------------
# TAB 4: Predictor de salario
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Predictor interactivo de salario")
    st.caption("Usa el modelo de regresión (Random Forest) entrenado.")
    modelo = cargar_modelo("modelo_regresion_salario.pkl")
    if modelo is None:
        st.info("Modelo no disponible. Ejecute: python main.py")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            in_nivel = st.selectbox("Nivel", niveles)
            in_mod = st.selectbox("Modalidad", modalidades)
        with col2:
            in_prov = st.selectbox("Provincia", provincias)
            in_cat = st.selectbox("Categoría principal",
                                  sorted(df["categoria_principal"].unique()))
        with col3:
            in_contrato = st.selectbox("Tipo de contrato",
                                       sorted(df["tipo_contrato"].unique()))
            in_nhab = st.slider("N° de habilidades", 1, 8, 4)
            in_emerg = st.slider("Habilidades emergentes", 0, 4, 1)
            in_demanda = st.slider("Índice de demanda promedio", 30, 100, 70)

        entrada = pd.DataFrame([{
            "modalidad": in_mod, "provincia": in_prov,
            "categoria_principal": in_cat, "tipo_contrato": in_contrato,
            "num_habilidades": in_nhab,
            "num_habilidades_emergentes": in_emerg,
            "indice_demanda_prom": in_demanda,
            "tiene_emergente": int(in_emerg > 0),
            "nivel": in_nivel,
        }])
        pred = float(modelo.predict(entrada)[0])
        st.success(f"💰 Salario mensual estimado: **${pred:,.0f} USD**")

# ---------------------------------------------------------------------------
# TAB 5: Datos
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Datos filtrados")
    cols = ["id_oferta", "titulo", "empresa", "fuente", "fecha_publicacion",
            "provincia", "modalidad", "nivel", "salario_promedio",
            "habilidades"]
    cols = [c for c in cols if c in dff.columns]
    st.dataframe(dff[cols], width="stretch", hide_index=True)
    st.download_button("Descargar CSV filtrado",
                       dff[cols].to_csv(index=False).encode("utf-8"),
                       "ofertas_filtradas.csv", "text/csv")

st.divider()
st.caption("Fuentes: (1) Ofertas de empleo IT · (2) Catálogo de tecnologías. "
           "Pipeline + ML + Dashboard — Segundo Parcial.")
