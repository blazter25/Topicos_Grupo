"""
Dashboard interactivo (Streamlit).
====================================================================
Proyecto: Análisis del Mercado Laboral IT en Panamá
Curso:    Gestión de la Información - UTP, I Semestre 2026
Autor:    Angel Martínez  -  Cédula: 8-893-602

Este archivo construye la interfaz web del proyecto. Lee los datos ya
procesados por el pipeline y los modelos de ML entrenados, y los muestra
en cinco pestañas interactivas (tendencias, habilidades, modelos,
predictor de salario y tabla de datos). Incluye además un "modo oscuro"
(modo black) activable desde la barra lateral.

Ejecutar:
    streamlit run app/dashboard.py
====================================================================
"""

# 'annotations' permite usar anotaciones de tipo modernas sin problemas
# de compatibilidad con versiones previas de Python.
from __future__ import annotations

import json     # Para leer el archivo de métricas (metricas.json).
import pickle   # Para cargar los modelos de ML serializados (.pkl).
from pathlib import Path  # Manejo de rutas multiplataforma (Windows/Linux).

import pandas as pd            # Manipulación de datos en forma de tablas.
import plotly.express as px    # Gráficos interactivos de alto nivel.
import plotly.io as pio        # Configuración global de los temas de Plotly.
import streamlit as st         # Framework del dashboard web.

# ---------------------------------------------------------------------------
# Rutas base del proyecto (calculadas a partir de la ubicación de este archivo)
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]   # Carpeta raíz del proyecto.
DIR_PROC = RAIZ / "data" / "processed"       # Datos procesados por el pipeline.
DIR_MODELOS = RAIZ / "models"                # Modelos entrenados + métricas.

# Configuración general de la página (título de pestaña, ícono y ancho completo).
st.set_page_config(page_title="Mercado Laboral IT - Panamá",
                   page_icon="💻", layout="wide")


# ---------------------------------------------------------------------------
# MODO OSCURO (modo black)
# ---------------------------------------------------------------------------
def aplicar_tema(oscuro: bool) -> str:
    """Aplica el tema visual según el interruptor del modo oscuro.

    - Cambia el tema por defecto de TODOS los gráficos Plotly de una sola vez
      (no hace falta tocar cada gráfico individualmente).
    - Inyecta CSS para pintar el fondo, el texto y la barra lateral en negro
      cuando el modo oscuro está activo.

    Devuelve el nombre de la plantilla de Plotly usada (por si se necesita).
    """
    if oscuro:
        # Plantilla oscura nativa de Plotly para todos los gráficos.
        pio.templates.default = "plotly_dark"
        # CSS personalizado: fondo negro y texto claro en toda la app.
        st.markdown(
            """
            <style>
              /* Contenedor principal y cuerpo de la app */
              [data-testid="stAppViewContainer"], .stApp {
                  background-color: #0e1117;
                  color: #fafafa;
              }
              /* Encabezado superior */
              [data-testid="stHeader"] { background-color: #0e1117; }
              /* Barra lateral */
              [data-testid="stSidebar"] {
                  background-color: #161b22;
              }
              /* Texto general (títulos, párrafos, etiquetas) */
              h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown {
                  color: #fafafa !important;
              }
              /* Tarjetas de métricas (KPIs) */
              [data-testid="stMetricValue"] { color: #58a6ff !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return "plotly_dark"
    else:
        # Tema claro por defecto de Plotly.
        pio.templates.default = "plotly"
        return "plotly"


# ---------------------------------------------------------------------------
# Carga de datos y modelos (cacheada para no releer en cada interacción)
# ---------------------------------------------------------------------------
@st.cache_data   # Cachea el resultado: los CSV solo se leen una vez.
def cargar_datos():
    """Carga los tres CSV procesados: ofertas, habilidades y clusters."""
    # Dataset principal de ofertas (una fila por oferta).
    df = pd.read_csv(DIR_PROC / "ofertas_procesadas.csv",
                     parse_dates=["fecha_publicacion", "anio_mes"])
    # Tabla en formato largo (una fila por oferta-tecnología).
    largo = pd.read_csv(DIR_PROC / "habilidades_largo.csv",
                        parse_dates=["anio_mes"])
    # Ofertas con su cluster asignado (puede no existir si no se entrenó).
    cluster = None
    ruta_cl = DIR_PROC / "ofertas_con_cluster.csv"
    if ruta_cl.exists():
        cluster = pd.read_csv(ruta_cl, parse_dates=["fecha_publicacion"])
    return df, largo, cluster


@st.cache_data
def cargar_metricas():
    """Lee las métricas de los modelos desde models/metricas.json."""
    ruta = DIR_MODELOS / "metricas.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return {}   # Si no existe, se devuelve un diccionario vacío.


@st.cache_resource   # Cachea el objeto modelo (no es serializable como dato).
def cargar_modelo(nombre):
    """Carga un modelo de ML serializado (.pkl) desde la carpeta models."""
    ruta = DIR_MODELOS / nombre
    if ruta.exists():
        with open(ruta, "rb") as f:
            return pickle.load(f)
    return None


def verificar_datos():
    """Detiene la app con un mensaje claro si faltan los datos procesados."""
    if not (DIR_PROC / "ofertas_procesadas.csv").exists():
        st.error("No se encontraron datos procesados. "
                 "Ejecute primero:  python main.py")
        st.stop()


# ---------------------------------------------------------------------------
# Punto de entrada de la app: verificar datos y cargarlos en memoria.
# ---------------------------------------------------------------------------
verificar_datos()
df, largo, cluster = cargar_datos()
metricas = cargar_metricas()

# Encabezado principal del dashboard.
st.title("💻 Análisis del Mercado Laboral IT en Panamá")
st.caption("Proyecto Integrador - Gestión de la Información (UTP) · "
           "Angel Martínez · Cédula 8-893-602")

# ---------------------------------------------------------------------------
# SIDEBAR - Filtros e interruptor de modo oscuro
# ---------------------------------------------------------------------------
# Interruptor del modo oscuro: debe evaluarse ANTES de crear los gráficos
# para que el tema se aplique a todas las figuras de la página.
modo_oscuro = st.sidebar.toggle("🌙 Modo oscuro (black)", value=False)
aplicar_tema(modo_oscuro)

st.sidebar.header("Filtros")

# Filtro por provincia (multiselección, todas marcadas por defecto).
provincias = sorted(df["provincia"].dropna().unique())
sel_prov = st.sidebar.multiselect("Provincia", provincias, default=provincias)

# Filtro por modalidad de trabajo (Presencial/Híbrido/Remoto).
modalidades = sorted(df["modalidad"].dropna().unique())
sel_mod = st.sidebar.multiselect("Modalidad", modalidades, default=modalidades)

# Filtro por nivel del puesto, respetando el orden lógico de seniority.
niveles = ["Junior", "Semi Senior", "Senior", "Lead"]
niveles = [n for n in niveles if n in df["nivel"].unique()]
sel_niv = st.sidebar.multiselect("Nivel", niveles, default=niveles)

# Filtro por tecnología específica (afecta qué ofertas se muestran).
techs = sorted(largo["tecnologia"].dropna().unique())
sel_tech = st.sidebar.selectbox("Tecnología (filtro de ofertas)",
                                ["(Todas)"] + techs)

# Filtro por rango salarial mediante un deslizador.
sal_min = int(df["salario_promedio"].min())
sal_max = int(df["salario_promedio"].max())
rango_sal = st.sidebar.slider("Rango salarial (USD)", sal_min, sal_max,
                              (sal_min, sal_max), step=50)

# Construcción de la máscara booleana que combina todos los filtros.
mask = (df["provincia"].isin(sel_prov)
        & df["modalidad"].isin(sel_mod)
        & df["nivel"].isin(sel_niv)
        & df["salario_promedio"].between(*rango_sal))

# Si se eligió una tecnología concreta, se filtran las ofertas que la piden.
if sel_tech != "(Todas)":
    ids = largo.loc[largo["tecnologia"] == sel_tech, "id_oferta"].unique()
    mask &= df["id_oferta"].isin(ids)

# DataFrame filtrado que alimentará todas las visualizaciones.
dff = df[mask].copy()

# Si los filtros dejan el conjunto vacío, se avisa y se detiene la app.
if dff.empty:
    st.warning("No hay ofertas que cumplan los filtros seleccionados.")
    st.stop()

# ---------------------------------------------------------------------------
# KPIs - Indicadores clave en la parte superior
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)   # Cuatro columnas, una métrica por columna.
c1.metric("Ofertas", f"{len(dff):,}")
c2.metric("Salario promedio", f"${dff['salario_promedio'].mean():,.0f}")
c3.metric("% Remoto/Híbrido",
          f"{(dff['modalidad'].isin(['Remoto', 'Híbrido']).mean()*100):.0f}%")
c4.metric("Tecnologías distintas", f"{largo['tecnologia'].nunique()}")

st.divider()   # Línea separadora horizontal.

# Definición de las cinco pestañas del dashboard.
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Tendencias", "🛠️ Habilidades", "🤖 Modelos ML",
     "🎯 Predictor de salario", "🔎 Datos"])

# ---------------------------------------------------------------------------
# TAB 1: Tendencias temporales
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Tendencia temporal de ofertas")
    # Conteo de ofertas por mes -> gráfico de área.
    serie = dff.groupby("anio_mes").size().reset_index(name="ofertas")
    fig = px.area(serie, x="anio_mes", y="ofertas",
                  labels={"anio_mes": "Mes", "ofertas": "N° de ofertas"})
    st.plotly_chart(fig, width="stretch")

    colA, colB = st.columns(2)
    with colA:
        # Evolución del salario promedio mes a mes.
        st.markdown("**Salario promedio por mes**")
        sm = dff.groupby("anio_mes")["salario_promedio"].mean().reset_index()
        st.plotly_chart(px.line(sm, x="anio_mes", y="salario_promedio",
                                markers=True), width="stretch")
    with colB:
        # Distribución de ofertas por modalidad -> gráfico de dona.
        st.markdown("**Ofertas por modalidad**")
        mod = dff["modalidad"].value_counts().reset_index()
        mod.columns = ["modalidad", "conteo"]
        st.plotly_chart(px.pie(mod, names="modalidad", values="conteo",
                               hole=0.4), width="stretch")

    # Dispersión salarial por nivel -> diagrama de caja (boxplot).
    st.markdown("**Distribución salarial por nivel**")
    st.plotly_chart(
        px.box(dff, x="nivel", y="salario_promedio", color="nivel",
               category_orders={"nivel": niveles}),
        width="stretch")

# ---------------------------------------------------------------------------
# TAB 2: Habilidades / tecnologías demandadas
# ---------------------------------------------------------------------------
with tab2:
    # Se filtra la tabla larga para que coincida con las ofertas filtradas.
    largo_f = largo[largo["id_oferta"].isin(dff["id_oferta"])]

    st.subheader("Top 15 tecnologías más demandadas")
    top = largo_f["tecnologia"].value_counts().head(15).reset_index()
    top.columns = ["tecnologia", "conteo"]
    # Barras horizontales ordenadas de mayor a menor demanda.
    st.plotly_chart(px.bar(top, x="conteo", y="tecnologia", orientation="h",
                           color="conteo", color_continuous_scale="Blues")
                    .update_layout(yaxis={"categoryorder": "total ascending"}),
                    width="stretch")

    colA, colB = st.columns(2)
    with colA:
        # Demanda agrupada por categoría de tecnología.
        st.markdown("**Demanda por categoría**")
        cat = largo_f["categoria"].value_counts().reset_index()
        cat.columns = ["categoria", "conteo"]
        st.plotly_chart(px.bar(cat, x="categoria", y="conteo", color="categoria"),
                        width="stretch")
    with colB:
        # Salario medio asociado a cada tecnología (top 12).
        st.markdown("**Salario medio por tecnología (top 12)**")
        sal_tech = (largo_f.groupby("tecnologia")["salario_promedio"]
                    .mean().sort_values(ascending=False).head(12).reset_index())
        st.plotly_chart(px.bar(sal_tech, x="salario_promedio", y="tecnologia",
                               orientation="h", color="salario_promedio",
                               color_continuous_scale="Greens")
                        .update_layout(yaxis={"categoryorder": "total ascending"}),
                        width="stretch")

    # Tendencias emergentes calculadas por el módulo de ML.
    st.subheader("Predicción de habilidades emergentes")
    st.caption("Tendencia por regresión lineal sobre la cuota mensual "
               "(% de ofertas que solicitan cada tecnología).")
    tend = metricas.get("tendencias_emergentes", {}).get("tendencias", [])
    if tend:
        td = pd.DataFrame(tend)
        st.dataframe(td, width="stretch", hide_index=True)
        # Top 8 tecnologías con mayor crecimiento de cuota mensual.
        st.plotly_chart(
            px.bar(td.head(8), x="crecimiento_pp_mes", y="tecnologia",
                   orientation="h", color="crecimiento_pp_mes",
                   color_continuous_scale="Oranges",
                   labels={"crecimiento_pp_mes":
                           "Crecimiento de cuota (puntos %/mes)"})
            .update_layout(yaxis={"categoryorder": "total ascending"}),
            width="stretch")

# ---------------------------------------------------------------------------
# TAB 3: Resultados de los modelos de Machine Learning
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Resultados de los modelos de Machine Learning")
    if metricas:
        m1, m2, m3 = st.columns(3)
        reg = metricas.get("regresion", {})       # Métricas de regresión.
        clf = metricas.get("clasificacion", {})    # Métricas de clasificación.
        with m1:
            # A) Modelo de regresión de salario.
            st.markdown("**A) Regresión de salario**")
            st.metric("R² (Random Forest)", reg.get("r2", "-"))
            st.metric("R² (baseline lineal)", reg.get("r2_base", "-"))
            st.metric("Error medio (MAE)", f"${reg.get('mae', 0):,.0f}")
        with m2:
            # B) Modelo de clasificación de nivel.
            st.markdown("**B) Clasificación de nivel**")
            st.metric("Accuracy", clf.get("accuracy", "-"))
            st.metric("F1 macro", clf.get("f1_macro", "-"))
            st.metric("Muestras de prueba", clf.get("n_test", "-"))
        with m3:
            # C) Modelo de clustering de perfiles.
            st.markdown("**C) Clustering de perfiles**")
            st.metric("N° de clusters", metricas.get("clustering", {}).get("k", "-"))

        # Tabla con el perfil promedio de cada segmento (cluster).
        perfil = metricas.get("clustering", {}).get("perfil", [])
        if perfil:
            st.markdown("**Perfiles (segmentos) identificados**")
            st.dataframe(pd.DataFrame(perfil), width="stretch",
                         hide_index=True)
        # Visualización 2D de los clusters (salario vs índice de demanda).
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
# TAB 4: Predictor interactivo de salario
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Predictor interactivo de salario")
    st.caption("Usa el modelo de regresión (Random Forest) entrenado.")
    modelo = cargar_modelo("modelo_regresion_salario.pkl")
    if modelo is None:
        st.info("Modelo no disponible. Ejecute: python main.py")
    else:
        # Controles de entrada para las características de la oferta.
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

        # Se arma una fila con el mismo formato que espera el modelo.
        entrada = pd.DataFrame([{
            "modalidad": in_mod, "provincia": in_prov,
            "categoria_principal": in_cat, "tipo_contrato": in_contrato,
            "num_habilidades": in_nhab,
            "num_habilidades_emergentes": in_emerg,
            "indice_demanda_prom": in_demanda,
            "tiene_emergente": int(in_emerg > 0),
            "nivel": in_nivel,
        }])
        # Predicción y presentación del resultado.
        pred = float(modelo.predict(entrada)[0])
        st.success(f"💰 Salario mensual estimado: **${pred:,.0f} USD**")

# ---------------------------------------------------------------------------
# TAB 5: Tabla de datos filtrados + descarga
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Datos filtrados")
    # Columnas relevantes a mostrar (solo las que existan en el DataFrame).
    cols = ["id_oferta", "titulo", "empresa", "fuente", "fecha_publicacion",
            "provincia", "modalidad", "nivel", "salario_promedio",
            "habilidades"]
    cols = [c for c in cols if c in dff.columns]
    st.dataframe(dff[cols], width="stretch", hide_index=True)
    # Botón para descargar el subconjunto filtrado como CSV.
    st.download_button("Descargar CSV filtrado",
                       dff[cols].to_csv(index=False).encode("utf-8"),
                       "ofertas_filtradas.csv", "text/csv")

st.divider()
st.caption("Fuentes: (1) Ofertas de empleo IT · (2) Catálogo de tecnologías. "
           "Pipeline + ML + Dashboard — Segundo Parcial.")
