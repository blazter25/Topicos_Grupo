"""
Generación de las fuentes de datos crudas del proyecto.
====================================================================
Proyecto: Análisis del Mercado Laboral IT en Panamá
Curso:    Gestión de la Información - UTP, I Semestre 2026
Autor:    Angel Martínez  -  Cédula: 8-893-602

Este módulo genera las DOS fuentes de datos crudas que alimentan el
pipeline. En un entorno con acceso a internet, la Fuente 1 provendría
del web scraping de portales de empleo (Konzerta, encuentra24,
LinkedIn) y la Fuente 2 de un catálogo público de tecnologías. Para
garantizar que el proyecto sea 100% reproducible y ejecutable sin
conexión, aquí se generan datos sintéticos REALISTAS calibrados con
las proporciones reales del mercado IT panameño.

  - Fuente 1 (CSV) : ofertas_empleo_it.csv  -> "ofertas scrapeadas"
  - Fuente 2 (JSON): catalogo_tecnologias.json -> catálogo de skills
====================================================================
"""

from __future__ import annotations

import json                              # Para escribir el catálogo en JSON.
import random                            # Generación de valores aleatorios.
from datetime import datetime, timedelta # Manejo de fechas de publicación.
from pathlib import Path                 # Rutas portables.

import numpy as np   # Ruido estadístico y distribuciones (normal, beta...).
import pandas as pd  # Construcción del DataFrame de ofertas.

# Semilla fija: garantiza que cada ejecución produzca los MISMOS datos
# (reproducibilidad, requisito clave en proyectos de datos).
SEMILLA = 42
random.seed(SEMILLA)
np.random.seed(SEMILLA)

# Carpeta de salida de los datos crudos (se crea si no existe).
RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ / "data" / "raw"
DIR_RAW.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Catálogo de tecnologías (Fuente 2)
# Cada tecnología tiene categoría, índice de demanda (0-100) y si es emergente.
# 'premium' = sobreprecio salarial aproximado que aporta dominar esa skill.
# ---------------------------------------------------------------------------
CATALOGO = {
    # Lenguajes
    "Python":      {"categoria": "Lenguaje",  "indice_demanda": 92, "emergente": False, "premium": 320},
    "JavaScript":  {"categoria": "Lenguaje",  "indice_demanda": 88, "emergente": False, "premium": 260},
    "Java":        {"categoria": "Lenguaje",  "indice_demanda": 80, "emergente": False, "premium": 300},
    "C#":          {"categoria": "Lenguaje",  "indice_demanda": 70, "emergente": False, "premium": 290},
    "PHP":         {"categoria": "Lenguaje",  "indice_demanda": 55, "emergente": False, "premium": 150},
    "TypeScript":  {"categoria": "Lenguaje",  "indice_demanda": 78, "emergente": True,  "premium": 300},
    "Go":          {"categoria": "Lenguaje",  "indice_demanda": 45, "emergente": True,  "premium": 420},
    "Rust":        {"categoria": "Lenguaje",  "indice_demanda": 28, "emergente": True,  "premium": 460},
    # Frontend
    "React":       {"categoria": "Frontend",  "indice_demanda": 85, "emergente": False, "premium": 300},
    "Angular":     {"categoria": "Frontend",  "indice_demanda": 60, "emergente": False, "premium": 270},
    "Vue":         {"categoria": "Frontend",  "indice_demanda": 42, "emergente": True,  "premium": 250},
    # Backend
    "Node.js":     {"categoria": "Backend",   "indice_demanda": 76, "emergente": False, "premium": 290},
    ".NET":        {"categoria": "Backend",   "indice_demanda": 64, "emergente": False, "premium": 300},
    "Spring":      {"categoria": "Backend",   "indice_demanda": 50, "emergente": False, "premium": 320},
    # Cloud
    "AWS":         {"categoria": "Cloud",     "indice_demanda": 82, "emergente": False, "premium": 480},
    "Azure":       {"categoria": "Cloud",     "indice_demanda": 74, "emergente": False, "premium": 450},
    "GCP":         {"categoria": "Cloud",     "indice_demanda": 40, "emergente": True,  "premium": 440},
    # DevOps
    "Docker":      {"categoria": "DevOps",    "indice_demanda": 72, "emergente": False, "premium": 360},
    "Kubernetes":  {"categoria": "DevOps",    "indice_demanda": 48, "emergente": True,  "premium": 520},
    "Git":         {"categoria": "DevOps",    "indice_demanda": 90, "emergente": False, "premium": 80},
    # Data / IA
    "SQL":         {"categoria": "Datos",     "indice_demanda": 89, "emergente": False, "premium": 200},
    "Power BI":    {"categoria": "Datos",     "indice_demanda": 66, "emergente": False, "premium": 220},
    "Spark":       {"categoria": "Datos",     "indice_demanda": 35, "emergente": True,  "premium": 430},
    "Machine Learning": {"categoria": "IA",   "indice_demanda": 58, "emergente": True,  "premium": 500},
    "LLM/IA Generativa": {"categoria": "IA",  "indice_demanda": 52, "emergente": True,  "premium": 560},
    # Bases de datos
    "PostgreSQL":  {"categoria": "Base de Datos", "indice_demanda": 63, "emergente": False, "premium": 210},
    "MongoDB":     {"categoria": "Base de Datos", "indice_demanda": 47, "emergente": False, "premium": 230},
    "Oracle":      {"categoria": "Base de Datos", "indice_demanda": 44, "emergente": False, "premium": 260},
}

# Mapa de rol -> conjunto de tecnologías típicas de ese puesto.
CATEGORIA_POR_ROL = {
    "Desarrollador Frontend":   ["JavaScript", "TypeScript", "React", "Angular", "Vue", "Git"],
    "Desarrollador Backend":    ["Python", "Java", "C#", "Node.js", ".NET", "Spring", "SQL", "Git"],
    "Desarrollador Full Stack": ["JavaScript", "TypeScript", "React", "Node.js", "Python", "SQL", "Docker", "Git"],
    "Ingeniero de Datos":       ["Python", "SQL", "Spark", "AWS", "PostgreSQL", "Power BI"],
    "Científico de Datos":      ["Python", "Machine Learning", "SQL", "LLM/IA Generativa", "Power BI"],
    "Ingeniero DevOps":         ["Docker", "Kubernetes", "AWS", "Azure", "GCP", "Git"],
    "Ingeniero Cloud":          ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Python"],
    "Administrador de BD":      ["SQL", "Oracle", "PostgreSQL", "MongoDB", "Python"],
    "Analista de BI":           ["Power BI", "SQL", "Python", "PostgreSQL"],
    "Ingeniero de IA/ML":       ["Python", "Machine Learning", "LLM/IA Generativa", "AWS", "Spark"],
}

# Catálogos de apoyo para dar realismo a las ofertas generadas.
EMPRESAS = [
    "Copa Airlines", "Banco General", "Banistmo", "Cable & Wireless", "Dell Panamá",
    "Globant", "TechMakers", "MultiBank", "Caja de Ahorros", "Telered",
    "Soluciones IT SA", "Panama Digital", "GBM Panamá", "Procesos y Sistemas",
    "Avianca Tech", "BAC Credomatic", "Innovatec", "DataPro Panamá", "Nearshore Devs",
]

FUENTES = ["Konzerta", "encuentra24", "LinkedIn", "Computrabajo"]
PROVINCIAS = ["Panamá", "Panamá Oeste", "Colón", "Chiriquí", "Coclé", "Herrera", "Los Santos", "Veraguas"]
# Pesos de provincia: reflejan la concentración real de empleo en la capital.
PESOS_PROVINCIA = [0.58, 0.12, 0.08, 0.09, 0.04, 0.03, 0.03, 0.03]
MODALIDADES = ["Presencial", "Híbrido", "Remoto"]
NIVELES = ["Junior", "Semi Senior", "Senior", "Lead"]
CONTRATOS = ["Permanente", "Temporal", "Por Proyecto", "Freelance"]

# Salario base mensual (USD/B.) por nivel - referencia mercado IT Panamá.
SALARIO_BASE = {"Junior": 900, "Semi Senior": 1500, "Senior": 2400, "Lead": 3200}
# Factor multiplicador del salario según la modalidad (remoto paga más).
FACTOR_MODALIDAD = {"Presencial": 1.0, "Híbrido": 1.07, "Remoto": 1.15}


def _texto_descripcion(rol: str, nivel: str, skills: list[str]) -> str:
    """Arma un texto de descripción genérico pero realista para la oferta."""
    return (
        f"Buscamos {rol} nivel {nivel} para integrarse a nuestro equipo. "
        f"Experiencia comprobable en {', '.join(skills[:4])}. "
        f"Ofrecemos crecimiento profesional y ambiente colaborativo."
    )


def generar_ofertas(n: int = 1300) -> pd.DataFrame:
    """Genera n ofertas de empleo IT con dispersión temporal de 18 meses."""
    # Ventana temporal: 540 días (~18 meses) hasta el 30/04/2026.
    fecha_fin = datetime(2026, 4, 30)
    fecha_ini = fecha_fin - timedelta(days=540)
    registros = []

    for i in range(1, n + 1):
        # Atributos categóricos aleatorios (con pesos realistas).
        rol = random.choice(list(CATEGORIA_POR_ROL.keys()))
        nivel = random.choices(NIVELES, weights=[0.34, 0.30, 0.26, 0.10])[0]
        modalidad = random.choices(MODALIDADES, weights=[0.45, 0.33, 0.22])[0]
        provincia = random.choices(PROVINCIAS, weights=PESOS_PROVINCIA)[0]

        # Factor temporal: las ofertas se generan en orden cronológico aprox.,
        # por lo que i/n aproxima el avance del tiempo (0 = pasado, 1 = reciente)
        sesgo = i / n

        # Selección de habilidades del rol (entre 3 y 6).
        pool = CATEGORIA_POR_ROL[rol]
        k = min(len(pool), random.randint(3, 6))
        skills = random.sample(pool, k)
        # Probabilidad creciente en el tiempo de añadir skills emergentes
        # (15% al inicio del periodo -> 65% en los meses recientes).
        # Esto reproduce la adopción real de tecnologías emergentes.
        emergentes = [s for s, v in CATALOGO.items() if v["emergente"]]
        prob_emergente = 0.15 + 0.50 * sesgo
        if random.random() < prob_emergente:
            skills.append(random.choice(emergentes))
        if random.random() < prob_emergente * 0.5:
            skills.append(random.choice(emergentes))
        skills = sorted(set(skills))   # Sin duplicados y ordenadas.

        # ----- Salario simulado de forma realista -----
        base = SALARIO_BASE[nivel]
        # El sobreprecio de las skills suma según el catálogo (premium).
        premium_skills = sum(CATALOGO[s]["premium"] for s in skills)
        sal_centro = (base + 0.55 * premium_skills) * FACTOR_MODALIDAD[modalidad]
        # Sesgo geográfico (la capital paga más).
        if provincia in ("Panamá", "Panamá Oeste"):
            sal_centro *= 1.06
        else:
            sal_centro *= 0.93
        # Ruido de mercado (variabilidad aleatoria del ±8%).
        sal_centro *= np.random.normal(1.0, 0.08)
        sal_centro = max(750, sal_centro)   # Piso salarial mínimo.

        # A partir del centro se arma un rango min-max redondeado a 50.
        rango = sal_centro * np.random.uniform(0.10, 0.22)
        salario_min = round((sal_centro - rango) / 50) * 50
        salario_max = round((sal_centro + rango) / 50) * 50

        # Tendencia temporal: las ofertas se concentran más en meses recientes.
        dias = int(np.clip(np.random.beta(1.6, 1.0) * 540 * (0.4 + 0.6 * sesgo), 0, 540))
        fecha = fecha_ini + timedelta(days=dias)

        # Datos faltantes intencionales para ejercitar el preprocesamiento.
        if random.random() < 0.12:
            salario_min = ""        # ~12% de ofertas sin salario mínimo.
        if random.random() < 0.07:
            salario_max = ""        # ~7% sin salario máximo.
        empresa = random.choice(EMPRESAS)
        if random.random() < 0.05:
            empresa = ""            # ~5% de ofertas confidenciales.

        # Registro final de la oferta como diccionario.
        registros.append({
            "id_oferta": f"OF-{i:05d}",
            "titulo": rol,
            "empresa": empresa,
            "fuente": random.choices(FUENTES, weights=[0.34, 0.26, 0.25, 0.15])[0],
            "fecha_publicacion": fecha.strftime("%Y-%m-%d"),
            "provincia": provincia,
            "modalidad": modalidad,
            "nivel": nivel,
            "tipo_contrato": random.choices(CONTRATOS, weights=[0.62, 0.12, 0.16, 0.10])[0],
            "salario_min": salario_min,
            "salario_max": salario_max,
            "moneda": "USD",
            "habilidades": ", ".join(skills),
            "descripcion": _texto_descripcion(rol, nivel, skills),
        })

    df = pd.DataFrame(registros)
    # Duplicados artificiales (mismo aviso re-publicado) para limpiar luego.
    dups = df.sample(frac=0.03, random_state=SEMILLA).copy()
    df = pd.concat([df, dups], ignore_index=True)
    return df


def main() -> None:
    """Genera y guarda en disco las dos fuentes crudas del proyecto."""
    print(">> Generando Fuente 1: ofertas de empleo IT (simulación de scraping)...")
    df = generar_ofertas()
    ruta_csv = DIR_RAW / "ofertas_empleo_it.csv"
    df.to_csv(ruta_csv, index=False, encoding="utf-8")
    print(f"   {len(df)} filas -> {ruta_csv}")

    print(">> Generando Fuente 2: catálogo de tecnologías (JSON)...")
    ruta_json = DIR_RAW / "catalogo_tecnologias.json"
    # Se transforma el diccionario CATALOGO en una lista de objetos JSON.
    catalogo_lista = [
        {"tecnologia": k, **v} for k, v in CATALOGO.items()
    ]
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(catalogo_lista, f, ensure_ascii=False, indent=2)
    print(f"   {len(catalogo_lista)} tecnologías -> {ruta_json}")
    print(">> Fuentes crudas generadas correctamente.")


# Punto de entrada: permite generar las fuentes de forma independiente.
if __name__ == "__main__":
    main()
