"""
Orquestador del proyecto - Segundo Parcial.
====================================================================
Proyecto: Análisis del Mercado Laboral IT en Panamá
Curso:    Gestión de la Información - UTP, I Semestre 2026
Autor:    Angel Martínez  -  Cédula: 8-893-602

Ejecuta de extremo a extremo:
    1. Generación de las fuentes crudas
    2. Pipeline ETL (ingesta -> limpieza -> transformación -> carga)
    3. Entrenamiento de los modelos de ML

Uso:
    python main.py            # ejecuta todo
    python main.py --skip-gen # no regenera datos crudos
====================================================================
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import generar_datos  # noqa: E402
import modelos_ml     # noqa: E402
import pipeline       # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Pipeline Mercado IT Panamá")
    parser.add_argument("--skip-gen", action="store_true",
                        help="No regenerar las fuentes crudas")
    args = parser.parse_args()

    print("=" * 64)
    print(" PROYECTO: Análisis del Mercado Laboral IT en Panamá")
    print(" Autor: Angel Martínez | Cédula: 8-893-602")
    print("=" * 64)

    if not args.skip_gen:
        print("\n### PASO 1: Generación de fuentes crudas ###")
        generar_datos.main()

    print("\n### PASO 2: Pipeline ETL ###")
    pipeline.ejecutar_pipeline()

    print("\n### PASO 3: Modelos de Machine Learning ###")
    modelos_ml.entrenar_todo()

    print("\n" + "=" * 64)
    print(" LISTO. Ejecute el dashboard con:")
    print("   streamlit run app/dashboard.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
