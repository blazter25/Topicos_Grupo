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

import argparse          # Lectura de argumentos de la línea de comandos.
import sys               # Acceso a la ruta de búsqueda de módulos (sys.path).
from pathlib import Path # Construcción de rutas portables.

# Se agrega la carpeta 'src' al path para poder importar los módulos internos
# (generar_datos, modelos_ml, pipeline) como si fueran librerías.
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import generar_datos  # noqa: E402  -> genera las fuentes de datos crudas.
import modelos_ml     # noqa: E402  -> entrena los modelos de ML.
import pipeline       # noqa: E402  -> ejecuta el ETL.


def main():
    """Orquesta la ejecución completa del proyecto en tres pasos."""
    # Definición y lectura del argumento opcional --skip-gen.
    parser = argparse.ArgumentParser(description="Pipeline Mercado IT Panamá")
    parser.add_argument("--skip-gen", action="store_true",
                        help="No regenerar las fuentes crudas")
    args = parser.parse_args()

    # Encabezado informativo en consola.
    print("=" * 64)
    print(" PROYECTO: Análisis del Mercado Laboral IT en Panamá")
    print(" Autor: Angel Martínez | Cédula: 8-893-602")
    print("=" * 64)

    # PASO 1: generar las fuentes crudas (se omite si se pasó --skip-gen).
    if not args.skip_gen:
        print("\n### PASO 1: Generación de fuentes crudas ###")
        generar_datos.main()

    # PASO 2: ejecutar el pipeline ETL sobre las fuentes crudas.
    print("\n### PASO 2: Pipeline ETL ###")
    pipeline.ejecutar_pipeline()

    # PASO 3: entrenar todos los modelos de Machine Learning.
    print("\n### PASO 3: Modelos de Machine Learning ###")
    modelos_ml.entrenar_todo()

    # Mensaje final con la instrucción para lanzar el dashboard.
    print("\n" + "=" * 64)
    print(" LISTO. Ejecute el dashboard con:")
    print("   streamlit run app/dashboard.py")
    print("=" * 64)


# Punto de entrada estándar: solo se ejecuta si el archivo se corre directo.
if __name__ == "__main__":
    main()
