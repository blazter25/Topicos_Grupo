@echo off
REM ============================================================
REM  Iniciar Dashboard - Parcial 2 Mercado IT Panama
REM  Angel Martinez - Cedula 8-893-602
REM  Doble clic para abrir el dashboard de Streamlit
REM ============================================================
cd /d "%~dp0"
echo Iniciando dashboard... se abrira en tu navegador.
echo Para detenerlo, cierra esta ventana o presiona Ctrl+C.
echo.
python -m streamlit run app/dashboard.py
pause
