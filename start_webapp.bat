@echo off
REM ===================================================================
REM MARKETING CAMPAIGN OPTIMIZER - WEB INTERFACE STARTER
REM ===================================================================
REM
REM This script starts the Streamlit web application for
REM marketing campaign optimization
REM
REM Requirements:
REM - Python 3.8+ installed
REM - requirements.txt packages installed
REM
REM ===================================================================

echo.
echo ========================================
echo  MARKETING CAMPAIGN OPTIMIZER
echo ========================================
echo.
echo Starting web interface...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not installed or not in PATH
    echo Please install Python 3.8+: https://python.org
    pause
    exit /b 1
)

REM Check if Streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found. Installing dependencies...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Dependency installation failed
        pause
        exit /b 1
    )
)

REM Start Streamlit app
echo Opening web interface in browser...
echo.
echo URL: http://localhost:8501
echo.
echo To exit: Press Ctrl+C
echo.

python -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost

pause