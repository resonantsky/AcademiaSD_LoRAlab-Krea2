@echo off
title Triton and SDNQ Installer by Academia SD
color 0A

:: This script lives in scripts\batch\, so anchor the working directory to the
:: project root (two levels up) before resolving venv\ and temp files.
cd /d "%~dp0..\.."

:: Define the portable Python path for ComfyUI
set PYTHON_EXE=venv\scripts\python.exe

:: Redirect Triton and PyTorch Inductor cache directories to .cache inside project root
set "TRITON_CACHE_DIR=%CD%\.cache\triton"
set "TORCHINDUCTOR_CACHE_DIR=%CD%\.cache\torchinductor"

:: Ensure cache directories exist
if not exist "%TRITON_CACHE_DIR%" mkdir "%TRITON_CACHE_DIR%"
if not exist "%TORCHINDUCTOR_CACHE_DIR%" mkdir "%TORCHINDUCTOR_CACHE_DIR%"

echo =======================================================
echo   Triton ^& SDNQ Installer 
echo   Created by: Academia SD
echo =======================================================
echo.

:: Check if we are in the correct folder
if not exist "%PYTHON_EXE%" (
    color 0C
    echo [ERROR] %PYTHON_EXE% not found.
    echo Please place this .bat file in the root folder.
    pause
    exit /b
)

echo =======================================================
echo    1. INSTALLING TRITON 3.7.1
echo =======================================================
"%PYTHON_EXE%" -m pip install -U triton-windows==3.7.1

echo.
echo =======================================================
echo    2. ANALYZING ENVIRONMENT FOR SDNQ
echo =======================================================

:: Create a temporary Python script for advanced logic
echo import sys, subprocess > temp_detector.py
echo try: >> temp_detector.py
echo     import torch >> temp_detector.py
echo except ImportError: >> temp_detector.py
echo     print("[ERROR] PyTorch is not installed. Cannot continue.") >> temp_detector.py
echo     sys.exit(1) >> temp_detector.py
echo. >> temp_detector.py
echo py_ver = f"{sys.version_info.major}.{sys.version_info.minor}" >> temp_detector.py
echo t_ver = torch.__version__ >> temp_detector.py
echo c_ver = torch.version.cuda or "None" >> temp_detector.py
echo print(f"[INFO] Detected Environment:") >> temp_detector.py
echo print(f"       - Python: {py_ver}") >> temp_detector.py
echo print(f"       - PyTorch: {t_ver}") >> temp_detector.py
echo print(f"       - CUDA: {c_ver}\n") >> temp_detector.py
echo. >> temp_detector.py
echo # Extract minor version of PyTorch (e.g., if 2.9.0, extract 9) >> temp_detector.py
echo t_minor = 0 >> temp_detector.py
echo if t_ver.startswith("2."): >> temp_detector.py
echo     try: t_minor = int(t_ver.split(".")[1]) >> temp_detector.py
echo     except: pass >> temp_detector.py
echo. >> temp_detector.py
echo print("[INFO] Installing SDNQ package via pip...") >> temp_detector.py
echo subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "sdnq"]) >> temp_detector.py

:: Execute the temporary script
"%PYTHON_EXE%" temp_detector.py

:: Clean up by deleting the temp file
del temp_detector.py

echo.
echo =======================================================
echo    PROCESS COMPLETED
echo =======================================================
pause