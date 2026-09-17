@echo off
setlocal EnableExtensions

title AcademiaSD - Krea-2 LoRA Trainer

cd /d "%~dp0"

set "BASE_DIR=%~dp0"
set "PYTHON_EXE="

echo.
echo ================================================================
echo        ACADEMIASD - KREA-2 LORA TRAINER
echo ================================================================
echo.
echo Trainer directory:
echo %BASE_DIR%
echo.

rem ================================================================
rem Find existing virtual environment
rem ================================================================

if exist "%BASE_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%BASE_DIR%.venv\Scripts\python.exe"
    goto :python_found
)

if exist "%BASE_DIR%venv\Scripts\python.exe" (
    set "PYTHON_EXE=%BASE_DIR%venv\Scripts\python.exe"
    goto :python_found
)

if exist "%BASE_DIR%env\Scripts\python.exe" (
    set "PYTHON_EXE=%BASE_DIR%env\Scripts\python.exe"
    goto :python_found
)

if exist "%BASE_DIR%..\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%BASE_DIR%..\venv\Scripts\python.exe"
    goto :python_found
)

if exist "%BASE_DIR%..\ .venv\Scripts\python.exe" (
    set "PYTHON_EXE=%BASE_DIR%..\ .venv\Scripts\python.exe"
    goto :python_found
)

echo.
echo [ERROR] Virtual environment not found.
echo.
echo The following paths were checked:
echo   %BASE_DIR%.venv\Scripts\python.exe
echo   %BASE_DIR%venv\Scripts\python.exe
echo   %BASE_DIR%env\Scripts\python.exe
echo   %BASE_DIR%..\venv\Scripts\python.exe
echo.
pause
exit /b 1


:python_found

echo Python environment found:
echo %PYTHON_EXE%
echo.

if not exist "%BASE_DIR%scripts\python\server.py" (
    echo [ERROR] scripts\python\server.py does not exist
    echo.
    pause
    exit /b 1
)

if not exist "%BASE_DIR%web\trainer_ui.html" (
    echo [ERROR] web\trainer_ui.html does not exist
    echo.
    pause
    exit /b 1
)

echo Checking Python...
"%PYTHON_EXE%" --version

if errorlevel 1 (
    echo.
    echo [ERROR] Cannot execute Python.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Starting web server...
echo ================================================================
echo.
echo Open in browser:
echo.
echo     http://127.0.0.1:5000
echo.
echo Close this window to stop the server.
echo.

"%PYTHON_EXE%" "%BASE_DIR%scripts\python\server.py"

echo.
echo ================================================================
echo Server has stopped.
echo ================================================================
pause

endlocal