@echo off
setlocal EnableExtensions EnableDelayedExpansion

title Krea-2 Trainer Venv Installer - NVIDIA

:: ========================================================
:: CONFIGURATION
:: ========================================================

set "BASE_DIR=%~dp0"
set "PYTHON_INSTALLER=%BASE_DIR%python-3.13.1-amd64.exe"
set "PYTHON_EXE="

echo ========================================================
echo   KREA-2 LORA TRAINER INSTALLER
echo   Python 3.13.1 + PyTorch CUDA Env
echo   Compatible with modern NVIDIA GPUs
echo ========================================================
echo.
echo Installer folder:
echo %BASE_DIR%
echo.

echo [1/8] Checking Python 3.13...

where python >nul 2>&1
if errorlevel 1 goto FIND_LOCAL_PYTHON

echo Python found in PATH:
python --version
echo.
echo Checking compatible version...

python -c "import sys; exit(0 if sys.version_info[:2] == (3,13) else 1)" >nul 2>&1
if errorlevel 1 goto NOT_313_IN_PATH

echo [OK] Python 3.13 detected.
set "PYTHON_EXE=python"
goto PYTHON_OK

:NOT_313_IN_PATH
echo [WARNING] Python found but it is not 3.13.

:: --------------------------------------------------------
:: Search Python 3.13 in common paths
:: --------------------------------------------------------

:FIND_LOCAL_PYTHON
echo.
echo Searching for existing Python 3.13 installations...

if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
    echo [OK] Found:
    echo %PYTHON_EXE%
    goto PYTHON_OK
)

if exist "%ProgramFiles%\Python313\python.exe" (
    set "PYTHON_EXE=%ProgramFiles%\Python313\python.exe"
    echo [OK] Found:
    echo %PYTHON_EXE%
    goto PYTHON_OK
)

if exist "%ProgramFiles(x86)%\Python313\python.exe" (
    set "PYTHON_EXE=%ProgramFiles(x86)%\Python313\python.exe"
    echo [OK] Found:
    echo %PYTHON_EXE%
    goto PYTHON_OK
)

:: ========================================================
:: INSTALL PYTHON 3.13.1
:: ========================================================

echo.
echo [INFO] Python 3.13 is not available on the system.
echo.

if exist "%PYTHON_INSTALLER%" goto DO_INSTALL

echo ========================================================
echo   ANALYZING AND STARTING AUTO DOWNLOAD
echo ========================================================
echo.
echo Analyzing system architecture...

rem PyTorch requires a 64-bit operating system (AMD64 / x64)
set "ARCH=amd64"
if "%PROCESSOR_ARCHITECTURE%"=="x86" (
    if not defined PROCESSOR_ARCHITEW6432 (
        echo [ERROR] 32-bit system detected.
        echo PyTorch and CUDA strictly require 64 bits.
        pause
        exit /b 1
    )
)

echo Compatible architecture detected: 64 bits %PROCESSOR_ARCHITECTURE%
echo Downloading Python 3.13.1 x64 from official site...
echo.

rem Download attempt 1: curl
curl -L "https://www.python.org/ftp/python/3.13.1/python-3.13.1-amd64.exe" -o "%PYTHON_INSTALLER%"

if exist "%PYTHON_INSTALLER%" goto DOWNLOAD_OK

rem Download attempt 2: PowerShell
echo [INFO] curl failed. Trying with PowerShell...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.13.1/python-3.13.1-amd64.exe' -OutFile '%PYTHON_INSTALLER%'" >nul 2>&1

if not exist "%PYTHON_INSTALLER%" (
    echo ========================================================
    echo [ERROR] Could not download Python.
    echo ========================================================
    echo Automatic download failed.
    echo Download manually from browser:
    echo https://www.python.org/ftp/python/3.13.1/python-3.13.1-amd64.exe
    echo Save it as "python-3.13.1-amd64.exe" next to this BAT file.
    echo.
    pause
    exit /b 1
)

:DOWNLOAD_OK
echo [OK] Successfully downloaded: %PYTHON_INSTALLER%
echo.

:DO_INSTALL
echo ========================================================
echo   INSTALLING PYTHON 3.13.1
echo ========================================================
echo.
echo Installer will run in the background.
echo Installing for current user and adding to PATH.
echo This may take a few minutes. Please wait...
echo.

"%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 Include_test=0

if errorlevel 1 (
    echo.
    echo [ERROR] Could not install Python 3.13.1.
    echo.
    pause
    exit /b 1
)

echo [OK] Python installation finished.
echo.

:: --------------------------------------------------------
:: Search for installed Python after installation
:: --------------------------------------------------------

if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
    goto PYTHON_OK
)

if exist "%ProgramFiles%\Python313\python.exe" (
    set "PYTHON_EXE=%ProgramFiles%\Python313\python.exe"
    goto PYTHON_OK
)

for /f "delims=" %%A in ('where python 2^>nul') do (
    set "PYTHON_EXE=%%A"
    goto PYTHON_OK
)

echo.
echo [ERROR] Python installed but not located.
echo Restart Windows and run again.
echo.
pause
exit /b 1

:: ========================================================
:: PYTHON OK
:: ========================================================

:PYTHON_OK

echo.
echo ========================================================
echo   PYTHON DETECTED
echo ========================================================
echo.
echo Executable:
echo %PYTHON_EXE%
echo.

"%PYTHON_EXE%" --version

if errorlevel 1 (
    echo.
    echo [ERROR] Python cannot run properly.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import sys; exit(0 if sys.version_info[:2] == (3,13) else 1)" >nul 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] Python version is not compatible.
    echo Python 3.13.x is required.
    echo.
    pause
    exit /b 1
)

echo [OK] Compatible Python 3.13 detected.

:: ========================================================
:: 2/8 - CHECK VENV
:: ========================================================

echo.
echo [2/8] Checking venv module...

"%PYTHON_EXE%" -m venv --help >nul 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] Python venv module is not available.
    echo.
    pause
    exit /b 1
)

echo [OK] venv module available.

:: ========================================================
:: 3/8 - CHECK NVIDIA GPU
:: ========================================================

echo.
echo [3/8] Checking NVIDIA compatibility...
echo GPU detection will occur after PyTorch installation.
echo.
echo [OK] Continuing installation.

echo.
echo [4/8] Preparing clean virtual environment...

if exist "%BASE_DIR%venv" (
    echo.
    echo Previous virtual environment detected. Removing...

    rmdir /s /q "%BASE_DIR%venv"

    if exist "%BASE_DIR%venv" (
        echo.
        echo [ERROR] Could not remove previous venv.
        echo Close any running program:
        echo     venv\Scripts\python.exe
        echo.
        pause
        exit /b 1
    )
)

echo.
echo Creating new virtual environment...

"%PYTHON_EXE%" -m venv "%BASE_DIR%venv"

if errorlevel 1 (
    echo.
    echo [ERROR] Could not create virtual environment.
    pause
    exit /b 1
)

:: ========================================================
:: 5/8 - ACTIVATE VENV
:: ========================================================

echo.
echo [5/8] Activating virtual environment...

call "%BASE_DIR%venv\Scripts\activate.bat"

if errorlevel 1 (
    echo.
    echo [ERROR] Could not activate virtual environment.
    pause
    exit /b 1
)

set "VENV_PYTHON=%BASE_DIR%venv\Scripts\python.exe"

echo.
echo Virtual environment Python:
echo %VENV_PYTHON%

"%VENV_PYTHON%" --version

:: ========================================================
:: 6/8 - UPGRADE TOOLS
:: ========================================================

echo.
echo [6/8] Upgrading pip, setuptools, and wheel...

"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel

if errorlevel 1 (
    echo.
    echo [ERROR] Could not upgrade Python tools.
    pause
    exit /b 1
)

:: ========================================================
:: 7/8 - INSTALL PYTORCH
:: ========================================================

echo.
echo [7/8] Installing PyTorch with CUDA support...
echo.
echo ========================================================
echo   IMPORTANT
echo ========================================================
echo.
echo PyTorch with CUDA 13.0 will be installed.
echo For modern NVIDIA GPUs (including RTX 50xx / Blackwell).
echo Installation may take minutes.
echo ========================================================
echo.

"%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

if errorlevel 1 (
    echo.
    echo [ERROR] Could not install PyTorch.
    pause
    exit /b 1
)

:: ========================================================
:: INSTALL KREA-2 DEPENDENCIES
:: ========================================================

echo.
echo Installing Diffusers, Transformers, PEFT, Accelerate, Safetensors, and Hugging Face Hub...

"%VENV_PYTHON%" -m pip install diffusers transformers peft accelerate safetensors huggingface_hub

if errorlevel 1 (
    echo.
    echo [ERROR] Error installing Diffusers dependencies.
    pause
    exit /b 1
)

echo.
echo Installing BitsAndBytes and utilities...

"%VENV_PYTHON%" -m pip install bitsandbytes sentencepiece protobuf

if errorlevel 1 (
    echo.
    echo [ERROR] Error installing BitsAndBytes or utilities.
    pause
    exit /b 1
)

:: Dataset curation (0_curate_dataset.py): facial identity scoring
:: with ArcFace. CPU-only; the model (~300 MB) is downloaded on first use.
echo.
echo Installing InsightFace for dataset curation...

"%VENV_PYTHON%" -m pip install insightface onnxruntime opencv-python

if errorlevel 1 (
    echo.
    echo [WARNING] InsightFace could not be installed; dataset curation will be unavailable.
    echo The rest of the trainer still works.
)

:: ========================================================
:: 8/8 - FINAL CHECK
:: ========================================================

echo.
echo [8/8] Verifying installation...
echo.

:: --------------------------------------------------------
:: PYTORCH
:: --------------------------------------------------------

echo ========================================================
echo PyTorch
echo ========================================================

"%VENV_PYTHON%" -c "import torch; print('PyTorch:', torch.__version__); print('Compiled CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPUs detected:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
if errorlevel 1 (
    echo.
    echo [ERROR] PyTorch failed to initialize.
    pause
    exit /b 1
)

:: --------------------------------------------------------
:: DIFFUSERS
:: --------------------------------------------------------

echo.
echo ========================================================
echo Diffusers
echo ========================================================

"%VENV_PYTHON%" -c "import diffusers; print('Diffusers:', diffusers.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] Diffusers is not installed properly.
    pause
    exit /b 1
)

echo [OK] Diffusers loaded successfully.

:: --------------------------------------------------------
:: HUGGING FACE HUB
:: --------------------------------------------------------

echo.
echo ========================================================
echo Hugging Face Hub
echo ========================================================

"%VENV_PYTHON%" -c "import huggingface_hub; print('Hugging Face Hub:', huggingface_hub.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] Hugging Face Hub is not installed properly.
    pause
    exit /b 1
)

echo [OK] Hugging Face Hub loaded successfully.

:: --------------------------------------------------------
:: TRANSFORMERS
:: --------------------------------------------------------

echo.
echo ========================================================
echo Transformers
echo ========================================================

"%VENV_PYTHON%" -c "import transformers; print('Transformers:', transformers.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] Transformers is not installed properly.
    pause
    exit /b 1
)

echo [OK] Transformers loaded successfully.

:: --------------------------------------------------------
:: PEFT
:: --------------------------------------------------------

echo.
echo ========================================================
echo PEFT
echo ========================================================

"%VENV_PYTHON%" -c "import peft; print('PEFT:', peft.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] PEFT is not installed properly.
    pause
    exit /b 1
)

echo [OK] PEFT loaded successfully.

:: --------------------------------------------------------
:: ACCELERATE
:: --------------------------------------------------------

echo.
echo ========================================================
echo Accelerate
echo ========================================================

"%VENV_PYTHON%" -c "import accelerate; print('Accelerate:', accelerate.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] Accelerate is not installed properly.
    pause
    exit /b 1
)

echo [OK] Accelerate loaded successfully.

:: --------------------------------------------------------
:: BITSANDBYTES
:: --------------------------------------------------------

echo.
echo ========================================================
echo BitsAndBytes
echo ========================================================

"%VENV_PYTHON%" -c "import bitsandbytes as bnb; print('BitsAndBytes:', bnb.__version__)"

if errorlevel 1 (
    echo.
    echo [WARNING] BitsAndBytes failed to initialize.
    echo This may affect 8-bit optimizers.
    echo.
) else (
    echo [OK] BitsAndBytes loaded successfully.
)

:: --------------------------------------------------------
:: CHECK CUDA
:: --------------------------------------------------------

echo.
echo ========================================================
echo   CHECK RESULT
echo ========================================================
echo.

"%VENV_PYTHON%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

if errorlevel 1 (
    echo [WARNING] PyTorch does NOT detect a CUDA GPU.
    echo.
    echo Possible causes:
    echo - Outdated NVIDIA driver.
    echo - Incorrect PyTorch installation.
    echo - GPU driver issue.
    echo.
    echo Run: nvidia-smi
    echo.
) else (
    echo [OK] PyTorch correctly detects NVIDIA GPU.
)

:: ========================================================
:: FINAL
:: ========================================================

echo.
echo ========================================================
echo   INSTALLATION COMPLETED
echo ========================================================
echo.
echo Virtual environment "venv" created.
echo.
echo Python used:
echo %PYTHON_EXE%
echo.
echo Virtual env Python:
echo %VENV_PYTHON%
echo.
echo Python version:
"%VENV_PYTHON%" --version
echo.
echo GPU detected by PyTorch:
"%VENV_PYTHON%" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT DETECTED')"
echo.
echo PyTorch CUDA version:
"%VENV_PYTHON%" -c "import torch; print(torch.version.cuda)"
echo.
echo Diffusers version:
"%VENV_PYTHON%" -c "import diffusers; print(diffusers.__version__)"
echo.
echo Hugging Face Hub version:
"%VENV_PYTHON%" -c "import huggingface_hub; print(huggingface_hub.__version__)"
echo.
echo ========================================================
echo.
echo Environment is ready to run Krea-2 Trainer.
echo.
pause

endlocal