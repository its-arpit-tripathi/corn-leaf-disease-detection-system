@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: Leafcheck - Corn Leaf Disease Detection Backend Runner
:: ============================================================================
title Leafcheck Backend Server

:: Resolve paths regardless of where the script is executed
set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%backend" (
    set "BACKEND_DIR=%SCRIPT_DIR%backend"
) else (
    set "BACKEND_DIR=%SCRIPT_DIR%"
)

cd /d "%BACKEND_DIR%"

echo.
echo ========================================================
echo   Starting Leafcheck FastAPI Backend Server
echo ========================================================
echo   Backend Directory: %BACKEND_DIR%
echo.

:: Reduce TensorFlow oneDNN warning noise
if not defined TF_ENABLE_ONEDNN_OPTS (
    set "TF_ENABLE_ONEDNN_OPTS=0"
)

:: Check for virtual environment in backend\.venv
if exist "%BACKEND_DIR%\.venv\Scripts\activate.bat" (
    echo [OK] Found virtual environment in .venv
    call "%BACKEND_DIR%\.venv\Scripts\activate.bat"
) else if exist "%SCRIPT_DIR%\..\.venv\Scripts\activate.bat" (
    echo [OK] Found virtual environment in .venv
    call "%SCRIPT_DIR%\..\.venv\Scripts\activate.bat"
) else (
    echo [!] No virtual environment found.
    echo [*] Checking for system Python...
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python is not installed or not found in system PATH.
        echo Please install Python 3.10+ and re-run this script.
        goto :error
    )
    echo [*] Creating virtual environment at %BACKEND_DIR%\.venv
    python -m venv "%BACKEND_DIR%\.venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        goto :error
    )
    call "%BACKEND_DIR%\.venv\Scripts\activate.bat"
    echo [*] Installing dependencies from requirements.txt
    pip install -r "%BACKEND_DIR%\requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        goto :error
    )
)

:: Verify uvicorn is installed in the active environment
where uvicorn >nul 2>&1
if errorlevel 1 (
    echo [!] uvicorn not found in current environment.
    echo [*] Installing requirements from %BACKEND_DIR%\requirements.txt
    pip install -r "%BACKEND_DIR%\requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements.
        goto :error
    )
)

echo.
echo ========================================================
echo   Backend is launching on http://127.0.0.1:8000
echo   - Interactive API Docs: http://127.0.0.1:8000/docs
echo   - Alternative ReDoc:    http://127.0.0.1:8000/redoc
echo   - Health check:         http://127.0.0.1:8000/api/health
echo ========================================================
echo   Press Ctrl+C to stop the server at any time.
echo.

uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

if errorlevel 1 (
    echo.
    echo [ERROR] Backend server exited with error code %errorlevel%.
    goto :error
)

goto :end

:error
echo.
echo Press any key to exit...
pause >nul

:end
