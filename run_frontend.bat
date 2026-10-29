@echo off
setlocal

title Leafcheck Frontend

set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%frontend" (
    set "FRONTEND_DIR=%SCRIPT_DIR%frontend"
) else (
    set "FRONTEND_DIR=%SCRIPT_DIR%"
)

cd /d "%FRONTEND_DIR%"

echo.
echo ========================================================
echo   Starting Leafcheck Next.js Frontend
echo ========================================================
echo   Frontend Directory: %FRONTEND_DIR%
echo.

if not exist "node_modules" (
    echo [*] Installing frontend npm packages...
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        pause
        exit /b 1
    )
)

if not exist ".env.local" (
    if exist ".env.example" (
        echo [*] Creating .env.local from .env.example
        copy .env.example .env.local >nul
    )
)

echo.
echo ========================================================
echo   Frontend is launching on http://localhost:3000
echo ========================================================
echo   Press Ctrl+C to stop the frontend at any time.
echo.

call npm run dev
if errorlevel 1 (
    echo.
    echo [ERROR] Frontend server stopped with error code %errorlevel%.
    pause
)

