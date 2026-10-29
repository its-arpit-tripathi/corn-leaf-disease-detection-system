@echo off
setlocal

title Leafcheck App Launcher

echo ========================================================
echo   Starting Leafcheck Application (Backend + Frontend)
echo ========================================================
echo.

start "Leafcheck Backend Server" cmd /k "%~dp0run_backend.bat"
start "Leafcheck Frontend Next.js" cmd /k "%~dp0run_frontend.bat"

echo Both servers have been launched in dedicated terminal windows:
echo   - Backend:  http://127.0.0.1:8000 (Docs: http://127.0.0.1:8000/docs)
echo   - Frontend: http://localhost:3000
echo.

