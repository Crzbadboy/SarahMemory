@echo off
TITLE SarahMemory AI Companion Launcher

:: =================================================================
::  CORRECTED SarahMemory Startup Script
::  This now points to the correct Python API server file.
:: =================================================================

echo Starting SarahMemory AI Companion...
echo.

:: --- [1/2] Start the Python Backend ---
echo Launching Python backend from C:\SarahMemory...
cd /d "C:\SarahMemory"

:: FIX: Changed 'app.py' to the correct server file
start "SarahMemory Backend" python SarahMemory-local_api_server.py

echo Backend process started in a new window.
echo.
timeout /t 2 >nul

:: --- [2/2] Start the React Frontend ---
echo Launching React frontend interface...

:: This assumes your React app is also in C:\SarahMemory
cd /d "C:\SarahMemory"

start "SarahMemory Frontend" npm run dev

echo Frontend process started in a new window.
echo.
echo =================================================================
echo  All systems go! Both processes are running.
echo  Your browser should open with the interface shortly.
echo  This window will close in 10 seconds.
echo =================================================================

timeout /t 10
