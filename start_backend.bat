@echo off
REM ====================================================================
REM Start Backend with Auto Docker Services
REM ====================================================================

echo ============================================
echo Starting Bank Trading Backend
echo ============================================
echo.
echo This will automatically:
echo 1. Check and start Docker services (Redis, HDFS)
echo 2. Connect to Cassandra (Windows native)
echo 3. Start FastAPI backend server
echo.

REM Navigate to backend directory
cd /d "%~dp0BACKEND"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH!
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Start backend
echo Starting backend server...
echo NOTE: Docker services will auto-start if Docker Desktop is running
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

REM If we get here, backend was stopped
echo.
echo Backend stopped.
pause
