@echo off
REM ====================================================================
REM Start Docker Services for Bank Trading Management
REM ====================================================================

echo ============================================
echo Starting Docker Services
echo ============================================

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running!
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)

echo ✅ Docker is running
echo.

REM ====================================================================
REM Option 1: Use Cassandra on Windows (already running)
REM ====================================================================
echo NOTE: Cassandra is already running on Windows (port 9042)
echo Skipping Docker Cassandra to avoid port conflict.
echo.

REM ====================================================================
REM Start Redis
REM ====================================================================
echo [1/2] Starting Redis...
docker-compose -f docker-compose.redis.yml up -d

if %errorlevel% equ 0 (
    echo ✅ Redis started successfully
) else (
    echo ❌ Redis failed to start
)
echo.

REM ====================================================================
REM Start HDFS (NameNode + DataNode)
REM ====================================================================
echo [2/2] Starting HDFS...
docker-compose -f docker-compose.hdfs.yml up -d

if %errorlevel% equ 0 (
    echo ✅ HDFS started successfully
) else (
    echo ❌ HDFS failed to start
)
echo.

REM ====================================================================
REM Show Status
REM ====================================================================
echo ============================================
echo Docker Containers Status
echo ============================================
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | findstr "bt-"

echo.
echo ============================================
echo Service URLs
echo ============================================
echo Redis:           redis://localhost:6379
echo HDFS NameNode:   http://localhost:9870
echo HDFS DataNode:   http://localhost:9864
echo Cassandra:       localhost:9042 (Windows native)
echo.

echo ============================================
echo Next Steps
echo ============================================
echo 1. Wait 30-60 seconds for HDFS to fully start
echo 2. Check HDFS Web UI: http://localhost:9870
echo 3. Restart backend to connect to all services
echo.
echo To stop services: docker-compose -f docker-compose.redis.yml down
echo                   docker-compose -f docker-compose.hdfs.yml down
echo.

pause
