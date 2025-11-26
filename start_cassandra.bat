@echo off
REM ====================================================================
REM Start Cassandra Native (Windows Installation)
REM ====================================================================

echo ============================================
echo Starting Cassandra (Native - Java 8)
echo ============================================
echo.

REM 1) Set Java 8 for this session
echo [1/3] Setting Java 8...
for /d %%G in ("%ProgramFiles%\Eclipse Adoptium\jdk-8*") do set "JAVA_HOME=%%G"
set "PATH=%JAVA_HOME%\bin;%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\WindowsPowerShell\v1.0"

if not defined JAVA_HOME (
    echo ERROR: Java 8 not found!
    echo Please install Eclipse Adoptium JDK 8
    pause
    exit /b 1
)

echo   Java Home: %JAVA_HOME%
echo.

REM 2) Go to Cassandra bin directory
echo [2/3] Navigating to Cassandra bin...
cd /d C:\cassandra\apache-cassandra-3.11.17\bin

if not exist cassandra.ps1 (
    echo ERROR: cassandra.ps1 not found!
    echo Check if Cassandra is installed at: C:\cassandra\apache-cassandra-3.11.17
    pause
    exit /b 1
)

echo   Found: C:\cassandra\apache-cassandra-3.11.17\bin
echo.

REM 3) Start Cassandra using PowerShell
echo [3/3] Starting Cassandra...
echo   This will open a new window
echo   DO NOT CLOSE that window while using the system!
echo.

start "Cassandra Server" "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File ".\cassandra.ps1"

echo.
echo ============================================
echo Cassandra is starting...
echo ============================================
echo.
echo Wait 60 seconds for Cassandra to fully start
echo Then run: .\start.ps1 (to start Backend/Frontend)
echo.

REM Only pause if not running in auto mode
if not "%1"=="--auto" pause
