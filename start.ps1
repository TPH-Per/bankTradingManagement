# ====================================================================
# START ALL - Smart Startup Script
# ====================================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "[START] Bank Trading System" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Get project root
$projectRoot = Get-Location

# ====================================================================
# Step 1: Check and Start Cassandra (Native)
# ====================================================================
Write-Host "[1/4] Checking Cassandra..." -ForegroundColor Yellow

$cassandraRunning = Test-NetConnection -ComputerName localhost -Port 9042 -WarningAction SilentlyContinue -InformationLevel Quiet 2>$null

if ($cassandraRunning) {
    Write-Host "  [OK] Cassandra is already running (port 9042)" -ForegroundColor Green
}
else {
    Write-Host "  [WARNING] Cassandra not running, starting now..." -ForegroundColor Yellow
    
    # Start Cassandra using the dedicated batch script
    $cassandraBat = Join-Path $projectRoot "start_cassandra.bat"
    Start-Process cmd -ArgumentList "/c", "start", "cmd", "/k", $cassandraBat, "--auto" -WindowStyle Hidden
    
    Write-Host "  [WAIT] Waiting 60 seconds for Cassandra to initialize..." -ForegroundColor Gray
    Start-Sleep -Seconds 60
    
    # Verify Cassandra started
    $cassandraRunning = Test-NetConnection -ComputerName localhost -Port 9042 -WarningAction SilentlyContinue -InformationLevel Quiet 2>$null
    
    if ($cassandraRunning) {
        Write-Host "  [OK] Cassandra started successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "  [ERROR] Cassandra failed to start!" -ForegroundColor Red
        Write-Host "  Please check C:\cassandra installation" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""

# ====================================================================
# Step 2: Start Docker Services (HDFS + Redis)
# ====================================================================
Write-Host "[2/4] Starting Docker services..." -ForegroundColor Yellow
Write-Host "  Services: HDFS + Redis" -ForegroundColor Gray

try {
    # Start HDFS NameNode & DataNode
    Write-Host "  Starting HDFS NameNode..." -ForegroundColor Gray
    docker start bt-hdfs-namenode 2>&1 | Out-Null
    
    Start-Sleep -Seconds 3
    
    Write-Host "  Starting HDFS DataNode..." -ForegroundColor Gray
    docker start bt-hdfs-datanode 2>&1 | Out-Null
    
    # Start Redis
    Write-Host "  Starting Redis..." -ForegroundColor Gray
    docker start bt-redis 2>&1 | Out-Null
    
    Write-Host "  [OK] Docker services started!" -ForegroundColor Green
}
catch {
    Write-Host "  [WARNING] Some services may already be running (OK)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  [WAIT] Waiting for services to initialize (15 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 15

# Initialize HDFS directories
Write-Host ""
Write-Host "  Initializing HDFS directories..." -ForegroundColor Gray
try {
    docker exec bt-hdfs-namenode hdfs dfs -mkdir -p /banktrading/data /banktrading/ml /banktrading/reports /banktrading/transactions /banktrading/customers 2>&1 | Out-Null
    docker exec bt-hdfs-namenode hdfs dfs -chmod -R 777 /banktrading 2>&1 | Out-Null
    Write-Host "  [OK] HDFS directories initialized" -ForegroundColor Green
}
catch {
    Write-Host "  [WARNING] HDFS directories may already exist (OK)" -ForegroundColor Yellow
}

# ====================================================================
# Step 3: Verify All Services
# ====================================================================
Write-Host ""
Write-Host "[3/4] Verifying all services..." -ForegroundColor Yellow

# Check Cassandra
$cassandraOK = Test-NetConnection -ComputerName localhost -Port 9042 -WarningAction SilentlyContinue -InformationLevel Quiet 2>$null
if ($cassandraOK) {
    Write-Host "  [OK] Cassandra (port 9042)" -ForegroundColor Green
}
else {
    Write-Host "  [ERROR] Cassandra (port 9042)" -ForegroundColor Red
}

# Check Redis
$redisOK = Test-NetConnection -ComputerName localhost -Port 6379 -WarningAction SilentlyContinue -InformationLevel Quiet 2>$null
if ($redisOK) {
    Write-Host "  [OK] Redis (port 6379)" -ForegroundColor Green
}
else {
    Write-Host "  [WARNING] Redis (port 6379)" -ForegroundColor Yellow
}

# Check HDFS
$hdfsOK = Test-NetConnection -ComputerName localhost -Port 9870 -WarningAction SilentlyContinue -InformationLevel Quiet 2>$null
if ($hdfsOK) {
    Write-Host "  [OK] HDFS NameNode (port 9870)" -ForegroundColor Green
}
else {
    Write-Host "  [WARNING] HDFS NameNode (port 9870)" -ForegroundColor Yellow
}

Write-Host ""

# ====================================================================
# Step 4: Start Backend & Frontend
# ====================================================================
Write-Host "[4/4] Starting application servers..." -ForegroundColor Yellow

# Start Backend
$backendPath = Join-Path $projectRoot "BACKEND"
$backendCmd = @"
Write-Host '============================================' -ForegroundColor Cyan
Write-Host 'Backend Server Starting...' -ForegroundColor Cyan
Write-Host '============================================' -ForegroundColor Cyan
Write-Host ''
Set-Location '$backendPath'
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd
Write-Host "  [OK] Backend starting in new window (port 8000)" -ForegroundColor Green

Start-Sleep -Seconds 2

# Start Frontend
$frontendPath = Join-Path $projectRoot "FRONTEND"
$frontendCmd = @"
Write-Host '============================================' -ForegroundColor Cyan
Write-Host 'Frontend Server Starting...' -ForegroundColor Cyan
Write-Host '============================================' -ForegroundColor Cyan
Write-Host ''
Set-Location '$frontendPath'
npm run dev
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd
Write-Host "  [OK] Frontend starting in new window (port 5173)" -ForegroundColor Green

# ====================================================================
# Summary
# ====================================================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "[SUCCESS] System Started!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

Write-Host "Services:" -ForegroundColor White
Write-Host "  • Cassandra (native):  localhost:9042" -ForegroundColor Gray
Write-Host "  • Redis (Docker):      localhost:6379" -ForegroundColor Gray
Write-Host "  • HDFS (Docker):       localhost:9870" -ForegroundColor Gray
Write-Host "  • Backend:             http://localhost:8000" -ForegroundColor Gray
Write-Host "  • Frontend:            http://localhost:5173" -ForegroundColor Gray
Write-Host ""

Write-Host "Open: http://localhost:5173" -ForegroundColor Cyan
Write-Host ""

Write-Host "[WAIT] Waiting for complete initialization (10 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "[SUCCESS] All set! System is ready!" -ForegroundColor Green
Write-Host ""

Read-Host "Press Enter to close this window"
