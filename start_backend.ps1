# ====================================================================
# Start Backend with Auto Docker Services (PowerShell)
# ====================================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Starting Bank Trading Backend" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Docker
Write-Host "[Step 1/4] Checking Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker not running"
    }
    Write-Host "  ✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Docker Desktop is not running!" -ForegroundColor Red
    Write-Host "  Please start Docker Desktop first." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Step 2: Start Docker Services
Write-Host "[Step 2/4] Starting Docker services..." -ForegroundColor Yellow
Write-Host "  Starting Redis, HDFS..." -ForegroundColor Gray

try {
    docker-compose up -d redis hdfs-namenode hdfs-datanode 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Docker services started" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Docker services may already be running" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ⚠️  Could not start Docker services: $_" -ForegroundColor Yellow
    Write-Host "  Continuing anyway..." -ForegroundColor Gray
}

Write-Host ""
Write-Host "  Waiting 10 seconds for services to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 10

# Step 3: Check services
Write-Host ""
Write-Host "[Step 3/4] Verifying services..." -ForegroundColor Yellow

# Check Redis
$redisRunning = Test-NetConnection -ComputerName localhost -Port 6379 -WarningAction SilentlyContinue -InformationLevel Quiet
if ($redisRunning) {
    Write-Host "  ✅ Redis (port 6379)" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Redis not accessible yet" -ForegroundColor Yellow
}

# Check HDFS NameNode
$hdfsRunning = Test-NetConnection -ComputerName localhost -Port 9870 -WarningAction SilentlyContinue -InformationLevel Quiet
if ($hdfsRunning) {
    Write-Host "  ✅ HDFS NameNode (port 9870)" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  HDFS not accessible yet" -ForegroundColor Yellow
}

# Check Cassandra
$cassRunning = Test-NetConnection -ComputerName localhost -Port 9042 -WarningAction SilentlyContinue -InformationLevel Quiet
if ($cassRunning) {
    Write-Host "  ✅ Cassandra (port 9042)" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Cassandra not accessible" -ForegroundColor Yellow
}

Write-Host ""

# Step 4: Start Backend
Write-Host "[Step 4/4] Starting Backend..." -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Backend will start now..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Change to BACKEND directory
Set-Location -Path "BACKEND"

# Start backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
