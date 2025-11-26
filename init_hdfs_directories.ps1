# ====================================================================
# Initialize HDFS Directories
# ====================================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "[INIT] Initializing HDFS Directories" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check HDFS
Write-Host "[1/2] Checking HDFS..." -ForegroundColor Yellow
$hdfsRunning = docker ps --filter "name=bt-hdfs-namenode" --format "{{.Names}}" 2>$null

if (-not $hdfsRunning) {
    Write-Host "  [ERROR] HDFS is not running!" -ForegroundColor Red
    Write-Host "  Start HDFS first: docker-compose up -d hdfs-namenode hdfs-datanode" -ForegroundColor Yellow
    exit 1
}

Write-Host "  [OK] HDFS is running" -ForegroundColor Green
Write-Host ""

# Create directories
Write-Host "[2/2] Creating directories..." -ForegroundColor Yellow

$directories = @(
    "/banktrading",
    "/banktrading/data",
    "/banktrading/ml",
    "/banktrading/reports",
    "/banktrading/transactions",
    "/banktrading/customers"
)

foreach ($dir in $directories) {
    Write-Host "  Creating $dir..." -ForegroundColor Gray
    docker exec bt-hdfs-namenode hdfs dfs -mkdir -p $dir 2>$null
}

Write-Host "  [OK] Directories created" -ForegroundColor Green
Write-Host ""

# Set permissions
Write-Host "  Setting permissions..." -ForegroundColor Gray
docker exec bt-hdfs-namenode hdfs dfs -chmod -R 777 /banktrading 2>$null
Write-Host "  [OK] Permissions set" -ForegroundColor Green
Write-Host ""

# Verify
Write-Host "============================================" -ForegroundColor Green
Write-Host "[SUCCESS] HDFS Initialization Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

Write-Host "Directory structure:" -ForegroundColor White
docker exec bt-hdfs-namenode hdfs dfs -ls -R /banktrading

Write-Host ""
Read-Host "Press Enter to close"
