# run_spark_etl.ps1
# Script to run Spark ETL on Docker

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "SPARK ETL ON DOCKER" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

$ProjectRoot = "C:\temp\bankTradingManagement-master\bankTradingManagement-master"

# Step 1: Ensure network exists
Write-Host "`n[1/4] Ensuring Docker network exists..." -ForegroundColor Yellow
docker network create banktrading_network 2>$null

# Step 2: Start Spark cluster
Write-Host "[2/4] Starting Spark cluster..." -ForegroundColor Yellow
docker-compose -f "$ProjectRoot\docker-compose.spark.yml" up -d

# Step 3: Wait for Spark to be ready
Write-Host "[3/4] Waiting for Spark Master to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check Spark Master status
$sparkReady = $false
for ($i = 0; $i -lt 6; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            $sparkReady = $true
            break
        }
    }
    catch {
        Write-Host "  Waiting... ($($i+1)/6)" -ForegroundColor Gray
        Start-Sleep -Seconds 5
    }
}

if (-not $sparkReady) {
    Write-Host "[ERROR] Spark Master not ready!" -ForegroundColor Red
    exit 1
}

Write-Host "  Spark Master is ready!" -ForegroundColor Green

# Step 4: Run ETL job
Write-Host "[4/4] Running Spark ETL job..." -ForegroundColor Yellow
Write-Host ""

docker exec bt-spark-master spark-submit `
    --master local[*] `
    /app/spark-etl.py `
    --mode local `
    --local-base /data

$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "=" * 60 -ForegroundColor Green
    Write-Host "SPARK ETL COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "=" * 60 -ForegroundColor Green
    Write-Host ""
    Write-Host "Results saved to: BACKEND/data/cash_daily_train_realistic.csv"
    Write-Host "Spark Web UI: http://localhost:8080"
}
else {
    Write-Host "=" * 60 -ForegroundColor Red
    Write-Host "SPARK ETL FAILED (exit code: $exitCode)" -ForegroundColor Red
    Write-Host "=" * 60 -ForegroundColor Red
}

# Optional: Stop Spark cluster after job
# Write-Host "`nStopping Spark cluster..."
# docker-compose -f "$ProjectRoot\docker-compose.spark.yml" down
