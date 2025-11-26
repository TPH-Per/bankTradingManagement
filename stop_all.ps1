# ====================================================================
# STOP ALL SERVICES - Bank Trading Management System
# ====================================================================

Write-Host "============================================" -ForegroundColor Red
Write-Host "🛑 Stopping ALL Services" -ForegroundColor Red
Write-Host "============================================" -ForegroundColor Red
Write-Host ""

# ====================================================================
# Step 1: Stop Docker Services
# ====================================================================
Write-Host "[1/2] Stopping Docker services..." -ForegroundColor Yellow

try {
    docker-compose down
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Docker services stopped" -ForegroundColor Green
    }
    else {
        Write-Host "  ⚠️  Some services may not have been running" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "  ❌ Error stopping Docker services: $_" -ForegroundColor Red
}

Write-Host ""

# ====================================================================
# Step 2: Info about Backend/Frontend
# ====================================================================
Write-Host "[2/2] Backend and Frontend..." -ForegroundColor Yellow
Write-Host "  ℹ️  Backend and Frontend run in separate windows" -ForegroundColor Gray
Write-Host "  📝 To stop them: Press Ctrl+C in their respective windows" -ForegroundColor Gray
Write-Host ""

# Try to find and close Python/Node processes (optional)
Write-Host "  Checking for running processes..." -ForegroundColor Gray

# Find uvicorn processes (Backend)
$uvicornProcesses = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*uvicorn*"
}

if ($uvicornProcesses) {
    Write-Host "  Found $($uvicornProcesses.Count) backend process(es)" -ForegroundColor Yellow
    $confirm = Read-Host "  Do you want to stop backend processes? (y/n)"
    if ($confirm -eq 'y') {
        $uvicornProcesses | Stop-Process -Force
        Write-Host "  ✅ Backend processes stopped" -ForegroundColor Green
    }
}

# Find node processes (Frontend)
$nodeProcesses = Get-Process -Name node -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*vite*" -or $_.Path -like "*frontend*"
}

if ($nodeProcesses) {
    Write-Host "  Found $($nodeProcesses.Count) frontend process(es)" -ForegroundColor Yellow
    $confirm = Read-Host "  Do you want to stop frontend processes? (y/n)"
    if ($confirm -eq 'y') {
        $nodeProcesses | Stop-Process -Force
        Write-Host "  ✅ Frontend processes stopped" -ForegroundColor Green
    }
}

# ====================================================================
# Summary
# ====================================================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "✅ Stop Complete" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Docker services stopped:" -ForegroundColor White
Write-Host "  • Cassandra" -ForegroundColor Gray
Write-Host "  • Redis" -ForegroundColor Gray
Write-Host "  • HDFS" -ForegroundColor Gray
Write-Host ""
Write-Host "If backend/frontend windows are still open:" -ForegroundColor Yellow
Write-Host "  Press Ctrl+C in each window to stop them" -ForegroundColor Gray
Write-Host ""

Read-Host "Press Enter to close"
