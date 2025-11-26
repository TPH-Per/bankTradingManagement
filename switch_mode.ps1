# ====================================================================
# Quick Mode Switcher
# ====================================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🔄 Mode Switcher" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Available modes:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. Development Mode" -ForegroundColor White
Write-Host "     - 1 Cassandra node" -ForegroundColor Gray
Write-Host "     - Full stack (Backend + Frontend)" -ForegroundColor Gray
Write-Host "     - RAM: ~3GB" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Fault Tolerance Demo" -ForegroundColor White
Write-Host "     - 3 Cassandra nodes (cluster)" -ForegroundColor Gray
Write-Host "     - Interactive demo" -ForegroundColor Gray
Write-Host "     - RAM: ~2GB" -ForegroundColor Gray
Write-Host ""

$choice = Read-Host "Select mode (1 or 2)"

Write-Host ""

if ($choice -eq "1") {
    Write-Host "🔧 Starting Development Mode..." -ForegroundColor Green
    Write-Host ""
    
    # Stop demo cluster if running
    Write-Host "  Checking for demo cluster..." -ForegroundColor Gray
    $demoRunning = docker ps --filter "name=bt-cassandra-node" --format "{{.Names}}" 2>$null
    
    if ($demoRunning) {
        Write-Host "  Stopping demo cluster..." -ForegroundColor Yellow
        docker-compose -f docker-compose.cassandra-cluster.yml down
        Start-Sleep -Seconds 5
    }
    
    # Start development
    Write-Host "  Starting development environment..." -ForegroundColor Yellow
    Write-Host ""
    .\start_all.ps1
    
}
elseif ($choice -eq "2") {
    Write-Host "🎓 Starting Fault Tolerance Demo..." -ForegroundColor Green
    Write-Host ""
    
    # Stop development if running
    Write-Host "  Checking for development services..." -ForegroundColor Gray
    $devRunning = docker ps --filter "name=bt-cassandra" --format "{{.Names}}" 2>$null | Where-Object { $_ -eq "bt-cassandra" }
    
    if ($devRunning) {
        Write-Host "  Stopping development services..." -ForegroundColor Yellow
        .\stop_all.ps1
        Start-Sleep -Seconds 5
    }
    
    # Start demo
    Write-Host "  Starting demo..." -ForegroundColor Yellow
    Write-Host ""
    .\demo_cassandra_fault_tolerance.ps1
    
}
else {
    Write-Host "❌ Invalid choice!" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}
